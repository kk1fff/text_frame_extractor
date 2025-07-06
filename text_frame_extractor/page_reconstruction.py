import cv2
import numpy as np

class PageReconstructor:
    """
    Combine frames into a single, high-quality image.

    This class implements an advanced image reconstruction algorithm that stitches
    multiple frames together by selecting the sharpest available pixels from all
    provided frames. It aligns the images, builds a composite using the
    best parts of each frame, and then performs a final, targeted sharpening
    pass only on the areas that remain blurry.
    """

    def reconstruct(self, frames, masks=None):
        """
        Reconstructs a single image from a list of frames.
        """
        if not frames:
            raise ValueError("No frames provided")

        if len(frames) == 1:
            return frames[0]

        # 1. Align Frames
        aligned_frames = self._align_frames(frames)

        # 2. Build Composite with "Best Pixels"
        composite_image, sharpness_map = self._build_composite(aligned_frames)

        # 3. Targeted Sharpening as a Last Resort
        final_image = self._selective_sharpen(composite_image, sharpness_map)

        return final_image

    def _align_frames(self, frames):
        """
        Aligns all frames to the perspective of the first frame.
        """
        sift = cv2.SIFT_create()
        reference_frame = frames[0]
        kp1, des1 = sift.detectAndCompute(cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY), None)

        aligned_frames = [reference_frame]
        for i in range(1, len(frames)):
            frame = frames[i]
            kp2, des2 = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

            if len(good_matches) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    h, w, _ = reference_frame.shape
                    warped_frame = cv2.warpPerspective(frame, M, (w, h))
                    aligned_frames.append(warped_frame)
                    continue
            # If not enough matches or homography fails, append frame as-is
            aligned_frames.append(frame)
        return aligned_frames

    def _calculate_sharpness_map(self, image):
        """
        Calculates a sharpness score for each pixel using the variance of the Laplacian.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise before calculating Laplacian
        blurred_gray = cv2.GaussianBlur(gray, (7, 7), 0)
        laplacian = cv2.Laplacian(blurred_gray, cv2.CV_64F)
        # Use the variance of the Laplacian for sharpness over a small window
        sharpness = cv2.filter2D(laplacian**2, -1, np.ones((3,3), np.float32)/9.0)
        return sharpness

    def _build_composite(self, frames):
        """
        Builds a composite image by selecting the sharpest pixel from all frames.
        """
        if not frames:
            return None, None

        # Calculate sharpness maps for all frames
        sharpness_maps = np.array([self._calculate_sharpness_map(frame) for frame in frames])

        # Find the index of the frame with the maximum sharpness for each pixel
        best_frame_indices = np.argmax(sharpness_maps, axis=0)

        # Initialize composite image
        composite_image = np.zeros_like(frames[0], dtype=np.uint8)

        # Construct the composite image by picking pixels from the best frame
        for i in range(frames[0].shape[0]):
            for j in range(frames[0].shape[1]):
                best_frame_idx = best_frame_indices[i, j]
                composite_image[i, j] = frames[best_frame_idx][i, j]

        # Recalculate the sharpness map for the final composite image
        final_sharpness_map = self._calculate_sharpness_map(composite_image)
        return composite_image, final_sharpness_map

    def _selective_sharpen(self, image, sharpness_map, blur_threshold=50):
        """
        Applies a sharpening filter only to blurry regions of the image.
        """
        # Find regions that are still blurry
        # The blur_threshold might need to be tuned based on image content
        blur_mask = (sharpness_map < blur_threshold).astype(np.uint8)
        
        if np.sum(blur_mask) == 0:
            return image # Nothing to sharpen

        # Create a sharpened version of the entire image
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened_image = cv2.filter2D(image, -1, kernel)

        # Combine the original image with the sharpened regions
        final_image = image.copy()
        # Expand the mask slightly to blend the edges
        blur_mask = cv2.dilate(blur_mask, np.ones((3,3), np.uint8), iterations=1)
        
        for c in range(3):
            final_image[:,:,c] = np.where(blur_mask == 1, sharpened_image[:,:,c], final_image[:,:,c])

        return final_image
