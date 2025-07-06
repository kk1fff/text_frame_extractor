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
        return aligned_frames

    def _calculate_sharpness_map(self, image):
        """
        Calculates a sharpness score for each pixel using the variance of the Laplacian.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise before calculating Laplacian
        blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        laplacian = cv2.Laplacian(blurred_gray, cv2.CV_64F)
        # Use the squared Laplacian directly for sharpness
        sharpness = laplacian**2
        return sharpness

    def _build_composite(self, frames):
        """
        Builds a composite image by selecting the sharpest pixel from all frames.
        """
        reference_frame = frames[0]
        composite_image = np.zeros_like(reference_frame, dtype=np.uint8)
        sharpness_map = np.full(reference_frame.shape[:2], -np.inf, dtype=np.float64)

        for frame in frames:
            
            # Create a mask for non-black pixels (valid content) in the current frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            valid_pixels_mask = gray_frame > 5  # Changed threshold from 0 to 5

            current_sharpness = self._calculate_sharpness_map(frame)
            
            # Only consider valid pixels for sharpness comparison
            comparison_mask = (current_sharpness > sharpness_map) & valid_pixels_mask
            
            # Update the composite image and the sharpness map
            for c in range(3):
                composite_image[:,:,c][comparison_mask] = frame[:,:,c][comparison_mask]
            
            sharpness_map[comparison_mask] = current_sharpness[comparison_mask]

        return composite_image, sharpness_map

    def _selective_sharpen(self, image, sharpness_map, blur_threshold=5):
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
        blur_mask = cv2.dilate(blur_mask, np.ones((1,1), np.uint8), iterations=1)
        
        for c in range(3):
            final_image[:,:,c] = np.where(blur_mask == 1, sharpened_image[:,:,c], final_image[:,:,c])

        return final_image
