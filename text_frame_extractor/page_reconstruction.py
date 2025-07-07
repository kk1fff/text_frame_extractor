import cv2
import numpy as np
from typing import List, Tuple

class PageReconstructor:
    """
    Combine frames into a single, high-quality image.

    This class implements an advanced image reconstruction algorithm that stitches
    multiple frames together by selecting the sharpest available pixels from all
    provided frames. It aligns the images, builds a composite using the
    best parts of each frame, and then performs a final, targeted sharpening
    pass only on the areas that remain blurry.
    """

    def reconstruct(self, frame_mask_pairs: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Reconstructs a single image from a list of (frame, mask) tuples.
        
        Args:
            frame_mask_pairs: List of tuples, each containing (frame, mask)
                             where mask is a binary mask indicating valid pixels
        """
        if not frame_mask_pairs:
            raise ValueError("No frame-mask pairs provided")

        if len(frame_mask_pairs) == 1:
            frame, mask = frame_mask_pairs[0]
            # Apply mask to return only valid pixels
            return self._apply_mask(frame, mask)

        # 1. Align Frames
        aligned_pairs = self._align_frames(frame_mask_pairs)

        # 2. Build Composite with "Best Pixels" using masks
        composite_image, sharpness_map = self._build_composite_with_masks(aligned_pairs)

        # 3. Targeted Sharpening as a Last Resort
        final_image = self._selective_sharpen(composite_image, sharpness_map)

        return final_image

    def _apply_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply mask to frame, setting non-masked pixels to white.
        """
        if mask is None:
            return frame
        
        # Ensure mask is binary
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Create white background
        result = np.ones_like(frame) * 255
        
        # Apply mask: only keep pixels where mask is 1
        for c in range(frame.shape[2]):
            result[:, :, c] = np.where(binary_mask == 1, frame[:, :, c], result[:, :, c])
        
        return result

    def _align_frames(self, frame_mask_pairs: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Aligns all frames to the perspective of the first frame.
        Returns list of (aligned_frame, aligned_mask) tuples.
        """
        sift = cv2.SIFT_create()
        reference_frame, reference_mask = frame_mask_pairs[0]
        kp1, des1 = sift.detectAndCompute(cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY), None)

        aligned_pairs = [(reference_frame, reference_mask)]
        for i in range(1, len(frame_mask_pairs)):
            frame, mask = frame_mask_pairs[i]
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
                    warped_mask = cv2.warpPerspective(mask, M, (w, h))
                    aligned_pairs.append((warped_frame, warped_mask))
                    continue
            # If not enough matches or homography fails, append frame as-is
            aligned_pairs.append((frame, mask))
        return aligned_pairs

    def _calculate_sharpness_map(self, image: np.ndarray, mask: np.ndarray = None):
        """
        Calculates a sharpness score for each pixel using the variance of the Laplacian.
        Only considers pixels within the mask.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise before calculating Laplacian
        blurred_gray = cv2.GaussianBlur(gray, (7, 7), 0)
        laplacian = cv2.Laplacian(blurred_gray, cv2.CV_64F)
        # Use the variance of the Laplacian for sharpness over a small window
        sharpness = cv2.filter2D(laplacian**2, -1, np.ones((3,3), np.float32)/9.0)
        
        # If mask is provided, set non-masked pixels to 0 sharpness
        if mask is not None:
            binary_mask = (mask > 0).astype(np.float64)
            sharpness = sharpness * binary_mask
        
        return sharpness

    def _build_composite_with_masks(self, frame_mask_pairs: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Builds a composite image by selecting the sharpest pixel from all frames,
        only considering pixels within their respective masks.
        """
        if not frame_mask_pairs:
            return None, None

        # Extract frames and masks
        frames = [pair[0] for pair in frame_mask_pairs]
        masks = [pair[1] for pair in frame_mask_pairs]
        
        # Calculate sharpness maps for all frames, considering masks
        sharpness_maps = []
        for frame, mask in frame_mask_pairs:
            sharpness_map = self._calculate_sharpness_map(frame, mask)
            sharpness_maps.append(sharpness_map)
        
        sharpness_maps = np.array(sharpness_maps)

        # Find the index of the frame with the maximum sharpness for each pixel
        # Only consider pixels where at least one mask is valid
        combined_mask = np.max(masks, axis=0) > 0
        best_frame_indices = np.argmax(sharpness_maps, axis=0)

        # Initialize composite image with white background
        composite_image = np.ones_like(frames[0], dtype=np.uint8) * 255

        # Construct the composite image by picking pixels from the best frame
        # Only where masks are valid
        for i in range(frames[0].shape[0]):
            for j in range(frames[0].shape[1]):
                if combined_mask[i, j]:
                    best_frame_idx = best_frame_indices[i, j]
                    # Only use pixel if it's within the mask of the best frame
                    if masks[best_frame_idx][i, j] > 0:
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
