import cv2
import numpy as np


class PageReconstructor:
    """Combine frames into a single image by stitching and sharpening."""

    def reconstruct(self, frames, masks=None):
        if not frames:
            raise ValueError("No frames provided")

        # Use the first frame as the reference
        reference_frame = frames[0]
        stitched_image = reference_frame.copy()

        # Create a SIFT detector
        sift = cv2.SIFT_create()

        # Find keypoints and descriptors for the reference frame
        kp1, des1 = sift.detectAndCompute(cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY), None)

        for i in range(1, len(frames)):
            frame = frames[i]

            # Find keypoints and descriptors for the current frame
            kp2, des2 = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)

            # Match descriptors
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            if len(good_matches) > 10:
                # Get the coordinates of the good matches
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Find the homography
                M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

                # Warp the current frame to the reference frame
                h, w, _ = reference_frame.shape
                warped_frame = cv2.warpPerspective(frame, M, (w, h))

                # Blend the warped frame with the stitched image
                stitched_image = cv2.addWeighted(stitched_image, 0.5, warped_frame, 0.5, 0)

        # Sharpen the stitched image
        sharpened_image = self.sharpen(stitched_image)

        return sharpened_image

    def sharpen(self, image):
        """Sharpen an image using a sharpening kernel."""
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)