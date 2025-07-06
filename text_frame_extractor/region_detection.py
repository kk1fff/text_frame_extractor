import cv2
import numpy as np

class RegionDetector:
    """Detect target reading region."""

    def detect(self, frame):
        """Detect the main document region in the frame using a robust approach focusing on rectangular shapes.
        This version uses bilateral filtering, adaptive thresholding, and a scoring system for contour selection.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Bilateral filter for noise reduction while preserving edges
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Morphological operations to clean up the image
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        best_bbox = None
        max_score = -1.0 # Using a scoring system to find the best candidate

        height, width = frame.shape[:2]
        frame_area = height * width

        for c in contours:
            # Approximate the contour to a polygon
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # Only consider contours with 4 vertices (potential rectangles)
            if len(approx) == 4:
                area = cv2.contourArea(approx)

                # Filter by area: ignore very small or very large contours
                if area < frame_area * 0.01 or area > frame_area * 0.99: 
                    continue

                # Calculate bounding box and aspect ratio
                x, y, w, h = cv2.boundingRect(approx)
                if w == 0 or h == 0:
                    continue
                aspect_ratio = float(w) / h

                # Filter by aspect ratio (typical for book pages, allowing for some tilt)
                # A wider range to be more forgiving, but still prioritize closer to 1.0
                if not (0.3 < aspect_ratio < 3.0): 
                    continue

                # Filter by solidity (how "solid" the object is)
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    continue
                solidity = float(area) / hull_area
                if solidity < 0.7: # Adjust threshold for solidity
                    continue

                # Check for convexity
                if not cv2.isContourConvex(approx):
                    continue

                # Calculate a score for the candidate
                # Prioritize larger area, higher solidity, and aspect ratio closer to 1.0
                # The 0.1 is added to the denominator to prevent division by zero if aspect_ratio is exactly 1.0
                score = area * solidity / (abs(1.0 - aspect_ratio) + 0.1)

                if score > max_score:
                    max_score = score
                    best_bbox = (x, y, w, h)

        if best_bbox:
            return best_bbox
        else:
            # Fallback: if no suitable region is found, return the full frame
            return (0, 0, width, height)