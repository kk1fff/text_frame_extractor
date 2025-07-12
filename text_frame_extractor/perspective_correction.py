import cv2
import numpy as np
from typing import List, Tuple, Optional


class PerspectiveCorrector:
    """
    Corrects 3D perspective distortion in document images.
    
    Detects page boundaries and vanishing points to rectify perspective
    distortion caused by camera viewpoint, making text regions more
    suitable for alignment and OCR.
    """
    
    def __init__(self):
        self.min_line_length = 50
        self.max_line_gap = 10
        self.hough_threshold = 100
        
    def correct_perspective(self, frame: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Correct perspective distortion in a frame.
        
        Args:
            frame: Input image
            mask: Binary mask indicating text regions
            
        Returns:
            Tuple of (rectified_frame, rectified_mask, rectification_matrix)
        """
        # Detect text regions to guide page boundary detection
        text_regions = self._detect_text_blocks(frame, mask)
        
        # Estimate page corners from text layout
        page_corners = self._estimate_page_corners(frame, text_regions)
        
        if page_corners is None:
            # No perspective correction needed/possible
            return frame, mask, None
            
        # Find vanishing points from page edges
        vanishing_points = self._detect_vanishing_points(frame, page_corners)
        
        # Compute rectification matrix
        rectification_matrix = self._compute_rectification(frame.shape, page_corners, vanishing_points)
        
        if rectification_matrix is None:
            return frame, mask, None
            
        # Apply rectification
        h, w = frame.shape[:2]
        rectified_frame = cv2.warpPerspective(frame, rectification_matrix, (w, h))
        rectified_mask = cv2.warpPerspective(mask, rectification_matrix, (w, h))
        
        return rectified_frame, rectified_mask, rectification_matrix
    
    def _detect_text_blocks(self, frame: np.ndarray, mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect rectangular text blocks to guide page boundary estimation.
        
        Returns:
            List of (x, y, w, h) bounding rectangles
        """
        # Use mask to focus on text regions
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        
        # Find contours in text regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_blocks = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                text_blocks.append((x, y, w, h))
                
        return text_blocks
    
    def _estimate_page_corners(self, frame: np.ndarray, text_regions: List[Tuple[int, int, int, int]]) -> Optional[np.ndarray]:
        """
        Estimate page corners from text block layout.
        
        Returns:
            4x2 array of corner points [top-left, top-right, bottom-right, bottom-left]
            or None if estimation fails
        """
        if not text_regions:
            return None
            
        # Find bounding box of all text regions
        all_x = []
        all_y = []
        for x, y, w, h in text_regions:
            all_x.extend([x, x + w])
            all_y.extend([y, y + h])
            
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        # Add margin around text to estimate page boundaries
        margin_x = int((max_x - min_x) * 0.1)
        margin_y = int((max_y - min_y) * 0.1)
        
        page_corners = np.array([
            [max(0, min_x - margin_x), max(0, min_y - margin_y)],           # top-left
            [min(frame.shape[1], max_x + margin_x), max(0, min_y - margin_y)],     # top-right
            [min(frame.shape[1], max_x + margin_x), min(frame.shape[0], max_y + margin_y)],   # bottom-right
            [max(0, min_x - margin_x), min(frame.shape[0], max_y + margin_y)]      # bottom-left
        ], dtype=np.float32)
        
        return page_corners
    
    def _detect_vanishing_points(self, frame: np.ndarray, page_corners: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Detect vanishing points from page edges.
        
        Returns:
            Tuple of (horizontal_vp, vertical_vp) or None if detection fails
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                               threshold=self.hough_threshold,
                               minLineLength=self.min_line_length,
                               maxLineGap=self.max_line_gap)
        
        if lines is None or len(lines) < 4:
            return None
            
        # Separate horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            if abs(angle) < 30 or abs(angle) > 150:  # Horizontal-ish
                horizontal_lines.append((x1, y1, x2, y2))
            elif 60 < abs(angle) < 120:  # Vertical-ish
                vertical_lines.append((x1, y1, x2, y2))
        
        # Find vanishing points by intersecting parallel lines
        h_vp = self._find_vanishing_point(horizontal_lines)
        v_vp = self._find_vanishing_point(vertical_lines)
        
        return h_vp, v_vp
    
    def _find_vanishing_point(self, lines: List[Tuple[int, int, int, int]]) -> Optional[np.ndarray]:
        """
        Find vanishing point from a set of parallel lines.
        
        Returns:
            Vanishing point as [x, y] or None if not found
        """
        if len(lines) < 2:
            return None
            
        intersections = []
        
        # Find intersections between all pairs of lines
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                intersection = self._line_intersection(lines[i], lines[j])
                if intersection is not None:
                    intersections.append(intersection)
        
        if not intersections:
            return None
            
        # Use median to find robust vanishing point
        intersections = np.array(intersections)
        vp = np.median(intersections, axis=0)
        
        return vp
    
    def _line_intersection(self, line1: Tuple[int, int, int, int], line2: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Find intersection point of two lines.
        
        Returns:
            Intersection point as [x, y] or None if lines are parallel
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-6:  # Lines are parallel
            return None
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        intersection_x = x1 + t * (x2 - x1)
        intersection_y = y1 + t * (y2 - y1)
        
        return np.array([intersection_x, intersection_y])
    
    def _compute_rectification(self, frame_shape: Tuple[int, int, int], 
                             page_corners: np.ndarray, 
                             vanishing_points: Tuple[Optional[np.ndarray], Optional[np.ndarray]]) -> Optional[np.ndarray]:
        """
        Compute rectification matrix to correct perspective.
        
        Returns:
            3x3 homography matrix or None if computation fails
        """
        h_vp, v_vp = vanishing_points
        
        if h_vp is None and v_vp is None:
            return None
            
        # Define target rectangle (frontal view)
        h, w = frame_shape[:2]
        target_corners = np.array([
            [0, 0],
            [w, 0], 
            [w, h],
            [0, h]
        ], dtype=np.float32)
        
        # If we have vanishing points, use them to improve corner estimation
        if h_vp is not None or v_vp is not None:
            # For now, use simple 4-point perspective transform
            # Could be enhanced with vanishing point constraints
            pass
            
        # Compute homography from distorted corners to target rectangle
        try:
            rectification_matrix = cv2.getPerspectiveTransform(page_corners, target_corners)
            return rectification_matrix
        except cv2.error:
            return None