import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from .ocr import OCR


class AlignmentQualityAssessor:
    """
    Multi-metric assessment of alignment quality between frames.
    
    Evaluates alignment using feature matching, text region overlap,
    photometric consistency, baseline alignment, and OCR confidence.
    """
    
    def __init__(self):
        self.ocr = OCR()
        self.sift = cv2.SIFT_create(nfeatures=500)
        
    def assess_alignment_quality(self, ref_frame: np.ndarray, target_frame: np.ndarray, 
                               transform: np.ndarray, ref_mask: np.ndarray, 
                               target_mask: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Comprehensive alignment quality assessment.
        
        Args:
            ref_frame: Reference frame
            target_frame: Target frame to be aligned
            transform: Homography matrix for alignment
            ref_mask: Reference frame text mask
            target_mask: Target frame text mask
            
        Returns:
            Tuple of (overall_quality_score, individual_scores_dict)
        """
        scores = {}
        
        # Metric 1: Feature matching quality
        scores['feature_match'] = self._compute_feature_match_score(ref_frame, target_frame, transform)
        
        # Metric 2: Text region overlap
        scores['text_overlap'] = self._compute_text_region_overlap(ref_mask, target_mask, transform)
        
        # Metric 3: Photometric consistency in text areas
        scores['photometric'] = self._compute_text_photometric_error(ref_frame, target_frame, transform, ref_mask)
        
        # Metric 4: Text line alignment
        scores['baseline_align'] = self._compute_baseline_alignment_score(ref_frame, target_frame, transform)
        
        # Metric 5: OCR confidence improvement
        scores['ocr_confidence'] = self._compute_ocr_confidence_gain(ref_frame, target_frame, transform)
        
        # Weighted combination
        quality_score = (
            scores['feature_match'] * 0.25 +
            scores['text_overlap'] * 0.25 + 
            scores['photometric'] * 0.2 +
            scores['baseline_align'] * 0.2 +
            scores['ocr_confidence'] * 0.1
        )
        
        return quality_score, scores
    
    def _compute_feature_match_score(self, ref_frame: np.ndarray, target_frame: np.ndarray, 
                                   transform: np.ndarray) -> float:
        """
        Evaluate alignment quality based on feature matching.
        
        Returns:
            Score between 0 and 1 (higher is better)
        """
        ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
        
        # Extract SIFT features
        kp1, des1 = self.sift.detectAndCompute(ref_gray, None)
        kp2, des2 = self.sift.detectAndCompute(target_gray, None)
        
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return 0.0
        
        # Match features
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 8:
            return 0.0
        
        # Extract matched points
        ref_pts = np.array([kp1[m.queryIdx].pt for m in good_matches])
        target_pts = np.array([kp2[m.trainIdx].pt for m in good_matches])
        
        # Transform target points and compute reprojection error
        target_pts_homog = np.column_stack([target_pts, np.ones(len(target_pts))])
        transformed_pts_homog = (transform @ target_pts_homog.T).T
        transformed_pts = transformed_pts_homog[:, :2] / transformed_pts_homog[:, 2:3]
        
        # Compute reprojection errors
        errors = np.linalg.norm(ref_pts - transformed_pts, axis=1)
        mean_error = np.mean(errors)
        
        # Convert error to score (lower error = higher score)
        max_acceptable_error = 5.0
        score = max(0.0, 1.0 - mean_error / max_acceptable_error)
        
        return score
    
    def _compute_text_region_overlap(self, ref_mask: np.ndarray, target_mask: np.ndarray, 
                                   transform: np.ndarray) -> float:
        """
        Compute overlap between text regions after alignment.
        
        Returns:
            Overlap ratio between 0 and 1
        """
        # Transform target mask to reference coordinate system
        h, w = ref_mask.shape
        transformed_target_mask = cv2.warpPerspective(target_mask, transform, (w, h))
        
        # Compute intersection and union
        intersection = cv2.bitwise_and(ref_mask, transformed_target_mask)
        union = cv2.bitwise_or(ref_mask, transformed_target_mask)
        
        intersection_area = np.sum(intersection > 0)
        union_area = np.sum(union > 0)
        
        if union_area == 0:
            return 0.0
            
        overlap_ratio = intersection_area / union_area
        return overlap_ratio
    
    def _compute_text_photometric_error(self, ref_frame: np.ndarray, target_frame: np.ndarray, 
                                      transform: np.ndarray, ref_mask: np.ndarray) -> float:
        """
        Compute photometric consistency in text regions.
        
        Returns:
            Consistency score between 0 and 1 (higher is better)
        """
        # Transform target frame to reference coordinate system
        h, w = ref_frame.shape[:2]
        transformed_target = cv2.warpPerspective(target_frame, transform, (w, h))
        
        # Convert to grayscale for comparison
        ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(transformed_target, cv2.COLOR_BGR2GRAY)
        
        # Compute difference only in text regions
        diff = cv2.absdiff(ref_gray, target_gray)
        text_diff = cv2.bitwise_and(diff, diff, mask=ref_mask)
        
        # Compute mean difference in text regions
        text_pixels = ref_mask > 0
        if np.sum(text_pixels) == 0:
            return 0.0
            
        mean_diff = np.mean(text_diff[text_pixels])
        
        # Convert to score (lower difference = higher score)
        max_acceptable_diff = 50.0  # Max intensity difference
        score = max(0.0, 1.0 - mean_diff / max_acceptable_diff)
        
        return score
    
    def _compute_baseline_alignment_score(self, ref_frame: np.ndarray, target_frame: np.ndarray, 
                                        transform: np.ndarray) -> float:
        """
        Evaluate alignment of text baselines.
        
        Returns:
            Baseline alignment score between 0 and 1
        """
        # Convert to grayscale
        ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
        
        # Transform target frame
        h, w = ref_gray.shape
        transformed_target = cv2.warpPerspective(target_gray, transform, (w, h))
        
        # Detect horizontal lines (text baselines) in both images
        ref_lines = self._detect_horizontal_lines(ref_gray)
        target_lines = self._detect_horizontal_lines(transformed_target)
        
        if len(ref_lines) == 0 or len(target_lines) == 0:
            return 0.0
        
        # Compute alignment score based on line correspondence
        alignment_score = self._compute_line_alignment(ref_lines, target_lines)
        
        return alignment_score
    
    def _detect_horizontal_lines(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect horizontal lines (potential text baselines).
        
        Returns:
            List of line coordinates (x1, y1, x2, y2)
        """
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=10)
        
        if lines is None:
            return []
        
        # Filter for horizontal lines
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # Keep lines that are roughly horizontal (within 15 degrees)
            if abs(angle) < 15 or abs(angle) > 165:
                horizontal_lines.append((x1, y1, x2, y2))
        
        return horizontal_lines
    
    def _compute_line_alignment(self, ref_lines: List[Tuple[int, int, int, int]], 
                              target_lines: List[Tuple[int, int, int, int]]) -> float:
        """
        Compute alignment score between two sets of lines.
        
        Returns:
            Alignment score between 0 and 1
        """
        if not ref_lines or not target_lines:
            return 0.0
        
        # Extract y-coordinates of line centers
        ref_y_coords = [(y1 + y2) / 2 for x1, y1, x2, y2 in ref_lines]
        target_y_coords = [(y1 + y2) / 2 for x1, y1, x2, y2 in target_lines]
        
        # Find best matching between lines
        total_alignment = 0.0
        matched_lines = 0
        
        for ref_y in ref_y_coords:
            # Find closest line in target
            min_distance = float('inf')
            for target_y in target_y_coords:
                distance = abs(ref_y - target_y)
                min_distance = min(min_distance, distance)
            
            # Convert distance to alignment score
            max_acceptable_distance = 10.0
            if min_distance < max_acceptable_distance:
                alignment = 1.0 - min_distance / max_acceptable_distance
                total_alignment += alignment
                matched_lines += 1
        
        if matched_lines == 0:
            return 0.0
            
        return total_alignment / len(ref_lines)
    
    def _compute_ocr_confidence_gain(self, ref_frame: np.ndarray, target_frame: np.ndarray, 
                                   transform: np.ndarray) -> float:
        """
        Evaluate OCR confidence improvement after alignment.
        
        Returns:
            Confidence gain score between 0 and 1
        """
        try:
            # Transform target frame
            h, w = ref_frame.shape[:2]
            transformed_target = cv2.warpPerspective(target_frame, transform, (w, h))
            
            # Extract text from both frames (simplified - just check if OCR works)
            ref_text = self.ocr.extract_text(ref_frame)
            target_text = self.ocr.extract_text(transformed_target)
            
            # Simple heuristic: longer text usually means better OCR
            ref_length = len(ref_text.strip())
            target_length = len(target_text.strip())
            
            if ref_length == 0 and target_length == 0:
                return 0.5  # Neutral score if both fail
            
            if ref_length == 0:
                return 1.0 if target_length > 0 else 0.0
            
            # Score based on text length consistency
            length_ratio = min(target_length, ref_length) / max(target_length, ref_length)
            
            return length_ratio
            
        except Exception:
            # If OCR fails, return neutral score
            return 0.5