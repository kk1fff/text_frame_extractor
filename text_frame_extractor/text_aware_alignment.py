import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional


class TextAwareAligner:
    """
    Advanced alignment using text-specific features and hierarchical matching.
    
    Extracts text-aware features (baselines, corners, characters) and performs
    multi-scale alignment with sub-pixel refinement for precise text alignment.
    """
    
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Lucas-Kanade parameters for sub-pixel tracking
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Harris corner parameters
        self.harris_params = dict(
            blockSize=2,
            ksize=3,
            k=0.04
        )
    
    def extract_text_features(self, frame: np.ndarray, mask: np.ndarray) -> Dict[str, any]:
        """
        Extract hierarchical and text-specific features.
        
        Returns:
            Dictionary containing multiple feature types
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        features = {
            'hierarchical': self._extract_hierarchical_features(gray, mask),
            'text_specific': self._extract_text_specific_features(gray, mask)
        }
        
        return features
    
    def _extract_hierarchical_features(self, gray: np.ndarray, mask: np.ndarray) -> List[Dict]:
        """
        Extract features at multiple scales for hierarchical alignment.
        
        Returns:
            List of feature dictionaries for each scale
        """
        hierarchical_features = []
        scales = [0.25, 0.5, 1.0]
        
        for scale in scales:
            if scale < 1.0:
                # Downsample for coarse alignment
                new_h, new_w = int(gray.shape[0] * scale), int(gray.shape[1] * scale)
                scaled_gray = cv2.resize(gray, (new_w, new_h))
                scaled_mask = cv2.resize(mask, (new_w, new_h))
            else:
                scaled_gray = gray
                scaled_mask = mask
            
            # Extract ORB features only in text regions
            masked_gray = cv2.bitwise_and(scaled_gray, scaled_gray, mask=scaled_mask)
            keypoints, descriptors = self.orb.detectAndCompute(masked_gray, scaled_mask)
            
            # Scale keypoints back to original coordinates
            if scale < 1.0:
                for kp in keypoints:
                    kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)
            
            hierarchical_features.append({
                'scale': scale,
                'keypoints': keypoints,
                'descriptors': descriptors,
                'num_features': len(keypoints) if keypoints else 0
            })
        
        return hierarchical_features
    
    def _extract_text_specific_features(self, gray: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract text-specific features like baselines, corners, and characters.
        
        Returns:
            Dictionary with different types of text features
        """
        text_features = {}
        
        # 1. Baseline points
        text_features['baselines'] = self._extract_baseline_points(gray, mask)
        
        # 2. Text corners (character edges)
        text_features['corners'] = self._extract_text_corners(gray, mask)
        
        # 3. Character keypoints
        text_features['characters'] = self._extract_character_keypoints(gray, mask)
        
        return text_features
    
    def _extract_baseline_points(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Extract points along text baselines.
        
        Returns:
            Array of baseline points (N, 2)
        """
        # Find text regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        baseline_points = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            if w < 50 or h < 10:  # Too small for text
                continue
                
            # Extract baseline from bottom of text region
            region = gray[y:y+h, x:x+w]
            region_mask = mask[y:y+h, x:x+w]
            
            # Find bottom pixels of text
            for col in range(0, w, 5):  # Sample every 5 pixels
                if col >= region.shape[1]:
                    continue
                    
                text_pixels = np.where(region_mask[:, col] > 0)[0]
                if len(text_pixels) > 0:
                    bottom_y = np.max(text_pixels)
                    baseline_points.append([x + col, y + bottom_y])
        
        return np.array(baseline_points) if baseline_points else np.empty((0, 2))
    
    def _extract_text_corners(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Extract corner points specifically in text regions.
        
        Returns:
            Array of corner points (N, 2)
        """
        # Apply Harris corner detection only in text regions
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Harris corner detection
        corners = cv2.cornerHarris(masked_gray, **self.harris_params)
        
        # Threshold and find corner coordinates
        corner_threshold = 0.01 * corners.max()
        corner_coords = np.where(corners > corner_threshold)
        
        if len(corner_coords[0]) == 0:
            return np.empty((0, 2))
        
        # Convert to (x, y) format
        corner_points = np.column_stack([corner_coords[1], corner_coords[0]])
        
        # Filter corners that are actually in text regions
        valid_corners = []
        for point in corner_points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                if mask[y, x] > 0:
                    valid_corners.append(point)
        
        return np.array(valid_corners) if valid_corners else np.empty((0, 2))
    
    def _extract_character_keypoints(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Extract keypoints that correspond to character features.
        
        Returns:
            Array of character keypoints (N, 2)
        """
        # Use FAST detector for character-level features
        fast = cv2.FastFeatureDetector_create(threshold=20)
        
        # Apply mask to focus on text regions
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        keypoints = fast.detect(masked_gray, mask=mask)
        
        if not keypoints:
            return np.empty((0, 2))
        
        # Convert keypoints to coordinate array
        char_points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        
        return char_points
    
    def hierarchical_align(self, ref_features: Dict, target_features: Dict) -> Optional[np.ndarray]:
        """
        Perform hierarchical alignment from coarse to fine.
        
        Returns:
            Homography matrix or None if alignment fails
        """
        ref_hierarchical = ref_features['hierarchical']
        target_hierarchical = target_features['hierarchical']
        
        current_transform = None
        
        # Process each scale from coarse to fine
        for i, (ref_scale_features, target_scale_features) in enumerate(zip(ref_hierarchical, target_hierarchical)):
            
            if ref_scale_features['num_features'] < 10 or target_scale_features['num_features'] < 10:
                continue
                
            # Match features at this scale
            matches = self._match_features(ref_scale_features, target_scale_features)
            
            if len(matches) < 10:
                continue
                
            # Estimate homography at this scale
            scale_transform = self._estimate_homography_from_matches(
                ref_scale_features, target_scale_features, matches
            )
            
            if scale_transform is None:
                continue
                
            if current_transform is None:
                current_transform = scale_transform
            else:
                # Refine previous transform with current scale
                current_transform = self._refine_transform(current_transform, scale_transform)
        
        return current_transform
    
    def _match_features(self, ref_features: Dict, target_features: Dict) -> List[cv2.DMatch]:
        """
        Match features between reference and target.
        
        Returns:
            List of good matches
        """
        if ref_features['descriptors'] is None or target_features['descriptors'] is None:
            return []
            
        # Match descriptors
        matches = self.bf_matcher.match(ref_features['descriptors'], target_features['descriptors'])
        
        # Sort by distance and keep best matches
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Filter matches using ratio test approximation
        good_matches = []
        if len(matches) > 1:
            for i, match in enumerate(matches[:-1]):
                if match.distance < 0.75 * matches[i+1].distance:
                    good_matches.append(match)
                    
                # Limit number of matches for efficiency
                if len(good_matches) >= 100:
                    break
        
        return good_matches
    
    def _estimate_homography_from_matches(self, ref_features: Dict, target_features: Dict, matches: List[cv2.DMatch]) -> Optional[np.ndarray]:
        """
        Estimate homography from feature matches.
        
        Returns:
            3x3 homography matrix or None if estimation fails
        """
        if len(matches) < 4:
            return None
            
        # Extract matched point coordinates
        ref_points = np.array([ref_features['keypoints'][m.queryIdx].pt for m in matches])
        target_points = np.array([target_features['keypoints'][m.trainIdx].pt for m in matches])
        
        # Estimate homography with RANSAC
        try:
            homography, mask = cv2.findHomography(
                target_points, ref_points,
                cv2.RANSAC, 
                ransacReprojThreshold=3.0,
                maxIters=1000,
                confidence=0.99
            )
            
            # Check if we have enough inliers
            if mask is not None and np.sum(mask) >= 8:
                return homography
                
        except cv2.error:
            pass
            
        return None
    
    def _refine_transform(self, coarse_transform: np.ndarray, fine_transform: np.ndarray) -> np.ndarray:
        """
        Refine coarse transform with fine-scale information.
        
        Returns:
            Refined homography matrix
        """
        # Simple composition - could be made more sophisticated
        return fine_transform @ coarse_transform
    
    def subpixel_refinement(self, ref_frame: np.ndarray, target_frame: np.ndarray, initial_transform: np.ndarray) -> Optional[np.ndarray]:
        """
        Refine alignment using optical flow for sub-pixel precision.
        
        Returns:
            Refined homography matrix or None if refinement fails
        """
        ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
        
        # Extract high-quality corner points from reference frame
        ref_corners = cv2.goodFeaturesToTrack(
            ref_gray,
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10,
            useHarrisDetector=True
        )
        
        if ref_corners is None or len(ref_corners) < 10:
            return initial_transform
            
        # Apply initial transform to get starting points in target frame
        ref_corners_homog = np.column_stack([ref_corners.reshape(-1, 2), np.ones(len(ref_corners))])
        warped_corners_homog = (initial_transform @ ref_corners_homog.T).T
        warped_corners = warped_corners_homog[:, :2] / warped_corners_homog[:, 2:3]
        
        # Track with sub-pixel optical flow
        tracked_corners, status, error = cv2.calcOpticalFlowPyrLK(
            ref_gray, target_gray,
            ref_corners.reshape(-1, 1, 2).astype(np.float32),
            warped_corners.reshape(-1, 1, 2).astype(np.float32),
            **self.lk_params
        )
        
        # Filter good tracking results
        good_ref = ref_corners[status.flatten() == 1]
        good_tracked = tracked_corners[status.flatten() == 1]
        
        if len(good_ref) < 8:
            return initial_transform
            
        # Recompute homography with refined points
        try:
            refined_transform, mask = cv2.findHomography(
                good_tracked.reshape(-1, 2), good_ref.reshape(-1, 2),
                cv2.RANSAC, 1.0
            )
            
            if refined_transform is not None and mask is not None and np.sum(mask) >= 6:
                return refined_transform
                
        except cv2.error:
            pass
            
        return initial_transform