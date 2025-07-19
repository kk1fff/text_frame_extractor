import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from .region_detection import RegionDetector


class AdvancedFrameSelector:
    """
    Advanced frame selection using complementary coverage optimization.
    
    Selects frames that together provide maximum readable text area coverage
    by detecting and avoiding overlapping occlusions (shadows, glare, fingers, blur).
    """
    
    def __init__(self, target_count: int = 5, min_quality_threshold: float = 0.3):
        self.target_count = target_count
        self.min_quality_threshold = min_quality_threshold
        self.region_detector = RegionDetector()
        
    def select(self, frames: List[np.ndarray], debug_mode: bool = False) -> List[np.ndarray]:
        """
        Select complementary frames for maximum text coverage.
        
        Args:
            frames: List of input frames
            debug_mode: If True, print selection progress
            
        Returns:
            List of selected frames optimized for complementary coverage
        """
        if len(frames) <= self.target_count:
            return frames
            
        if debug_mode:
            print(f"Advanced frame selection from {len(frames)} frames...")
        
        # Stage 1: Quality-based pre-filtering
        quality_filtered = self._filter_by_quality(frames, debug_mode)
        
        if len(quality_filtered) <= self.target_count:
            return quality_filtered
            
        # Stage 2: Analyze all frames for occlusions and coverage
        frame_analysis = self._analyze_all_frames(quality_filtered, debug_mode)
        
        # Stage 3: Select complementary coverage set
        selected = self._select_complementary_frames(frame_analysis, debug_mode)
        
        if debug_mode:
            print(f"Selected {len(selected)} frames with optimized coverage")
            
        return selected
    
    def _filter_by_quality(self, frames: List[np.ndarray], debug_mode: bool = False) -> List[np.ndarray]:
        """Filter out poor quality frames based on sharpness."""
        quality_scores = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality_scores.append(sharpness)
        
        # Normalize scores to 0-1 range
        max_score = max(quality_scores) if quality_scores else 1.0
        normalized_scores = [score / max_score for score in quality_scores]
        
        # Filter frames above threshold
        quality_filtered = []
        for i, score in enumerate(normalized_scores):
            if score >= self.min_quality_threshold:
                quality_filtered.append(frames[i])
                
        if debug_mode:
            print(f"  Quality filtering: {len(quality_filtered)}/{len(frames)} frames passed threshold {self.min_quality_threshold}")
            
        return quality_filtered if quality_filtered else frames[:self.target_count]
    
    def _analyze_all_frames(self, frames: List[np.ndarray], debug_mode: bool = False) -> List[Dict]:
        """Analyze each frame for text regions, occlusions, and coverage."""
        analysis = []
        
        for i, frame in enumerate(frames):
            if debug_mode:
                print(f"  Analyzing frame {i+1}/{len(frames)}")
                
            # Detect text regions
            text_regions = self.region_detector.detect(frame)
            
            # Detect occlusions
            occlusions = self._detect_occlusions(frame)
            
            # Create coverage map
            coverage_map = self._create_coverage_map(frame, text_regions, occlusions)
            
            # Compute quality metrics
            quality_score = self._compute_frame_quality(frame, coverage_map)
            readable_area = np.sum(coverage_map)
            
            frame_data = {
                'frame': frame,
                'frame_index': i,
                'text_regions': text_regions,
                'occlusions': occlusions,
                'coverage_map': coverage_map,
                'quality_score': quality_score,
                'readable_area': readable_area,
                'total_area': coverage_map.size
            }
            
            analysis.append(frame_data)
            
            if debug_mode:
                coverage_pct = (readable_area / coverage_map.size) * 100
                print(f"    Coverage: {coverage_pct:.1f}%, Quality: {quality_score:.3f}")
        
        return analysis
    
    def _detect_occlusions(self, frame: np.ndarray) -> np.ndarray:
        """Detect various types of occlusions in the frame."""
        h, w = frame.shape[:2]
        occlusion_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 1. Shadow detection
        shadows = self._detect_shadows(frame)
        occlusion_mask = np.logical_or(occlusion_mask, shadows)
        
        # 2. Glare/highlight detection  
        glare = self._detect_glare(frame)
        occlusion_mask = np.logical_or(occlusion_mask, glare)
        
        # 3. Blur detection
        blur = self._detect_blur_regions(frame)
        occlusion_mask = np.logical_or(occlusion_mask, blur)
        
        # 4. Finger/object detection
        fingers = self._detect_finger_occlusions(frame)
        occlusion_mask = np.logical_or(occlusion_mask, fingers)
        
        # 5. Edge regions (likely partial coverage)
        edge_occlusions = self._detect_edge_occlusions(frame)
        occlusion_mask = np.logical_or(occlusion_mask, edge_occlusions)
        
        return occlusion_mask.astype(np.uint8)
    
    def _detect_shadows(self, frame: np.ndarray) -> np.ndarray:
        """Detect shadow regions using brightness analysis."""
        # Convert to LAB color space for better shadow detection
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Shadows have significantly lower brightness
        mean_brightness = np.mean(l_channel)
        std_brightness = np.std(l_channel)
        shadow_threshold = max(20, mean_brightness - 1.5 * std_brightness)
        
        shadows = l_channel < shadow_threshold
        
        # Morphological cleanup to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        shadows = cv2.morphologyEx(shadows.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        shadows = cv2.morphologyEx(shadows, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        
        return shadows > 0
    
    def _detect_glare(self, frame: np.ndarray) -> np.ndarray:
        """Detect glare/highlight regions."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Glare regions are in the top 2% of brightness
        glare_threshold = np.percentile(gray, 98)
        glare = gray > glare_threshold
        
        # Also check for saturated regions in color channels
        b, g, r = cv2.split(frame)
        saturation_threshold = 250
        saturated = ((b > saturation_threshold) | 
                    (g > saturation_threshold) | 
                    (r > saturation_threshold))
        
        glare = np.logical_or(glare, saturated)
        
        # Remove small isolated bright spots
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        glare = cv2.morphologyEx(glare.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        
        return glare > 0
    
    def _detect_blur_regions(self, frame: np.ndarray) -> np.ndarray:
        """Detect blurry regions using local variance."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute local variance using a sliding window
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        # Mean and mean of squares
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        mean_sq = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        
        # Local variance
        variance = mean_sq - mean**2
        
        # Blur threshold based on global variance
        global_variance = np.var(gray)
        blur_threshold = global_variance * 0.1  # 10% of global variance
        
        blur_regions = variance < blur_threshold
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        blur_regions = cv2.morphologyEx(blur_regions.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        return blur_regions > 0
    
    def _detect_finger_occlusions(self, frame: np.ndarray) -> np.ndarray:
        """Detect finger/hand occlusions using skin tone and shape analysis."""
        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Multiple skin tone ranges to handle different lighting
        skin_ranges = [
            ([0, 20, 70], [20, 255, 255]),    # Light skin
            ([0, 40, 80], [25, 255, 255]),    # Medium skin
            ([0, 60, 90], [30, 255, 255])     # Darker skin
        ]
        
        skin_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for (lower, upper) in skin_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            skin_mask = cv2.bitwise_or(skin_mask, mask)
        
        # Filter by shape and size constraints
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        finger_mask = np.zeros_like(skin_mask)
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Reasonable finger/hand size range
            if 200 < area < 100000:
                # Check shape characteristics
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                
                # Fingers/hands tend to be elongated or roughly square
                if aspect_ratio > 1.5 or (area > 5000 and aspect_ratio > 1.2):
                    cv2.fillPoly(finger_mask, [contour], 1)
        
        return finger_mask > 0
    
    def _detect_edge_occlusions(self, frame: np.ndarray) -> np.ndarray:
        """Detect edge regions that likely represent partial page coverage."""
        h, w = frame.shape[:2]
        edge_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Define edge regions (outer 5% of image)
        edge_width = min(w, h) // 20
        
        # Mark edge regions
        edge_mask[:edge_width, :] = 1          # Top edge
        edge_mask[-edge_width:, :] = 1         # Bottom edge
        edge_mask[:, :edge_width] = 1          # Left edge
        edge_mask[:, -edge_width:] = 1         # Right edge
        
        return edge_mask > 0
    
    def _create_coverage_map(self, frame: np.ndarray, text_regions: List, occlusions: np.ndarray) -> np.ndarray:
        """Create binary map of readable text areas."""
        h, w = frame.shape[:2]
        coverage_map = np.zeros((h, w), dtype=np.uint8)
        
        # Mark text regions as potentially readable
        for region in text_regions:
            # Convert polygon to mask
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [region.polygon.astype(np.int32)], 1)
            coverage_map = np.logical_or(coverage_map, mask)
        
        # Remove occluded areas from readable coverage
        readable_coverage = np.logical_and(coverage_map, ~occlusions.astype(bool))
        
        return readable_coverage.astype(np.uint8)
    
    def _compute_frame_quality(self, frame: np.ndarray, coverage_map: np.ndarray) -> float:
        """Compute overall quality score for the frame."""
        # Base sharpness score
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Coverage ratio
        total_pixels = coverage_map.size
        readable_pixels = np.sum(coverage_map)
        coverage_ratio = readable_pixels / total_pixels if total_pixels > 0 else 0
        
        # Combined quality score
        # Normalize sharpness to roughly 0-1 range (typical values 0-2000)
        normalized_sharpness = min(sharpness / 1000.0, 1.0)
        
        # Weighted combination: 60% coverage, 40% sharpness
        quality_score = 0.6 * coverage_ratio + 0.4 * normalized_sharpness
        
        return quality_score
    
    def _select_complementary_frames(self, frame_analysis: List[Dict], debug_mode: bool = False) -> List[np.ndarray]:
        """Select frames using greedy algorithm for maximum complementary coverage."""
        if not frame_analysis:
            return []
            
        # Sort by individual quality first
        frame_analysis.sort(key=lambda x: x['quality_score'], reverse=True)
        
        selected_frames = []
        cumulative_coverage = None
        total_area = frame_analysis[0]['total_area']
        
        if debug_mode:
            print(f"  Starting complementary selection:")
        
        # Always start with highest quality frame
        best_frame = frame_analysis[0]
        selected_frames.append(best_frame['frame'])
        cumulative_coverage = best_frame['coverage_map'].copy()
        
        if debug_mode:
            initial_coverage = np.sum(cumulative_coverage) / total_area * 100
            print(f"    Frame {best_frame['frame_index']}: {initial_coverage:.1f}% coverage (initial)")
        
        # Greedily add frames that maximize new readable area
        for iteration in range(self.target_count - 1):
            best_addition = None
            best_new_coverage = 0
            best_weighted_score = 0
            
            for candidate in frame_analysis:
                if any(np.array_equal(candidate['frame'], sf) for sf in selected_frames):
                    continue  # Already selected
                    
                # Compute additional readable area this frame would provide
                combined_coverage = np.logical_or(cumulative_coverage, candidate['coverage_map'])
                new_readable_pixels = np.sum(combined_coverage) - np.sum(cumulative_coverage)
                
                # Weight by frame quality to break ties
                quality_weight = candidate['quality_score'] * candidate['readable_area']
                weighted_score = new_readable_pixels + 0.1 * quality_weight
                
                if weighted_score > best_weighted_score:
                    best_weighted_score = weighted_score
                    best_new_coverage = new_readable_pixels
                    best_addition = candidate
            
            # Add best frame if it provides meaningful new coverage
            if best_addition and best_new_coverage > 0:
                selected_frames.append(best_addition['frame'])
                cumulative_coverage = np.logical_or(cumulative_coverage, best_addition['coverage_map'])
                
                if debug_mode:
                    total_coverage = np.sum(cumulative_coverage) / total_area * 100
                    new_coverage_pct = best_new_coverage / total_area * 100
                    print(f"    Frame {best_addition['frame_index']}: +{new_coverage_pct:.1f}% → {total_coverage:.1f}% total")
            else:
                if debug_mode:
                    print(f"    No more frames provide significant additional coverage")
                break
        
        return selected_frames