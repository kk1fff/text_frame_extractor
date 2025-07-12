import cv2
import numpy as np
from typing import List, Tuple, Optional

from .perspective_correction import PerspectiveCorrector
from .page_dewarping import PageDewarper
from .text_aware_alignment import TextAwareAligner
from .alignment_quality import AlignmentQualityAssessor


class AdvancedPageReconstructor:
    """
    Advanced multi-frame page reconstruction with comprehensive alignment.
    
    Integrates perspective correction, page dewarping, text-aware alignment,
    and quality assessment to achieve superior text reconstruction from
    multiple frames with various distortions.
    """
    
    def __init__(self, quality_threshold: float = 0.7):
        self.perspective_corrector = PerspectiveCorrector()
        self.page_dewarper = PageDewarper()
        self.text_aligner = TextAwareAligner()
        self.quality_assessor = AlignmentQualityAssessor()
        self.quality_threshold = quality_threshold
        
    def reconstruct(self, frame_mask_pairs: List[Tuple[np.ndarray, np.ndarray]], 
                   debug_mode: bool = False) -> np.ndarray:
        """
        Enhanced reconstruction with comprehensive alignment pipeline.
        
        Args:
            frame_mask_pairs: List of (frame, mask) tuples
            debug_mode: If True, print detailed progress information
            
        Returns:
            Reconstructed high-quality image
        """
        if not frame_mask_pairs:
            raise ValueError("No frame-mask pairs provided")
            
        if len(frame_mask_pairs) == 1:
            return self._handle_single_frame(frame_mask_pairs[0])
        
        if debug_mode:
            print(f"Starting advanced reconstruction with {len(frame_mask_pairs)} frames")
        
        # Stage 1: Geometric preprocessing for all frames
        preprocessed_pairs = self._geometric_preprocessing(frame_mask_pairs, debug_mode)
        
        if len(preprocessed_pairs) < 1:
            raise ValueError("No frames survived geometric preprocessing")
        
        # Stage 2-4: Advanced alignment
        aligned_pairs = self._advanced_alignment(preprocessed_pairs, debug_mode)
        
        if len(aligned_pairs) < 1:
            # Fallback to single best frame
            return self._handle_single_frame(preprocessed_pairs[0])
        
        # Stage 5: Best-pixel reconstruction
        composite_image, sharpness_map = self._build_composite_with_masks(aligned_pairs)
        final_image = self._selective_sharpen(composite_image, sharpness_map)
        
        if debug_mode:
            print(f"Reconstruction completed using {len(aligned_pairs)} aligned frames")
        
        return final_image
    
    def _handle_single_frame(self, frame_mask_pair: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Handle reconstruction from a single frame."""
        frame, mask = frame_mask_pair
        return self._apply_mask(frame, mask)
    
    def _geometric_preprocessing(self, frame_mask_pairs: List[Tuple[np.ndarray, np.ndarray]], 
                               debug_mode: bool) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Apply perspective correction and page dewarping to all frames.
        
        Returns:
            List of preprocessed (frame, mask) pairs
        """
        preprocessed_pairs = []
        
        for i, (frame, mask) in enumerate(frame_mask_pairs):
            if debug_mode:
                print(f"  Preprocessing frame {i+1}/{len(frame_mask_pairs)}")
            
            # Step 1: Perspective correction
            rect_frame, rect_mask, rect_matrix = self.perspective_corrector.correct_perspective(frame, mask)
            
            # Step 2: Page dewarping
            dewarp_frame, dewarp_mask, dewarp_mesh = self.page_dewarper.dewarp_page(rect_frame, rect_mask)
            
            preprocessed_pairs.append((dewarp_frame, dewarp_mask))
            
            if debug_mode:
                perspective_applied = rect_matrix is not None
                dewarping_applied = dewarp_mesh is not None
                print(f"    Perspective correction: {'✓' if perspective_applied else '✗'}")
                print(f"    Page dewarping: {'✓' if dewarping_applied else '✗'}")
        
        return preprocessed_pairs
    
    def _advanced_alignment(self, preprocessed_pairs: List[Tuple[np.ndarray, np.ndarray]], 
                          debug_mode: bool) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Perform advanced text-aware alignment with quality assessment.
        
        Returns:
            List of successfully aligned (frame, mask) pairs
        """
        if len(preprocessed_pairs) < 2:
            return preprocessed_pairs
        
        # Use first frame as reference
        reference_frame, reference_mask = preprocessed_pairs[0]
        aligned_pairs = [(reference_frame, reference_mask)]
        
        # Extract features from reference frame
        ref_features = self.text_aligner.extract_text_features(reference_frame, reference_mask)
        
        if debug_mode:
            ref_feature_counts = self._count_features(ref_features)
            print(f"  Reference frame features: {ref_feature_counts}")
        
        # Align each subsequent frame to the reference
        for i in range(1, len(preprocessed_pairs)):
            frame, mask = preprocessed_pairs[i]
            
            if debug_mode:
                print(f"  Aligning frame {i+1}/{len(preprocessed_pairs)}")
            
            # Extract features from target frame
            target_features = self.text_aligner.extract_text_features(frame, mask)
            
            # Hierarchical alignment
            coarse_transform = self.text_aligner.hierarchical_align(ref_features, target_features)
            
            if coarse_transform is None:
                if debug_mode:
                    print(f"    ✗ Hierarchical alignment failed")
                continue
            
            # Sub-pixel refinement
            refined_transform = self.text_aligner.subpixel_refinement(
                reference_frame, frame, coarse_transform
            )
            
            # Quality assessment
            quality_score, quality_breakdown = self.quality_assessor.assess_alignment_quality(
                reference_frame, frame, refined_transform, reference_mask, mask
            )
            
            if debug_mode:
                print(f"    Quality score: {quality_score:.3f} (threshold: {self.quality_threshold})")
                for metric, score in quality_breakdown.items():
                    print(f"      {metric}: {score:.3f}")
            
            # Only use if quality is sufficient
            if quality_score >= self.quality_threshold:
                h, w = reference_frame.shape[:2]
                aligned_frame = cv2.warpPerspective(frame, refined_transform, (w, h))
                aligned_mask = cv2.warpPerspective(mask, refined_transform, (w, h))
                aligned_pairs.append((aligned_frame, aligned_mask))
                
                if debug_mode:
                    print(f"    ✓ Frame accepted")
            else:
                if debug_mode:
                    print(f"    ✗ Frame rejected (quality too low)")
        
        return aligned_pairs
    
    def _count_features(self, features: dict) -> str:
        """Helper to count and format feature information."""
        hierarchical = features.get('hierarchical', [])
        text_specific = features.get('text_specific', {})
        
        total_keypoints = sum(f.get('num_features', 0) for f in hierarchical)
        baselines = len(text_specific.get('baselines', []))
        corners = len(text_specific.get('corners', []))
        characters = len(text_specific.get('characters', []))
        
        return f"keypoints={total_keypoints}, baselines={baselines}, corners={corners}, chars={characters}"
    
    def _apply_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask to frame, setting non-masked pixels to white."""
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
    
    def _calculate_sharpness_map(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Calculate sharpness score for each pixel using variance of Laplacian."""
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
    
    def _build_composite_with_masks(self, frame_mask_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build composite image by selecting sharpest pixel from all frames.
        
        Returns:
            Tuple of (composite_image, final_sharpness_map)
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
    
    def _selective_sharpen(self, image: np.ndarray, sharpness_map: np.ndarray, 
                         blur_threshold: float = 50) -> np.ndarray:
        """Apply sharpening filter only to blurry regions of the image."""
        # Find regions that are still blurry
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