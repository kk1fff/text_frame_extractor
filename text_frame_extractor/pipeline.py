import os
from typing import List, Tuple, Optional

import cv2
import numpy as np

from .frame_selection import FrameSelector
from .region_detection import RegionDetector
from .alignment import FrameAligner
from .occlusion_masking import OcclusionMasker
from .page_reconstruction import PageReconstructor
from .region_stitching import RegionStitcher
from .ocr import OCR
from .text_structure import TextStructureAnalyzer
from .quality_scoring import QualityScorer


def _save_debug_frame_mask_pair(frame: np.ndarray, mask: np.ndarray, output_path: str, index: int):
    """Save a frame-mask pair as a debug image with masked areas in white and mask contour drawn."""
    # Create a copy of the frame
    debug_frame = frame.copy()
    
    # Apply mask: set non-masked pixels to white
    binary_mask = (mask > 0).astype(np.uint8)
    for c in range(frame.shape[2]):
        debug_frame[:, :, c] = np.where(binary_mask == 1, frame[:, :, c], 255)
    
    # Find and draw the contour of the mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw all contours in red
    cv2.drawContours(debug_frame, contours, -1, (0, 0, 255), 2)
    
    # Save the debug image
    cv2.imwrite(output_path, debug_frame)


def process_frames(frames: List[np.ndarray], debug_mode: bool = False, debug_output_dir: str = "local.debug") -> Tuple[np.ndarray, str, float]:
    """Process frames and return reconstructed image, text, and quality score."""
    selector = FrameSelector()
    detector = RegionDetector()
    aligner = FrameAligner()
    masker = OcclusionMasker()
    reconstructor = PageReconstructor()
    stitcher = RegionStitcher()
    ocr = OCR()
    analyzer = TextStructureAnalyzer()
    scorer = QualityScorer()

    # Create debug output directory if needed
    if debug_mode:
        os.makedirs(debug_output_dir, exist_ok=True)

    selected = selector.select(frames)
    processed_frames = []
    
    for frame in selected:
        regions = detector.detect(frame)
        
        if not regions:
            # No regions detected, use the entire frame
            processed_frame = frame
        elif len(regions) == 1:
            # Single region, just align it
            aligned = aligner.align(frame, regions[0].polygon)
            processed_frame = aligned
        else:
            # Multiple regions, align each and stitch them together
            aligned_frames = []
            for region in regions:
                aligned = aligner.align(frame, region.polygon)
                aligned_frames.append(aligned)
            
            # Stitch the aligned regions into a single composite
            processed_frame = stitcher.stitch_regions(frame, regions, aligned_frames)
        
        processed_frames.append(processed_frame)

    # Now process the stitched frames through the rest of the pipeline
    frame_mask_pairs = []
    for i, frame in enumerate(processed_frames):
        mask = masker.mask(frame)
        frame_mask_pairs.append((frame, mask))
        
        # Save debug images if debug mode is enabled
        if debug_mode:
            debug_path = os.path.join(debug_output_dir, f"frame_mask_pair_{i:03d}.png")
            _save_debug_frame_mask_pair(frame, mask, debug_path, i)

    reconstructed = reconstructor.reconstruct(frame_mask_pairs)
    text = ocr.extract_text(reconstructed)
    structured_text = analyzer.analyze(text)
    score = scorer.score(reconstructed)
    return reconstructed, structured_text, score


def process_video(video_path: str, debug_mode: bool = False, debug_output_dir: str = "local.debug") -> Tuple[np.ndarray, str, float]:
    """Load frames from video and process them."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    while success:
        frames.append(frame)
        success, frame = cap.read()
    cap.release()
    return process_frames(frames, debug_mode, debug_output_dir)
