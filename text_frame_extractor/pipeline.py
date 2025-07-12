import os
from typing import List, Tuple, Optional

import cv2
import numpy as np

from .frame_selection import FrameSelector
from .region_detection import RegionDetector
from .page_reconstruction import PageReconstructor
from .region_stitching import RegionStitcher
from .ocr import OCR
from .text_structure import TextStructureAnalyzer
from .quality_scoring import QualityScorer


def _polygon_to_mask(polygon: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Convert a polygon to a binary mask of given shape."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def _save_debug_frame_mask_pair(frame: np.ndarray, mask: np.ndarray, output_path: str, index: int):
    """Save a frame-mask pair as a debug image with masked areas in white and mask contour drawn."""
    debug_frame = frame.copy()
    binary_mask = (mask > 0).astype(np.uint8)
    for c in range(frame.shape[2]):
        debug_frame[:, :, c] = np.where(binary_mask == 1, frame[:, :, c], 255)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(debug_frame, contours, -1, (0, 0, 255), 2)
    cv2.imwrite(output_path, debug_frame)


def process_frames(frames: List[np.ndarray], debug_mode: bool = False, debug_output_dir: str = "local.debug") -> Tuple[np.ndarray, str, float]:
    """Process frames and return reconstructed image, text, and quality score."""
    selector = FrameSelector()
    detector = RegionDetector()
    reconstructor = PageReconstructor()
    stitcher = RegionStitcher()
    ocr = OCR()
    analyzer = TextStructureAnalyzer()
    scorer = QualityScorer()

    if debug_mode:
        os.makedirs(debug_output_dir, exist_ok=True)

    selected = selector.select(frames)
    frame_mask_pairs = []
    debug_idx = 0
    for frame in selected:
        regions = detector.detect(frame)
        if not regions:
            # No usable area in the frame, skip it
            continue
        for region in regions:
            mask = _polygon_to_mask(region.polygon, frame.shape[:2])
            frame_mask_pairs.append((frame, mask))
            if debug_mode:
                debug_path = os.path.join(debug_output_dir, f"frame_mask_pair_{debug_idx:03d}.png")
                _save_debug_frame_mask_pair(frame, mask, debug_path, debug_idx)
                debug_idx += 1

    reconstructed = reconstructor.reconstruct(frame_mask_pairs)
    text = ocr.extract_text(reconstructed)
    structured_text = analyzer.analyze(text)
    score = scorer.score(reconstructed)
    return reconstructed, structured_text, score


def process_video(video_path: str, debug_mode: bool = False, debug_output_dir: str = "local.debug") -> Tuple[np.ndarray, str, float]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    while success:
        frames.append(frame)
        success, frame = cap.read()
    cap.release()
    return process_frames(frames, debug_mode, debug_output_dir)
