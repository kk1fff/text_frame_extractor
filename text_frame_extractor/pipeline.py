from typing import List, Tuple

import cv2
import numpy as np

from .frame_selection import FrameSelector
from .region_detection import RegionDetector
from .alignment import FrameAligner
from .occlusion_masking import OcclusionMasker
from .page_reconstruction import PageReconstructor
from .ocr import OCR
from .text_structure import TextStructureAnalyzer
from .quality_scoring import QualityScorer


def process_frames(frames: List[np.ndarray]) -> Tuple[np.ndarray, str, float]:
    """Process frames and return reconstructed image, text, and quality score."""
    selector = FrameSelector()
    detector = RegionDetector()
    aligner = FrameAligner()
    masker = OcclusionMasker()
    reconstructor = PageReconstructor()
    ocr = OCR()
    analyzer = TextStructureAnalyzer()
    scorer = QualityScorer()

    selected = selector.select(frames)
    aligned_frames = []
    masks = []
    for frame in selected:
        bbox = detector.detect(frame)
        aligned = aligner.align(frame, bbox)
        mask = masker.mask(aligned)
        aligned_frames.append(aligned)
        masks.append(mask)

    reconstructed = reconstructor.reconstruct(aligned_frames, masks)
    text = ocr.extract_text(reconstructed)
    structured_text = analyzer.analyze(text)
    score = scorer.score(reconstructed)
    return reconstructed, structured_text, score


def process_video(video_path: str) -> Tuple[np.ndarray, str, float]:
    """Load frames from video and process them."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    while success:
        frames.append(frame)
        success, frame = cap.read()
    cap.release()
    return process_frames(frames)
