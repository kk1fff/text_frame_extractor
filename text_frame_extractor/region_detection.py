import cv2
import numpy as np
from dataclasses import dataclass
from typing import List

from .occlusion_masking import OcclusionMasker


@dataclass
class DetectedRegion:
    """Representation of a detected region."""

    polygon: np.ndarray  # Nx2 array of integer coordinates
    score: float

class RegionDetector:
    """Detect target reading region and return scored polygons."""

    def __init__(self, max_regions: int = 3):
        self.max_regions = max_regions
        self.masker = OcclusionMasker()

    def detect(self, frame) -> List[DetectedRegion]:
        """Detect document-like regions in ``frame``.

        The returned list is ordered by descending score and contains at most
        ``max_regions`` elements.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Simple binary threshold to separate foreground from background
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours of connected components
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = self.masker.mask(frame)
        candidates: List[DetectedRegion] = []

        height, width = frame.shape[:2]
        frame_area = height * width

        for c in contours:
            # Approximate the contour to a polygon
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            area = cv2.contourArea(approx)
            if area < frame_area * 0.01 or area > frame_area * 0.99:
                continue

            x, y, w, h = cv2.boundingRect(approx)
            if w == 0 or h == 0:
                continue
            aspect_ratio = float(w) / h
            if not (0.3 < aspect_ratio < 3.0):
                continue

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = float(area) / hull_area
            if solidity < 0.7:
                continue

            coverage = np.mean(mask[y : y + h, x : x + w] / 255.0)
            score = (area * solidity / (abs(1.0 - aspect_ratio) + 0.1)) * coverage

            polygon = approx.reshape(-1, 2).astype(int)
            candidates.append(DetectedRegion(polygon=polygon, score=float(score)))

        if not candidates:
            full_poly = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
            return [DetectedRegion(polygon=full_poly, score=0.0)]

        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates[: self.max_regions]
