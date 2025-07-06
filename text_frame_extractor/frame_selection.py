import cv2
import numpy as np

class FrameSelector:
    """Select frames based on simple heuristics."""

    def __init__(self, top_n: int = 5):
        self.top_n = top_n

    def select(self, frames):
        """Return the top-N sharpest frames."""
        if len(frames) <= self.top_n:
            return frames

        scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            scores.append(variance)
        indices = np.argsort(scores)[::-1][: self.top_n]
        return [frames[i] for i in indices]
