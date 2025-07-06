import cv2
import numpy as np

class FrameAligner:
    """Align frames to a reference."""
    def __init__(self, output_size=None):
        self.output_size = output_size

    def align(self, frame, bbox):
        x, y, w, h = bbox
        cropped = frame[y : y + h, x : x + w]
        if self.output_size is not None:
            cropped = cv2.resize(cropped, self.output_size)
        return cropped
