import cv2
import numpy as np

class FrameAligner:
    """Align frames to a reference."""
    def __init__(self, output_size=None):
        self.output_size = output_size

    def align(self, frame, polygon):
        """Crop the frame using the bounding box of ``polygon``."""
        polygon = np.asarray(polygon, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(polygon)
        cropped = frame[y : y + h, x : x + w]
        if self.output_size is not None:
            cropped = cv2.resize(cropped, self.output_size)
        return cropped
