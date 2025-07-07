import cv2
import numpy as np

class FrameAligner:
    """Align frames to a reference."""
    def __init__(self, output_size=None):
        self.output_size = output_size

    def align(self, frame, polygon):
        """Crop the frame using the bounding box of ``polygon``.
        
        If polygon is None, return the entire frame.
        """
        if polygon is None:
            # Use the entire frame
            aligned = frame.copy()
        else:
            polygon = np.asarray(polygon, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(polygon)
            aligned = frame[y : y + h, x : x + w]
            
        if self.output_size is not None:
            aligned = cv2.resize(aligned, self.output_size)
        return aligned
