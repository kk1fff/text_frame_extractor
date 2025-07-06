import numpy as np

class OcclusionMasker:
    """Return a mask of valid pixels. Currently assume all pixels are valid."""

    def mask(self, frame):
        """Placeholder that marks all pixels valid."""
        return np.ones(frame.shape[:2], dtype=np.uint8) * 255
