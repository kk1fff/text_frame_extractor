import numpy as np

class PageReconstructor:
    """Combine aligned frames into a single image by averaging."""

    def reconstruct(self, frames, masks=None):
        if not frames:
            raise ValueError("No frames provided")
        accum = np.zeros_like(frames[0], dtype=np.float32)
        count = np.zeros(frames[0].shape[:2], dtype=np.float32)
        for idx, frame in enumerate(frames):
            mask = np.ones(frame.shape[:2], dtype=np.float32)
            if masks is not None:
                mask = masks[idx].astype(np.float32) / 255.0
            for c in range(frame.shape[2]):
                accum[:, :, c] += frame[:, :, c] * mask
            count += mask
        count[count == 0] = 1.0
        result = (accum / count[..., None]).astype(np.uint8)
        return result
