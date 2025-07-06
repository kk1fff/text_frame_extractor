import numpy as np
import cv2


class PageReconstructor:
    """Combine aligned frames into a single image by averaging."""

    def reconstruct(self, frames, masks=None):
        if not frames:
            raise ValueError("No frames provided")

        target_shape = frames[0].shape
        target_h, target_w = target_shape[:2]

        accum = np.zeros(target_shape, dtype=np.float32)
        count = np.zeros((target_h, target_w), dtype=np.float32)

        for idx, frame in enumerate(frames):
            # Resize frame to the target shape if it doesn't match
            if frame.shape[:2] != (target_h, target_w):
                frame = cv2.resize(frame, (target_w, target_h))

            mask = np.ones((target_h, target_w), dtype=np.float32)
            if masks is not None:
                current_mask = masks[idx]
                # Resize mask to the target shape if it doesn't match
                if current_mask.shape[:2] != (target_h, target_w):
                    current_mask = cv2.resize(current_mask, (target_w, target_h))

                # Ensure mask is single channel (grayscale)
                if current_mask.ndim == 3:
                    current_mask = cv2.cvtColor(current_mask, cv2.COLOR_BGR2GRAY)
                
                mask = current_mask.astype(np.float32) / 255.0

            for c in range(frame.shape[2]):
                accum[:, :, c] += frame[:, :, c] * mask
            count += mask

        count[count == 0] = 1.0
        result = (accum / count[..., None]).astype(np.uint8)
        return result
