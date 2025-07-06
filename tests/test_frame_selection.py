import cv2
import numpy as np
from text_frame_extractor.frame_selection import FrameSelector

def test_frame_selection():
    sharp = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.putText(sharp, "A", (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    blurry = cv2.GaussianBlur(sharp, (5, 5), 3)
    frames = [blurry, sharp, blurry]
    selector = FrameSelector(top_n=1)
    selected = selector.select(frames)
    assert len(selected) == 1
    # sharpest frame should be selected
    assert np.array_equal(selected[0], sharp)
