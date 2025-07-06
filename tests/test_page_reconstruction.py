import numpy as np
import cv2
from text_frame_extractor.page_reconstruction import PageReconstructor

def test_page_reconstruction_stitch_fallback():
    # Create two identical frames with no features
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame1[:, :] = (255, 0, 0)  # Blue
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2[:, :] = (0, 0, 255)  # Red

    reconstructor = PageReconstructor()
    result = reconstructor.reconstruct([frame1, frame2])

    # The stitched image should be the average of the two frames
    assert result[0, 0, 0] == 127 or result[0, 0, 0] == 128
    assert result[0, 0, 2] == 127 or result[0, 0, 2] == 128
