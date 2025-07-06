import numpy as np
from text_frame_extractor.page_reconstruction import PageReconstructor

def test_page_reconstruction_average():
    frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
    frame1[:, :, 0] = 255
    frame2 = np.zeros((10, 10, 3), dtype=np.uint8)
    frame2[:, :, 2] = 255
    reconstructor = PageReconstructor()
    result = reconstructor.reconstruct([frame1, frame2])
    assert result[0, 0, 0] == 127 or result[0, 0, 0] == 128
    assert result[0, 0, 2] == 127 or result[0, 0, 2] == 128
