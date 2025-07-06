import numpy as np
import cv2
from text_frame_extractor.page_reconstruction import PageReconstructor

def create_text_image(text: str, shape=(100, 200, 3)) -> np.ndarray:
    img = np.ones(shape, dtype=np.uint8) * 255
    cv2.putText(img, text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return img

def test_best_pixel_selection():
    """ 
    Tests that the reconstructor correctly selects the sharper pixels.
    """
    reconstructor = PageReconstructor()

    # Create a sharp and a blurry version of the same image
    sharp_frame = create_text_image("Sharp Text")
    blurry_frame = cv2.GaussianBlur(sharp_frame, (25, 25), 0)

    # The reconstructor should pick the pixels from the sharp frame
    result = reconstructor.reconstruct([blurry_frame, sharp_frame])

    # The result should be much closer to the sharp frame than the blurry one
    sharp_diff = np.sum(cv2.absdiff(result, sharp_frame))
    blurry_diff = np.sum(cv2.absdiff(result, blurry_frame))

    assert sharp_diff < blurry_diff

def test_selective_sharpening():
    """
    Tests that sharpening is only applied to blurry regions.
    """
    reconstructor = PageReconstructor()

    # Create a frame that is sharp in one half and blurry in the other
    sharp_half = create_text_image("Sharp")
    blurry_half = cv2.GaussianBlur(create_text_image("Blurry"), (25, 25), 0)
    
    test_frame = np.hstack([sharp_half, blurry_half])

    # Reconstruct from this single frame (which will trigger sharpening)
    result = reconstructor._selective_sharpen(test_frame, reconstructor._calculate_sharpness_map(test_frame), blur_threshold=50)

    # The sharp half should remain unchanged
    result_sharp_half = result[:, :200, :]
    assert np.allclose(result_sharp_half, sharp_half, atol=1)

    # The blurry half should be sharpened (i.e., different from the original blurry half)
    result_blurry_half = result[:, 200:, :]
    assert not np.allclose(result_blurry_half, blurry_half)