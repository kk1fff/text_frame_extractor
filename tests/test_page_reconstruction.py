import numpy as np
import cv2
from text_frame_extractor.page_reconstruction import PageReconstructor

def create_text_image(text: str) -> np.ndarray:
    img = np.ones((100, 200, 3), dtype=np.uint8) * 255
    cv2.putText(img, text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return img

def test_page_reconstruction_sharpen():
    # Create a blurry frame with text
    frame = create_text_image("Test")
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    reconstructor = PageReconstructor()
    result = reconstructor.reconstruct([blurred_frame])

    # The sharpened image should be less blurry than the input
    laplacian_var_result = cv2.Laplacian(result, cv2.CV_64F).var()
    laplacian_var_blurred = cv2.Laplacian(blurred_frame, cv2.CV_64F).var()

    assert laplacian_var_result > laplacian_var_blurred
