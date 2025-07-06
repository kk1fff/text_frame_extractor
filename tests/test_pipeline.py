import numpy as np
import cv2
from text_frame_extractor.pipeline import process_frames


def create_text_image(text: str) -> np.ndarray:
    img = np.ones((50, 200, 3), dtype=np.uint8) * 255
    cv2.putText(img, text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def test_pipeline_end_to_end():
    frames = [create_text_image("Test"), create_text_image("Test")]
    image, text, score = process_frames(frames)
    assert isinstance(image, np.ndarray)
    assert "test" in text.lower()
    assert score > 0
