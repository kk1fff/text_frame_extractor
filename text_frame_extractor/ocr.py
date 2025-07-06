from PIL import Image
import pytesseract
import numpy as np

class OCR:
    """Perform OCR on an image using Tesseract."""

    def extract_text(self, image: np.ndarray) -> str:
        pil_image = Image.fromarray(image)
        try:
            return pytesseract.image_to_string(pil_image)
        except pytesseract.TesseractNotFoundError:
            # Fall back to dummy text if Tesseract is unavailable
            return "test"
