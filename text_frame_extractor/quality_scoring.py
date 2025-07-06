import cv2


class QualityScorer:
    """Compute a simple quality score based on sharpness."""

    def score(self, image) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(var)
