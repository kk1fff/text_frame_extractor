import cv2
import numpy as np
from text_frame_extractor.region_detection import RegionDetector

def test_region_detection():
    detector = RegionDetector()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 20), (90, 80), (255, 255, 255), -1)
    regions = detector.detect(frame)
    assert regions
    x, y, w, h = cv2.boundingRect(np.asarray(regions[0].polygon))
    assert (x, y, w, h) == (10, 20, 81, 61)
    assert regions[0].score > 0
