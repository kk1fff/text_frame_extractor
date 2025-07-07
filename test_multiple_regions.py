#!/usr/bin/env python3
"""
Test script to verify multiple region handling.
"""

import cv2
import numpy as np
from text_frame_extractor.pipeline import process_frames
from text_frame_extractor.region_detection import RegionDetector


def create_test_frame_with_multiple_regions():
    """Create a test frame with multiple text-like regions."""
    # Create a white background
    frame = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Add multiple rectangular regions that look like text areas
    regions = [
        # Top left region - make it more text-like
        (50, 50, 200, 150),
        # Top right region - make it more text-like
        (550, 50, 200, 150),
        # Bottom left region - make it more text-like
        (50, 400, 200, 150),
        # Bottom right region - make it more text-like
        (550, 400, 200, 150),
    ]
    
    for x, y, w, h in regions:
        # Create a more realistic text region with proper contrast
        # Fill the region with light gray background
        cv2.rectangle(frame, (x, y), (x + w, y + h), (240, 240, 240), -1)
        # Add border
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
        
        # Add multiple text lines to make it look like actual text
        for i in range(5):
            line_y = y + 20 + i * 25
            if line_y < y + h - 10:
                # Draw text-like lines
                cv2.line(frame, (x + 10, line_y), (x + w - 10, line_y), (50, 50, 50), 2)
                # Add some variation to make it look more realistic
                if i % 2 == 0:
                    cv2.line(frame, (x + 10, line_y + 5), (x + w - 20, line_y + 5), (50, 50, 50), 1)
    
    return frame


def test_multiple_region_detection():
    """Test that multiple regions are detected and processed correctly."""
    print("Testing multiple region detection...")
    
    # Create test frame
    frame = create_test_frame_with_multiple_regions()
    frames = [frame]
    
    # Test region detection
    detector = RegionDetector(max_regions=5)  # Allow more regions
    regions = detector.detect(frame)
    
    print(f"Detected {len(regions)} regions")
    for i, region in enumerate(regions):
        print(f"  Region {i}: score={region.score:.2f}, points={len(region.polygon)}")
        # Print bounding box info
        polygon = region.polygon
        x_coords = polygon[:, 0]
        y_coords = polygon[:, 1]
        print(f"    Bounds: x=[{x_coords.min()}, {x_coords.max()}], y=[{y_coords.min()}, {y_coords.max()}]")
    
    # Test full pipeline
    try:
        reconstructed, text, score = process_frames(frames)
        print(f"Pipeline completed successfully!")
        print(f"Reconstructed image shape: {reconstructed.shape}")
        print(f"Quality score: {score:.2f}")
        print(f"Extracted text length: {len(text)}")
        
        # Save the test frame and reconstructed image
        cv2.imwrite("test_input_frame.png", frame)
        cv2.imwrite("test_reconstructed.png", reconstructed)
        print("Saved test images: test_input_frame.png, test_reconstructed.png")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_multiple_region_detection() 