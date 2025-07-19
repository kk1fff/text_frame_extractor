#!/usr/bin/env python3
"""
Test script for the advanced frame selection system.
"""

import sys
import os
import numpy as np

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_frame_with_occlusion(size=(480, 640, 3), occlusion_type='shadow'):
    """Create a synthetic test frame with specific occlusion types."""
    frame = np.random.randint(100, 200, size, dtype=np.uint8)  # Gray background
    
    # Add some "text-like" regions (darker rectangular areas)
    for i in range(5):
        x = np.random.randint(50, size[1] - 150)
        y = np.random.randint(50, size[0] - 50)
        w = np.random.randint(80, 120)
        h = np.random.randint(20, 40)
        frame[y:y+h, x:x+w] = 50  # Dark text regions
    
    if occlusion_type == 'shadow':
        # Add shadow (dark region)
        shadow_y = size[0] // 3
        shadow_h = size[0] // 3
        frame[shadow_y:shadow_y+shadow_h, :size[1]//2] = frame[shadow_y:shadow_y+shadow_h, :size[1]//2] * 0.3
        
    elif occlusion_type == 'glare':
        # Add glare (bright region)
        glare_x = size[1] // 3
        glare_w = size[1] // 3
        frame[:size[0]//2, glare_x:glare_x+glare_w] = 240
        
    elif occlusion_type == 'finger':
        # Add finger-like skin-colored region
        finger_x = size[1] // 2
        finger_y = size[0] // 4
        finger_w = 60
        finger_h = 200
        frame[finger_y:finger_y+finger_h, finger_x:finger_x+finger_w] = [180, 140, 120]  # Skin tone
        
    elif occlusion_type == 'blur':
        # Add blurred region
        import cv2
        blur_region = frame[size[0]//4:3*size[0]//4, size[1]//4:3*size[1]//4].copy()
        blurred = cv2.GaussianBlur(blur_region, (25, 25), 0)
        frame[size[0]//4:3*size[0]//4, size[1]//4:3*size[1]//4] = blurred
        
    return frame

def test_occlusion_detection():
    """Test occlusion detection methods."""
    print("Testing occlusion detection...")
    
    try:
        from text_frame_extractor.advanced_frame_selection import AdvancedFrameSelector
        selector = AdvancedFrameSelector()
        
        # Test different occlusion types
        occlusion_types = ['shadow', 'glare', 'finger', 'blur', 'clean']
        
        for occ_type in occlusion_types:
            frame = create_test_frame_with_occlusion(occlusion_type=occ_type)
            occlusions = selector._detect_occlusions(frame)
            
            occlusion_percentage = (np.sum(occlusions) / occlusions.size) * 100
            print(f"  {occ_type}: {occlusion_percentage:.1f}% detected as occluded")
            
        print("✓ Occlusion detection test passed")
        return True
        
    except Exception as e:
        print(f"✗ Occlusion detection test failed: {e}")
        return False

def test_frame_selection():
    """Test the complete frame selection process."""
    print("\nTesting frame selection...")
    
    try:
        from text_frame_extractor.advanced_frame_selection import AdvancedFrameSelector
        
        # Create frames with different occlusion patterns
        frames = []
        occlusion_patterns = ['shadow', 'glare', 'finger', 'clean', 'blur', 'shadow', 'clean']
        
        for pattern in occlusion_patterns:
            frame = create_test_frame_with_occlusion(occlusion_type=pattern)
            frames.append(frame)
        
        # Test advanced selection
        selector = AdvancedFrameSelector(target_count=3)
        selected = selector.select(frames, debug_mode=True)
        
        print(f"Selected {len(selected)} frames from {len(frames)} input frames")
        print("✓ Frame selection test passed")
        return True
        
    except Exception as e:
        print(f"✗ Frame selection test failed: {e}")
        return False

def test_pipeline_integration():
    """Test integration with the pipeline."""
    print("\nTesting pipeline integration...")
    
    try:
        from text_frame_extractor.pipeline import process_frames
        
        # Create test frames
        frames = []
        for i in range(7):
            pattern = ['clean', 'shadow', 'glare', 'finger', 'blur'][i % 5]
            frame = create_test_frame_with_occlusion(occlusion_type=pattern)
            frames.append(frame)
        
        print("Testing with advanced frame selection...")
        try:
            result = process_frames(frames, debug_mode=True, use_advanced_frame_selection=True)
            print("✓ Advanced frame selection pipeline test passed")
            advanced_success = True
        except Exception as e:
            print(f"✗ Advanced frame selection failed: {e}")
            advanced_success = False
        
        print("\nTesting with simple frame selection...")
        try:
            result = process_frames(frames, debug_mode=True, use_advanced_frame_selection=False)
            print("✓ Simple frame selection pipeline test passed")
            simple_success = True
        except Exception as e:
            print(f"✗ Simple frame selection failed: {e}")
            simple_success = False
            
        return advanced_success and simple_success
        
    except Exception as e:
        print(f"✗ Pipeline integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Advanced Frame Selection Test Suite")
    print("=" * 50)
    
    # Test occlusion detection
    occlusion_ok = test_occlusion_detection()
    
    # Test frame selection
    selection_ok = test_frame_selection()
    
    # Test pipeline integration
    pipeline_ok = test_pipeline_integration()
    
    # Summary
    print("\n" + "=" * 50)
    if occlusion_ok and selection_ok and pipeline_ok:
        print("✅ All tests passed! Advanced frame selection is ready to use.")
        print("\nTo use the advanced frame selection:")
        print("  from text_frame_extractor.pipeline import process_frames")
        print("  result = process_frames(frames, use_advanced_frame_selection=True)")
        return 0
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())