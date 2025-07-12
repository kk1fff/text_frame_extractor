#!/usr/bin/env python3
"""
Simple test script for the advanced reconstruction pipeline.
"""

import sys
import os

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all new modules can be imported."""
    print("Testing imports...")
    
    try:
        from text_frame_extractor.perspective_correction import PerspectiveCorrector
        print("✓ PerspectiveCorrector imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PerspectiveCorrector: {e}")
        return False
    
    try:
        from text_frame_extractor.page_dewarping import PageDewarper
        print("✓ PageDewarper imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PageDewarper: {e}")
        return False
    
    try:
        from text_frame_extractor.text_aware_alignment import TextAwareAligner
        print("✓ TextAwareAligner imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TextAwareAligner: {e}")
        return False
    
    try:
        from text_frame_extractor.alignment_quality import AlignmentQualityAssessor
        print("✓ AlignmentQualityAssessor imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import AlignmentQualityAssessor: {e}")
        return False
    
    try:
        from text_frame_extractor.advanced_page_reconstruction import AdvancedPageReconstructor
        print("✓ AdvancedPageReconstructor imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import AdvancedPageReconstructor: {e}")
        return False
    
    return True

def test_pipeline_integration():
    """Test that the pipeline can be imported with advanced reconstruction."""
    print("\nTesting pipeline integration...")
    
    try:
        from text_frame_extractor.pipeline import process_frames
        print("✓ Pipeline with advanced reconstruction imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import pipeline: {e}")
        return False

def main():
    """Run all tests."""
    print("Advanced Reconstruction Test Suite")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test pipeline integration
    pipeline_ok = test_pipeline_integration()
    
    # Summary
    print("\n" + "=" * 50)
    if imports_ok and pipeline_ok:
        print("✅ All tests passed! Advanced reconstruction is ready to use.")
        print("\nTo use the advanced reconstruction:")
        print("  from text_frame_extractor.pipeline import process_frames")
        print("  result = process_frames(frames, use_advanced_reconstruction=True)")
        return 0
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())