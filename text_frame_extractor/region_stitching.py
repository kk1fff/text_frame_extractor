import cv2
import numpy as np
from typing import List, Tuple
from .region_detection import DetectedRegion


class RegionStitcher:
    """
    Stitch multiple regions from the same frame into a single composite image.
    
    This class handles the case where multiple text regions are detected in a single
    frame and need to be combined into one coherent image for further processing.
    """
    
    def __init__(self, padding: int = 50):
        """
        Initialize the region stitcher.
        
        Args:
            padding: Padding to add between regions when stitching
        """
        self.padding = padding
    
    def stitch_regions(self, frame: np.ndarray, regions: List[DetectedRegion], 
                      aligned_frames: List[np.ndarray]) -> np.ndarray:
        """
        Stitch multiple regions from a frame into a single composite image.
        
        Args:
            frame: Original frame containing the regions
            regions: List of detected regions
            aligned_frames: List of cropped/aligned frames corresponding to each region
            
        Returns:
            Composite image containing all regions stitched together
        """
        if not regions or not aligned_frames:
            return frame
        
        if len(regions) == 1:
            return aligned_frames[0]
        
        # Sort regions by their position (top to bottom, left to right)
        sorted_regions = self._sort_regions_by_position(regions)
        
        # Create a composite image
        composite = self._create_composite_image(frame, sorted_regions, aligned_frames)
        
        return composite
    
    def _sort_regions_by_position(self, regions: List[DetectedRegion]) -> List[DetectedRegion]:
        """
        Sort regions by their position in the frame (top to bottom, left to right).
        """
        def get_region_center(region):
            polygon = region.polygon
            center_x = np.mean(polygon[:, 0])
            center_y = np.mean(polygon[:, 1])
            return center_y, center_x  # Sort by y first, then x
        
        return sorted(regions, key=get_region_center)
    
    def _create_composite_image(self, frame: np.ndarray, regions: List[DetectedRegion], 
                               aligned_frames: List[np.ndarray]) -> np.ndarray:
        """
        Create a composite image by arranging regions in a logical layout.
        """
        # Calculate the total dimensions needed
        total_width, total_height = self._calculate_composite_dimensions(regions, aligned_frames)
        
        # Create a white background
        composite = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
        
        # Place each region in the composite
        current_x = self.padding
        current_y = self.padding
        max_height_in_row = 0
        
        for i, (region, aligned_frame) in enumerate(zip(regions, aligned_frames)):
            h, w = aligned_frame.shape[:2]
            
            # Check if we need to move to a new row
            if current_x + w + self.padding > total_width:
                current_x = self.padding
                current_y += max_height_in_row + self.padding
                max_height_in_row = 0
            
            # Place the region in the composite
            composite[current_y:current_y + h, current_x:current_x + w] = aligned_frame
            
            # Update position for next region
            current_x += w + self.padding
            max_height_in_row = max(max_height_in_row, h)
        
        return composite
    
    def _calculate_composite_dimensions(self, regions: List[DetectedRegion], 
                                      aligned_frames: List[np.ndarray]) -> Tuple[int, int]:
        """
        Calculate the dimensions needed for the composite image.
        """
        if not aligned_frames:
            return 800, 600  # Default size
        
        # Calculate total area and find optimal layout
        total_area = sum(frame.shape[0] * frame.shape[1] for frame in aligned_frames)
        
        # Estimate width based on average aspect ratio
        avg_aspect_ratio = np.mean([frame.shape[1] / frame.shape[0] for frame in aligned_frames])
        
        # Calculate optimal dimensions
        estimated_width = int(np.sqrt(total_area * avg_aspect_ratio))
        estimated_height = int(total_area / estimated_width)
        
        # Add padding
        estimated_width += self.padding * 2
        estimated_height += self.padding * 2
        
        # Ensure minimum dimensions
        estimated_width = max(estimated_width, 800)
        estimated_height = max(estimated_height, 600)
        
        return estimated_width, estimated_height 