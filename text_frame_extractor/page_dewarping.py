import cv2
import numpy as np
from typing import List, Tuple, Optional
from scipy import interpolate
from scipy.spatial.distance import cdist


class PageDewarper:
    """
    Corrects curved and warped pages using text-line analysis.
    
    Detects text baselines and models page surface curvature to generate
    a correction mesh that flattens the page for better OCR results.
    """
    
    def __init__(self):
        self.min_text_line_length = 50
        self.baseline_detection_threshold = 0.7
        self.curve_smoothing_factor = 0.3
        
    def dewarp_page(self, frame: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        Dewarp a curved/warped page.
        
        Args:
            frame: Input image
            mask: Binary mask indicating text regions
            
        Returns:
            Tuple of (dewarped_frame, dewarped_mask, dewarp_mesh)
        """
        # Extract text lines for curvature analysis
        text_lines = self._extract_text_baselines(frame, mask)
        
        if len(text_lines) < 2:
            # Not enough text lines for dewarping
            return frame, mask, None
            
        # Model page surface curvature
        surface_params = self._fit_curved_surface(text_lines, frame.shape)
        
        if surface_params is None:
            return frame, mask, None
            
        # Generate dewarping mesh
        dewarp_mesh_x, dewarp_mesh_y = self._create_dewarp_mesh(surface_params, frame.shape)
        
        # Apply mesh-based correction
        dewarped_frame = cv2.remap(frame, dewarp_mesh_x, dewarp_mesh_y, cv2.INTER_CUBIC)
        dewarped_mask = cv2.remap(mask, dewarp_mesh_x, dewarp_mesh_y, cv2.INTER_NEAREST)
        
        return dewarped_frame, dewarped_mask, (dewarp_mesh_x, dewarp_mesh_y)
    
    def _extract_text_baselines(self, frame: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
        """
        Extract text baseline coordinates from the image.
        
        Returns:
            List of baseline arrays, each shaped (N, 2) with x,y coordinates
        """
        # Convert to grayscale and apply mask
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Find text regions using connected components
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        baselines = []
        
        for contour in contours:
            # Filter small regions
            area = cv2.contourArea(contour)
            if area < 200:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract text line from this region
            baseline = self._extract_baseline_from_region(masked_gray[y:y+h, x:x+w], x, y)
            
            if baseline is not None and len(baseline) > self.min_text_line_length:
                baselines.append(baseline)
                
        return baselines
    
    def _extract_baseline_from_region(self, region: np.ndarray, offset_x: int, offset_y: int) -> Optional[np.ndarray]:
        """
        Extract baseline from a single text region.
        
        Returns:
            Baseline points as (N, 2) array or None if extraction fails
        """
        h, w = region.shape
        
        if h < 10 or w < self.min_text_line_length:
            return None
            
        # Use horizontal projection to find text baseline
        horizontal_proj = np.sum(region > 0, axis=0)
        
        # Find the bottom of text characters (baseline)
        baseline_points = []
        
        for x in range(w):
            if horizontal_proj[x] > 0:
                # Find the lowest text pixel in this column
                text_pixels = np.where(region[:, x] > 0)[0]
                if len(text_pixels) > 0:
                    baseline_y = np.max(text_pixels)
                    # Convert to global coordinates
                    global_x = x + offset_x
                    global_y = baseline_y + offset_y
                    baseline_points.append([global_x, global_y])
        
        if len(baseline_points) < self.min_text_line_length:
            return None
            
        return np.array(baseline_points)
    
    def _fit_curved_surface(self, text_lines: List[np.ndarray], frame_shape: Tuple[int, int, int]) -> Optional[dict]:
        """
        Fit a curved surface model to the text baselines.
        
        Returns:
            Dictionary with surface parameters or None if fitting fails
        """
        if len(text_lines) < 2:
            return None
            
        h, w = frame_shape[:2]
        
        # Combine all baseline points
        all_points = []
        for baseline in text_lines:
            all_points.extend(baseline)
            
        if len(all_points) < 10:
            return None
            
        all_points = np.array(all_points)
        
        # Separate x, y coordinates
        x_coords = all_points[:, 0]
        y_coords = all_points[:, 1]
        
        # Model page curvature as deviation from ideal straight lines
        # For each text line, compute deviation from horizontal
        line_deviations = []
        
        for baseline in text_lines:
            if len(baseline) < 10:
                continue
                
            x = baseline[:, 0]
            y = baseline[:, 1]
            
            # Fit polynomial to this baseline
            try:
                poly_coeffs = np.polyfit(x, y, deg=2)  # Quadratic curve
                ideal_y = np.polyval([0, 0, np.mean(y)], x)  # Horizontal line
                actual_y = np.polyval(poly_coeffs, x)
                deviation = actual_y - ideal_y
                
                line_deviations.append({
                    'x': x,
                    'y': y,
                    'deviation': deviation,
                    'poly_coeffs': poly_coeffs
                })
            except np.linalg.LinAlgError:
                continue
        
        if not line_deviations:
            return None
            
        # Create surface model
        surface_params = {
            'frame_shape': frame_shape,
            'line_deviations': line_deviations,
            'global_curvature': self._estimate_global_curvature(line_deviations, w, h)
        }
        
        return surface_params
    
    def _estimate_global_curvature(self, line_deviations: List[dict], width: int, height: int) -> dict:
        """
        Estimate global page curvature parameters.
        
        Returns:
            Dictionary with global curvature parameters
        """
        # Collect curvature information from all lines
        all_x = []
        all_y = []
        all_deviations = []
        
        for line_data in line_deviations:
            all_x.extend(line_data['x'])
            all_y.extend(line_data['y'])
            all_deviations.extend(line_data['deviation'])
        
        if len(all_x) < 10:
            return {'type': 'none'}
            
        # Fit global surface model
        try:
            # Create regular grid for interpolation
            grid_x = np.linspace(0, width-1, min(width//10, 50))
            grid_y = np.linspace(0, height-1, min(height//10, 30))
            
            # Interpolate deviations over the grid
            from scipy.interpolate import griddata
            
            points = np.column_stack([all_x, all_y])
            grid_points = np.meshgrid(grid_x, grid_y)
            grid_points = np.column_stack([grid_points[0].ravel(), grid_points[1].ravel()])
            
            interpolated_deviations = griddata(
                points, all_deviations, grid_points, 
                method='linear', fill_value=0
            )
            
            # Reshape back to grid
            deviation_grid = interpolated_deviations.reshape(len(grid_y), len(grid_x))
            
            return {
                'type': 'grid',
                'grid_x': grid_x,
                'grid_y': grid_y,
                'deviation_grid': deviation_grid
            }
            
        except Exception:
            # Fallback to simple model
            mean_deviation = np.mean(all_deviations)
            return {
                'type': 'uniform',
                'mean_deviation': mean_deviation
            }
    
    def _create_dewarp_mesh(self, surface_params: dict, frame_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create dewarping mesh from surface parameters.
        
        Returns:
            Tuple of (mesh_x, mesh_y) for cv2.remap
        """
        h, w = frame_shape[:2]
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        mesh_x = x_coords.astype(np.float32)
        mesh_y = y_coords.astype(np.float32)
        
        global_curvature = surface_params['global_curvature']
        
        if global_curvature['type'] == 'grid':
            # Apply grid-based correction
            grid_x = global_curvature['grid_x']
            grid_y = global_curvature['grid_y']
            deviation_grid = global_curvature['deviation_grid']
            
            # Interpolate deviations to full resolution
            from scipy.interpolate import RectBivariateSpline
            
            try:
                spline = RectBivariateSpline(grid_y, grid_x, deviation_grid, kx=1, ky=1)
                full_deviations = spline(np.arange(h), np.arange(w))
                
                # Apply corrections (subtract curvature to flatten)
                mesh_y = mesh_y - full_deviations.astype(np.float32)
                
            except Exception:
                # Fallback: no correction
                pass
                
        elif global_curvature['type'] == 'uniform':
            # Apply uniform correction
            mean_deviation = global_curvature['mean_deviation']
            if abs(mean_deviation) > 1:
                # Simple barrel/pincushion correction
                center_x, center_y = w // 2, h // 2
                max_radius = min(center_x, center_y)
                
                for y in range(h):
                    for x in range(w):
                        dx = x - center_x
                        dy = y - center_y
                        radius = np.sqrt(dx*dx + dy*dy)
                        
                        if radius > 0:
                            correction_factor = 1.0 - (mean_deviation / max_radius) * (radius / max_radius)
                            mesh_x[y, x] = center_x + dx * correction_factor
                            mesh_y[y, x] = center_y + dy * correction_factor
        
        # Ensure mesh coordinates are within bounds
        mesh_x = np.clip(mesh_x, 0, w-1)
        mesh_y = np.clip(mesh_y, 0, h-1)
        
        return mesh_x, mesh_y