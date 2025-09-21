import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import math
from collections import defaultdict

class DensityAnalyzer:
    """Analyzes crowd density using grid-based approach and generates heatmaps"""
    
    def __init__(self, grid_rows: int = 10, grid_cols: int = 10, 
                 normal_threshold: int = 5, high_threshold: int = 15):
        """
        Initialize the density analyzer
        
        Args:
            grid_rows: Number of rows in the analysis grid
            grid_cols: Number of columns in the analysis grid
            normal_threshold: People count threshold for normal density
            high_threshold: People count threshold for high density
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.normal_threshold = normal_threshold
        self.high_threshold = high_threshold
        
        # Historical data for tracking
        self.density_history = []
        self.hotspot_tracking = defaultdict(int)
        
        # Smoothing parameters
        self.temporal_smoothing = 0.3  # Weight for temporal smoothing
        self.previous_density_matrix = None
    
    def analyze_frame(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        """
        Analyze crowd density in a single frame
        
        Args:
            frame: Input video frame
            detections: List of person detections from YOLOv8
            
        Returns:
            Dictionary containing density analysis results
        """
        height, width = frame.shape[:2]
        
        # Create density grid
        density_matrix = self._create_density_grid(width, height, detections)
        
        # Apply temporal smoothing if previous frame exists
        if self.previous_density_matrix is not None:
            density_matrix = self._apply_temporal_smoothing(density_matrix)
        
        # Analyze zones
        zone_analysis = self._analyze_zones(density_matrix)
        
        # Detect hotspots
        hotspots = self._detect_hotspots(density_matrix)
        
        # Calculate overall metrics
        metrics = self._calculate_metrics(density_matrix, detections)
        
        # Store for history
        density_data = {
            'density_matrix': density_matrix.tolist(),
            'zone_densities': density_matrix.flatten().tolist(),
            'active_zones': zone_analysis['active_zones'],
            'high_density_zones': zone_analysis['high_density_zones'],
            'normal_density_zones': zone_analysis['normal_density_zones'],
            'hotspots': hotspots,
            'metrics': metrics,
            'grid_dimensions': (self.grid_rows, self.grid_cols),
            'thresholds': {
                'normal': self.normal_threshold,
                'high': self.high_threshold
            }
        }
        
        self.density_history.append(density_data)
        self.previous_density_matrix = density_matrix.copy()
        
        # Keep only recent history (last 50 frames)
        if len(self.density_history) > 50:
            self.density_history.pop(0)
        
        return density_data
    
    def _create_density_grid(self, width: int, height: int, detections: List[Dict]) -> np.ndarray:
        """Create a grid-based density matrix"""
        density_matrix = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        
        # Calculate grid cell dimensions
        cell_width = width / self.grid_cols
        cell_height = height / self.grid_rows
        
        # Count people in each grid cell
        for detection in detections:
            center_x, center_y = detection['center']
            
            # Determine grid cell
            grid_col = min(int(center_x / cell_width), self.grid_cols - 1)
            grid_row = min(int(center_y / cell_height), self.grid_rows - 1)
            
            # Increment count for this cell
            density_matrix[grid_row, grid_col] += 1
        
        return density_matrix
    
    def _apply_temporal_smoothing(self, current_matrix: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to reduce noise"""
        if self.previous_density_matrix is not None:
            smoothed_matrix = (self.temporal_smoothing * current_matrix + 
                              (1 - self.temporal_smoothing) * self.previous_density_matrix)
            return smoothed_matrix
        return current_matrix
    
    def _analyze_zones(self, density_matrix: np.ndarray) -> Dict:
        """Analyze different density zones"""
        flat_densities = density_matrix.flatten()
        
        active_zones = np.sum(flat_densities > 0)
        normal_zones = np.sum((flat_densities > 0) & (flat_densities <= self.normal_threshold))
        moderate_zones = np.sum((flat_densities > self.normal_threshold) & (flat_densities <= self.high_threshold))
        high_density_zones = np.sum(flat_densities > self.high_threshold)
        
        return {
            'active_zones': int(active_zones),
            'normal_density_zones': int(normal_zones),
            'moderate_density_zones': int(moderate_zones),
            'high_density_zones': int(high_density_zones),
            'total_zones': self.grid_rows * self.grid_cols
        }
    
    def _detect_hotspots(self, density_matrix: np.ndarray) -> List[Dict]:
        """Detect crowd hotspots (areas with high local density)"""
        hotspots = []
        
        # Find zones that exceed high threshold
        high_density_indices = np.where(density_matrix > self.high_threshold)
        
        for row, col in zip(high_density_indices[0], high_density_indices[1]):
            hotspot_id = f"R{row}C{col}"
            density = density_matrix[row, col]
            
            # Track persistent hotspots
            self.hotspot_tracking[hotspot_id] += 1
            
            hotspot = {
                'id': hotspot_id,
                'row': int(row),
                'col': int(col),
                'density': float(density),
                'persistence': self.hotspot_tracking[hotspot_id],
                'severity': self._calculate_hotspot_severity(float(density))
            }
            hotspots.append(hotspot)
        
        # Decay non-active hotspots
        active_hotspot_ids = {f"R{row}C{col}" for row, col in zip(high_density_indices[0], high_density_indices[1])}
        for hotspot_id in list(self.hotspot_tracking.keys()):
            if hotspot_id not in active_hotspot_ids:
                self.hotspot_tracking[hotspot_id] = max(0, self.hotspot_tracking[hotspot_id] - 1)
                if self.hotspot_tracking[hotspot_id] == 0:
                    del self.hotspot_tracking[hotspot_id]
        
        return hotspots
    
    def _calculate_hotspot_severity(self, density: float) -> str:
        """Calculate hotspot severity based on density"""
        if density > self.high_threshold * 2:
            return 'critical'
        elif density > self.high_threshold * 1.5:
            return 'high'
        elif density > self.high_threshold:
            return 'moderate'
        else:
            return 'low'
    
    def _calculate_metrics(self, density_matrix: np.ndarray, detections: List[Dict]) -> Dict:
        """Calculate overall crowd metrics"""
        total_people = len(detections)
        occupied_cells = np.sum(density_matrix > 0)
        total_cells = self.grid_rows * self.grid_cols
        
        # Calculate density statistics
        flat_densities = density_matrix.flatten()
        non_zero_densities = flat_densities[flat_densities > 0]
        
        metrics = {
            'total_people': total_people,
            'occupancy_rate': float(occupied_cells / total_cells),
            'average_density': float(np.mean(non_zero_densities)) if len(non_zero_densities) > 0 else 0.0,
            'max_density': float(np.max(density_matrix)),
            'density_variance': float(np.var(non_zero_densities)) if len(non_zero_densities) > 0 else 0.0,
            'density_std': float(np.std(non_zero_densities)) if len(non_zero_densities) > 0 else 0.0
        }
        
        return metrics
    
    def get_density_heatmap_data(self, normalize: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Get heatmap data for visualization
        
        Args:
            normalize: Whether to normalize the density values
            
        Returns:
            Tuple of (heatmap_matrix, metadata)
        """
        if not self.density_history:
            return np.zeros((self.grid_rows, self.grid_cols)), {}
        
        latest_data = self.density_history[-1]
        density_matrix = np.array(latest_data['density_matrix'])
        
        if normalize and np.max(density_matrix) > 0:
            density_matrix = density_matrix / np.max(density_matrix)
        
        metadata = {
            'max_value': float(np.max(density_matrix)),
            'min_value': float(np.min(density_matrix)),
            'mean_value': float(np.mean(density_matrix)),
            'active_cells': int(np.sum(density_matrix > 0)),
            'grid_shape': density_matrix.shape
        }
        
        return density_matrix, metadata
    
    def calculate_crowd_flow(self, time_window: int = 5) -> Dict:
        """
        Calculate crowd flow patterns over recent frames
        
        Args:
            time_window: Number of recent frames to analyze
            
        Returns:
            Dictionary containing flow analysis
        """
        if len(self.density_history) < 2:
            return {'flow_vectors': [], 'overall_flow': 'stable'}
        
        recent_history = self.density_history[-time_window:]
        flow_analysis = {
            'flow_vectors': [],
            'overall_flow': 'stable',
            'flow_magnitude': 0.0
        }
        
        if len(recent_history) >= 2:
            # Compare first and last frames in window
            start_matrix = np.array(recent_history[0]['density_matrix'])
            end_matrix = np.array(recent_history[-1]['density_matrix'])
            
            # Calculate difference
            diff_matrix = end_matrix - start_matrix
            
            # Calculate overall flow magnitude
            flow_magnitude = np.mean(np.abs(diff_matrix))
            
            # Determine overall flow direction
            if flow_magnitude > 1.0:
                if np.sum(diff_matrix) > 0:
                    overall_flow = 'increasing'
                else:
                    overall_flow = 'decreasing'
            else:
                overall_flow = 'stable'
            
            flow_analysis.update({
                'overall_flow': overall_flow,
                'flow_magnitude': float(flow_magnitude),
                'change_matrix': diff_matrix.tolist()
            })
        
        return flow_analysis
    
    def update_thresholds(self, normal_threshold: int, high_threshold: int):
        """Update density thresholds"""
        self.normal_threshold = normal_threshold
        self.high_threshold = high_threshold
    
    def set_grid_size(self, rows: int, cols: int):
        """Update grid dimensions"""
        self.grid_rows = rows
        self.grid_cols = cols
        # Reset previous matrix when grid size changes
        self.previous_density_matrix = None
    
    def get_statistics(self) -> Dict:
        """Get overall statistics from density analysis history"""
        if not self.density_history:
            return {}
        
        # Calculate statistics across all frames
        all_densities = []
        all_hotspot_counts = []
        
        for frame_data in self.density_history:
            all_densities.extend(frame_data['zone_densities'])
            all_hotspot_counts.append(len(frame_data['hotspots']))
        
        all_densities = [d for d in all_densities if d > 0]  # Non-zero densities only
        
        return {
            'total_frames_analyzed': len(self.density_history),
            'average_density': float(np.mean(all_densities)) if all_densities else 0.0,
            'max_recorded_density': float(max(all_densities)) if all_densities else 0.0,
            'average_hotspots_per_frame': float(np.mean(all_hotspot_counts)) if all_hotspot_counts else 0.0,
            'max_hotspots_in_frame': max(all_hotspot_counts) if all_hotspot_counts else 0,
            'persistent_hotspots': len(self.hotspot_tracking)
        }
