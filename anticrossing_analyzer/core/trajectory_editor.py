"""
Trajectory editor module for interactive mode trajectory marking.

This module handles storage and manipulation of mode trajectories and data masks.
"""

from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import json
import os
from ..config.settings import GUI_SETTINGS, FILE_FORMATS


class Trajectory:
    """
    Single trajectory representing one mode (cavity or FMR).
    """
    
    def __init__(self, name: str, mode_type: str = 'fmr'):
        """
        Initialize trajectory.
        
        Parameters:
        -----------
        name : str
            Name of the trajectory (e.g., 'cavity', 'fmr1', 'fmr2')
        mode_type : str
            Type of mode: 'cavity' or 'fmr'
        """
        self.name = name
        self.mode_type = mode_type
        self.points = []  # List of (field, frequency) tuples
        self.color = self._get_color()
        
    def _get_color(self) -> str:
        """Get color for this trajectory based on type."""
        if self.mode_type == 'cavity':
            return GUI_SETTINGS['cavity_color']
        else:
            # Cycle through FMR colors
            colors = GUI_SETTINGS['trajectory_colors']
            return colors[hash(self.name) % len(colors)]
    
    def add_point(self, field: float, frequency: float):
        """Add point to trajectory."""
        self.points.append((field, frequency))
        self._sort_points()
    
    def remove_point(self, point_idx: int) -> bool:
        """
        Remove point from trajectory.
        
        Returns:
        --------
        bool
            True if point was removed successfully
        """
        if 0 <= point_idx < len(self.points):
            self.points.pop(point_idx)
            return True
        return False
    
    def remove_closest_point(self, field: float, frequency: float, tolerance: float = None) -> bool:
        """
        Remove point closest to given coordinates.
        
        Parameters:
        -----------
        field, frequency : float
            Target coordinates
        tolerance : float, optional
            Maximum distance for removal
            
        Returns:
        --------
        bool
            True if point was removed
        """
        if not self.points:
            return False
            
        if tolerance is None:
            tolerance = float('inf')
            
        # Find closest point
        distances = [np.sqrt((f - field)**2 + (freq - frequency)**2) 
                    for f, freq in self.points]
        min_idx = np.argmin(distances)
        
        if distances[min_idx] <= tolerance:
            self.points.pop(min_idx)
            return True
        return False
    
    def _sort_points(self):
        """Sort points by field value."""
        self.points.sort(key=lambda p: p[0])
    
    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get trajectory as numpy arrays.
        
        Returns:
        --------
        tuple
            (fields, frequencies) as numpy arrays
        """
        if not self.points:
            return np.array([]), np.array([])
        
        fields, frequencies = zip(*self.points)
        return np.array(fields), np.array(frequencies)
    
    def interpolate(self, target_fields: np.ndarray) -> np.ndarray:
        """
        Interpolate trajectory to target field values.
        
        Parameters:
        -----------
        target_fields : np.ndarray
            Field values for interpolation
            
        Returns:
        --------
        np.ndarray
            Interpolated frequencies
        """
        if len(self.points) < 2:
            return np.full_like(target_fields, np.nan)
        
        fields, frequencies = self.get_arrays()
        return np.interp(target_fields, fields, frequencies)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'mode_type': self.mode_type,
            'points': self.points,
            'color': self.color
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trajectory':
        """Create trajectory from dictionary."""
        traj = cls(data['name'], data['mode_type'])
        traj.points = data['points']
        if 'color' in data:
            traj.color = data['color']
        return traj


class MaskArea:
    """
    Rectangular area to be masked (excluded from analysis).
    """
    
    def __init__(self, field_range: Tuple[float, float], freq_range: Tuple[float, float]):
        """
        Initialize mask area.
        
        Parameters:
        -----------
        field_range : tuple
            (min_field, max_field)
        freq_range : tuple
            (min_freq, max_freq)
        """
        self.field_range = field_range
        self.freq_range = freq_range
        self.color = GUI_SETTINGS['mask_color']
        self.alpha = GUI_SETTINGS['mask_alpha']
    
    def contains_point(self, field: float, frequency: float) -> bool:
        """Check if point is inside this mask area."""
        return (self.field_range[0] <= field <= self.field_range[1] and
                self.freq_range[0] <= frequency <= self.freq_range[1])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert mask to dictionary for JSON serialization."""
        return {
            'field_range': self.field_range,
            'freq_range': self.freq_range
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MaskArea':
        """Create mask from dictionary."""
        return cls(data['field_range'], data['freq_range'])


class TrajectoryEditor:
    """
    Interactive editor for marking mode trajectories and problem areas.
    
    Manages collections of trajectories and masks, with save/load functionality.
    """
    
    def __init__(self):
        """Initialize empty trajectory editor."""
        self.trajectories: Dict[str, Trajectory] = {}
        self.masks: List[MaskArea] = []
        self.current_mode = 'fmr'  # Current editing mode: 'cavity', 'fmr', 'mask'
        self.current_trajectory_name = None
        self._fmr_counter = 1  # For auto-naming FMR trajectories
        
    def set_mode(self, mode: str):
        """
        Set current editing mode.
        
        Parameters:
        -----------
        mode : str
            Mode: 'cavity', 'fmr', or 'mask'
        """
        if mode in ['cavity', 'fmr', 'mask']:
            self.current_mode = mode
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def add_trajectory_point(self, field: float, frequency: float, trajectory_name: str = None):
        """
        Add point to current trajectory.
        
        Parameters:
        -----------
        field, frequency : float
            Point coordinates
        trajectory_name : str, optional
            Specific trajectory name. If None, uses current mode logic.
        """
        if self.current_mode == 'mask':
            return  # Points not added in mask mode
        
        # Determine trajectory name
        if trajectory_name is None:
            if self.current_mode == 'cavity':
                trajectory_name = 'cavity'
            else:  # fmr mode
                if self.current_trajectory_name is None:
                    trajectory_name = f'fmr{self._fmr_counter}'
                    self._fmr_counter += 1
                    self.current_trajectory_name = trajectory_name
                else:
                    trajectory_name = self.current_trajectory_name
        
        # Create trajectory if it doesn't exist
        if trajectory_name not in self.trajectories:
            mode_type = 'cavity' if trajectory_name == 'cavity' else 'fmr'
            self.trajectories[trajectory_name] = Trajectory(trajectory_name, mode_type)
        
        # Add point
        self.trajectories[trajectory_name].add_point(field, frequency)
    
    def remove_trajectory_point(self, field: float, frequency: float, tolerance: float = None):
        """
        Remove closest trajectory point.
        
        Parameters:
        -----------
        field, frequency : float
            Target coordinates
        tolerance : float, optional
            Maximum distance for removal
        """
        if tolerance is None:
            # Use default tolerance based on data ranges
            tolerance = 50.0  # Will be adjusted based on actual data ranges
        
        # Try to remove from all trajectories, find closest
        best_distance = float('inf')
        best_trajectory = None
        
        for traj in self.trajectories.values():
            if not traj.points:
                continue
            distances = [np.sqrt((f - field)**2 + (freq - frequency)**2) 
                        for f, freq in traj.points]
            min_distance = min(distances)
            if min_distance < best_distance and min_distance <= tolerance:
                best_distance = min_distance
                best_trajectory = traj
        
        if best_trajectory:
            return best_trajectory.remove_closest_point(field, frequency, tolerance)
        
        return False
    
    def add_mask_area(self, field_range: Tuple[float, float], freq_range: Tuple[float, float]):
        """
        Add masked area.
        
        Parameters:
        -----------
        field_range : tuple
            (min_field, max_field)
        freq_range : tuple  
            (min_freq, max_freq)
        """
        mask = MaskArea(field_range, freq_range)
        self.masks.append(mask)
    
    def start_new_fmr_trajectory(self):
        """Start drawing a new FMR trajectory."""
        if self.current_mode == 'fmr':
            self.current_trajectory_name = None  # This will create new trajectory on next point
    
    def get_trajectory_names(self) -> List[str]:
        """Get list of all trajectory names."""
        return list(self.trajectories.keys())
    
    def get_trajectory(self, name: str) -> Optional[Trajectory]:
        """Get trajectory by name."""
        return self.trajectories.get(name)
    
    def remove_trajectory(self, name: str) -> bool:
        """
        Remove entire trajectory.
        
        Returns:
        --------
        bool
            True if trajectory was removed
        """
        if name in self.trajectories:
            del self.trajectories[name]
            return True
        return False
    
    def clear_all(self):
        """Clear all trajectories and masks."""
        self.trajectories.clear()
        self.masks.clear()
        self.current_trajectory_name = None
        self._fmr_counter = 1
    
    def save_to_file(self, filename: str):
        """
        Save trajectories and masks to JSON file.
        
        Parameters:
        -----------
        filename : str
            Output filename
        """
        data = {
            'trajectories': {name: traj.to_dict() for name, traj in self.trajectories.items()},
            'masks': [mask.to_dict() for mask in self.masks],
            'metadata': {
                'version': '1.0',
                'fmr_counter': self._fmr_counter
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Trajectories and masks saved to: {filename}")
    
    def load_from_file(self, filename: str) -> bool:
        """
        Load trajectories and masks from JSON file.
        
        Parameters:
        -----------
        filename : str
            Input filename
            
        Returns:
        --------
        bool
            True if loaded successfully
        """
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return False
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Clear current data
            self.clear_all()
            
            # Load trajectories
            if 'trajectories' in data:
                for name, traj_data in data['trajectories'].items():
                    self.trajectories[name] = Trajectory.from_dict(traj_data)
            
            # Load masks
            if 'masks' in data:
                for mask_data in data['masks']:
                    self.masks.append(MaskArea.from_dict(mask_data))
            
            # Load metadata
            if 'metadata' in data:
                self._fmr_counter = data['metadata'].get('fmr_counter', 1)
            
            print(f"Loaded {len(self.trajectories)} trajectories and {len(self.masks)} masks from: {filename}")
            return True
            
        except Exception as e:
            print(f"Error loading file {filename}: {e}")
            return False
    
    def get_default_filename(self, base_name: str = "trajectories") -> str:
        """Get default filename for saving trajectories."""
        return f"{base_name}{FILE_FORMATS['trajectory_extension']}"
    
    def print_summary(self):
        """Print summary of current trajectories and masks."""
        print("\n" + "="*40)
        print("TRAJECTORY EDITOR SUMMARY")
        print("="*40)
        print(f"Current mode: {self.current_mode}")
        
        if self.trajectories:
            print(f"\nTrajectories ({len(self.trajectories)}):")
            for name, traj in self.trajectories.items():
                fields, freqs = traj.get_arrays()
                if len(fields) > 0:
                    print(f"  {name} ({traj.mode_type}): {len(traj.points)} points")
                    print(f"    Field range: {fields.min():.0f} - {fields.max():.0f} Oe")
                    print(f"    Freq range: {freqs.min():.3f} - {freqs.max():.3f} GHz")
                else:
                    print(f"  {name} ({traj.mode_type}): empty")
        else:
            print("\nNo trajectories defined")
        
        if self.masks:
            print(f"\nMasks ({len(self.masks)}):")
            for i, mask in enumerate(self.masks):
                print(f"  Mask {i+1}: Fields {mask.field_range[0]:.0f}-{mask.field_range[1]:.0f} Oe, "
                      f"Freqs {mask.freq_range[0]:.3f}-{mask.freq_range[1]:.3f} GHz")
        else:
            print("\nNo masks defined")
        
        print("="*40)