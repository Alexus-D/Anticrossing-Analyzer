"""
Trajectory utilities for processing and manipulating trajectory data.

This module provides functions for interpolating, smoothing, and analyzing
trajectory data marked by users.
"""

import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from typing import Tuple, List, Optional, Dict, Any
from ..config.settings import DATA_SETTINGS


def interpolate_trajectory(fields: np.ndarray, frequencies: np.ndarray, 
                         target_fields: np.ndarray, method: str = 'cubic') -> np.ndarray:
    """
    Interpolate trajectory to target field values.
    
    Parameters:
    -----------
    fields : np.ndarray
        Original field values (must be sorted)
    frequencies : np.ndarray
        Original frequency values
    target_fields : np.ndarray
        Target field values for interpolation
    method : str
        Interpolation method: 'linear', 'cubic', 'pchip'
        
    Returns:
    --------
    np.ndarray
        Interpolated frequencies at target fields
    """
    if len(fields) < 2:
        return np.full_like(target_fields, np.nan)
    
    # Ensure data is sorted by field
    sort_idx = np.argsort(fields)
    fields_sorted = fields[sort_idx]
    frequencies_sorted = frequencies[sort_idx]
    
    # Remove duplicates
    unique_mask = np.diff(fields_sorted, prepend=fields_sorted[0]-1) != 0
    fields_unique = fields_sorted[unique_mask]
    frequencies_unique = frequencies_sorted[unique_mask]
    
    if len(fields_unique) < 2:
        return np.full_like(target_fields, np.nan)
    
    # Perform interpolation
    try:
        if method == 'linear':
            interp_func = interpolate.interp1d(fields_unique, frequencies_unique, 
                                             kind='linear', bounds_error=False, 
                                             fill_value=np.nan)
        elif method == 'cubic':
            if len(fields_unique) >= 4:
                interp_func = interpolate.interp1d(fields_unique, frequencies_unique,
                                                 kind='cubic', bounds_error=False,
                                                 fill_value=np.nan)
            else:
                # Fall back to linear for insufficient points
                interp_func = interpolate.interp1d(fields_unique, frequencies_unique,
                                                 kind='linear', bounds_error=False,
                                                 fill_value=np.nan)
        elif method == 'pchip':
            interp_func = interpolate.PchipInterpolator(fields_unique, frequencies_unique,
                                                      extrapolate=False)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        return interp_func(target_fields)
        
    except Exception as e:
        print(f"Interpolation failed: {e}")
        return np.full_like(target_fields, np.nan)


def smooth_trajectory(fields: np.ndarray, frequencies: np.ndarray, 
                     sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth trajectory using Gaussian filter.
    
    Parameters:
    -----------
    fields : np.ndarray
        Field values
    frequencies : np.ndarray
        Frequency values  
    sigma : float
        Standard deviation for Gaussian kernel
        
    Returns:
    --------
    tuple
        (smoothed_fields, smoothed_frequencies)
    """
    if len(fields) < 3:
        return fields.copy(), frequencies.copy()
    
    # Sort by field
    sort_idx = np.argsort(fields)
    fields_sorted = fields[sort_idx]
    frequencies_sorted = frequencies[sort_idx]
    
    # Apply Gaussian smoothing to frequencies
    frequencies_smooth = gaussian_filter1d(frequencies_sorted, sigma=sigma)
    
    return fields_sorted, frequencies_smooth


def find_trajectory_intersections(traj1_fields: np.ndarray, traj1_freqs: np.ndarray,
                                traj2_fields: np.ndarray, traj2_freqs: np.ndarray,
                                tolerance: float = 0.05) -> List[Tuple[float, float]]:
    """
    Find intersections between two trajectories.
    
    Parameters:
    -----------
    traj1_fields, traj1_freqs : np.ndarray
        First trajectory
    traj2_fields, traj2_freqs : np.ndarray
        Second trajectory
    tolerance : float
        Tolerance for intersection detection (in GHz)
        
    Returns:
    --------
    list
        List of (field, frequency) intersection points
    """
    intersections = []
    
    # Find overlapping field range
    field_min = max(traj1_fields.min(), traj2_fields.min())
    field_max = min(traj1_fields.max(), traj2_fields.max())
    
    if field_min >= field_max:
        return intersections
    
    # Create common field grid
    n_points = 100
    common_fields = np.linspace(field_min, field_max, n_points)
    
    # Interpolate both trajectories to common grid
    freq1_interp = interpolate_trajectory(traj1_fields, traj1_freqs, common_fields)
    freq2_interp = interpolate_trajectory(traj2_fields, traj2_freqs, common_fields)
    
    # Find crossings
    freq_diff = freq1_interp - freq2_interp
    
    # Look for sign changes in frequency difference
    for i in range(len(freq_diff) - 1):
        if (not np.isnan(freq_diff[i]) and not np.isnan(freq_diff[i+1]) and
            freq_diff[i] * freq_diff[i+1] < 0):  # Sign change
            
            # Linear interpolation to find exact crossing point
            alpha = abs(freq_diff[i]) / (abs(freq_diff[i]) + abs(freq_diff[i+1]))
            field_cross = common_fields[i] + alpha * (common_fields[i+1] - common_fields[i])
            freq_cross = freq1_interp[i] + alpha * (freq1_interp[i+1] - freq1_interp[i])
            
            intersections.append((field_cross, freq_cross))
    
    return intersections


def estimate_trajectory_parameters(fields: np.ndarray, frequencies: np.ndarray,
                                 trajectory_type: str = 'fmr') -> Dict[str, float]:
    """
    Estimate physical parameters from trajectory.
    
    Parameters:
    -----------
    fields : np.ndarray
        Field values (Oe)
    frequencies : np.ndarray
        Frequency values (GHz)
    trajectory_type : str
        Type of trajectory: 'fmr' or 'cavity'
        
    Returns:
    --------
    dict
        Estimated parameters
    """
    params = {}
    
    if trajectory_type == 'fmr':
        # For FMR: f = f0 + gamma * H
        # Fit linear relationship
        if len(fields) >= 2:
            # Linear fit
            coeffs = np.polyfit(fields, frequencies, 1)
            gamma_fit = coeffs[0]  # GHz/Oe
            f0_fit = coeffs[1]     # GHz
            
            params['gamma'] = gamma_fit
            params['f0'] = f0_fit
            params['field_range'] = (fields.min(), fields.max())
            params['freq_range'] = (frequencies.min(), frequencies.max())
            
            # Calculate R²
            freq_pred = np.polyval(coeffs, fields)
            ss_res = np.sum((frequencies - freq_pred) ** 2)
            ss_tot = np.sum((frequencies - np.mean(frequencies)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            params['r_squared'] = r_squared
            
    elif trajectory_type == 'cavity':
        # For cavity: approximately constant frequency
        params['freq_mean'] = np.mean(frequencies)
        params['freq_std'] = np.std(frequencies)
        params['field_range'] = (fields.min(), fields.max())
        params['freq_range'] = (frequencies.min(), frequencies.max())
    
    return params


def validate_trajectory(fields: np.ndarray, frequencies: np.ndarray,
                       trajectory_type: str = 'fmr') -> Dict[str, Any]:
    """
    Validate trajectory data quality.
    
    Parameters:
    -----------
    fields : np.ndarray
        Field values
    frequencies : np.ndarray
        Frequency values
    trajectory_type : str
        Type of trajectory
        
    Returns:
    --------
    dict
        Validation results
    """
    validation = {
        'is_valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Basic checks
    if len(fields) < 2:
        validation['errors'].append("Trajectory has fewer than 2 points")
        validation['is_valid'] = False
        return validation
    
    # Check for monotonic field values
    field_diffs = np.diff(fields)
    if not (np.all(field_diffs >= 0) or np.all(field_diffs <= 0)):
        validation['warnings'].append("Field values are not monotonic")
    
    # Check for reasonable frequency range
    freq_range = frequencies.max() - frequencies.min()
    if freq_range > 5.0:  # GHz
        validation['warnings'].append(f"Large frequency range: {freq_range:.2f} GHz")
    
    # Type-specific checks
    if trajectory_type == 'fmr':
        # Check for reasonable slope
        params = estimate_trajectory_parameters(fields, frequencies, 'fmr')
        if 'gamma' in params:
            gamma = params['gamma']
            expected_gamma = 0.0028  # GHz/Oe for g=2
            if abs(gamma - expected_gamma) > expected_gamma * 0.5:
                validation['warnings'].append(
                    f"Unusual FMR slope: {gamma:.4f} GHz/Oe (expected ~{expected_gamma:.4f})")
    
    elif trajectory_type == 'cavity':
        # Check for approximately constant frequency
        freq_variation = np.std(frequencies) / np.mean(frequencies)
        if freq_variation > 0.05:  # 5% variation
            validation['warnings'].append(
                f"Large cavity frequency variation: {freq_variation*100:.1f}%")
    
    return validation


def trajectory_to_initial_guess(fields: np.ndarray, frequencies: np.ndarray,
                               target_field: float, trajectory_type: str = 'fmr') -> float:
    """
    Get initial guess for fitting parameter from trajectory.
    
    Parameters:
    -----------
    fields : np.ndarray
        Trajectory field values
    frequencies : np.ndarray
        Trajectory frequency values
    target_field : float
        Field value for which to estimate frequency
    trajectory_type : str
        Type of trajectory
        
    Returns:
    --------
    float
        Estimated frequency at target field
    """
    if len(fields) < 2:
        return np.nan
    
    # Interpolate to target field
    freq_estimate = interpolate_trajectory(fields, frequencies, 
                                         np.array([target_field]))[0]
    
    return freq_estimate if not np.isnan(freq_estimate) else np.mean(frequencies)