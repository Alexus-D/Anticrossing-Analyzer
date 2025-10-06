"""
Mathematical utilities for anticrossing analysis.

This module provides common mathematical functions used throughout
the anticrossing analysis package.
"""

import numpy as np
from scipy import optimize
from typing import Tuple, Optional, Callable, Any


def complex_s21_to_db(s21_complex: np.ndarray) -> np.ndarray:
    """
    Convert complex S21 to magnitude in dB.
    
    Parameters:
    -----------
    s21_complex : np.ndarray
        Complex S21 values
        
    Returns:
    --------
    np.ndarray
        S21 magnitude in dB
    """
    return 20 * np.log10(np.abs(s21_complex))


def db_to_complex_s21(s21_db: np.ndarray, phase: np.ndarray = None) -> np.ndarray:
    """
    Convert S21 in dB to complex values.
    
    Parameters:
    -----------
    s21_db : np.ndarray
        S21 magnitude in dB
    phase : np.ndarray, optional
        Phase in radians. If None, assumes zero phase.
        
    Returns:
    --------
    np.ndarray
        Complex S21 values
    """
    magnitude = 10**(s21_db / 20)
    if phase is None:
        return magnitude
    else:
        return magnitude * np.exp(1j * phase)


def lorentzian(x: np.ndarray, x0: float, gamma: float, amplitude: float = 1.0) -> np.ndarray:
    """
    Lorentzian function.
    
    Parameters:
    -----------
    x : np.ndarray
        Input variable
    x0 : float
        Center position
    gamma : float
        Half-width at half-maximum (HWHM)
    amplitude : float
        Peak amplitude
        
    Returns:
    --------
    np.ndarray
        Lorentzian values
    """
    return amplitude * gamma**2 / ((x - x0)**2 + gamma**2)


def complex_lorentzian(x: np.ndarray, x0: float, gamma: float) -> np.ndarray:
    """
    Complex Lorentzian function: 1 / (x - x0 + 1j*gamma)
    
    Parameters:
    -----------
    x : np.ndarray
        Input variable (typically frequency)
    x0 : float
        Resonance frequency
    gamma : float
        Damping parameter
        
    Returns:
    --------
    np.ndarray
        Complex Lorentzian values
    """
    return 1.0 / (x - x0 + 1j * gamma)


def calculate_r_squared(y_data: np.ndarray, y_fit: np.ndarray) -> float:
    """
    Calculate coefficient of determination (R²).
    
    Parameters:
    -----------
    y_data : np.ndarray
        Experimental data
    y_fit : np.ndarray
        Fitted data
        
    Returns:
    --------
    float
        R² value
    """
    # Remove NaN values
    mask = ~(np.isnan(y_data) | np.isnan(y_fit))
    if np.sum(mask) < 2:
        return 0.0
    
    y_data_clean = y_data[mask]
    y_fit_clean = y_fit[mask]
    
    ss_res = np.sum((y_data_clean - y_fit_clean)**2)
    ss_tot = np.sum((y_data_clean - np.mean(y_data_clean))**2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1.0 - ss_res / ss_tot


def calculate_chi_squared(y_data: np.ndarray, y_fit: np.ndarray, 
                         uncertainties: np.ndarray = None) -> float:
    """
    Calculate chi-squared goodness of fit.
    
    Parameters:
    -----------
    y_data : np.ndarray
        Experimental data
    y_fit : np.ndarray
        Fitted data
    uncertainties : np.ndarray, optional
        Data uncertainties. If None, assumes uniform uncertainties.
        
    Returns:
    --------
    float
        Chi-squared value
    """
    # Remove NaN values
    mask = ~(np.isnan(y_data) | np.isnan(y_fit))
    if np.sum(mask) == 0:
        return np.inf
    
    y_data_clean = y_data[mask]
    y_fit_clean = y_fit[mask]
    
    if uncertainties is None:
        uncertainties_clean = np.ones_like(y_data_clean)
    else:
        uncertainties_clean = uncertainties[mask]
        uncertainties_clean = np.where(uncertainties_clean > 0, uncertainties_clean, 1.0)
    
    chi_squared = np.sum(((y_data_clean - y_fit_clean) / uncertainties_clean)**2)
    
    return chi_squared


def reduced_chi_squared(y_data: np.ndarray, y_fit: np.ndarray, 
                       n_params: int, uncertainties: np.ndarray = None) -> float:
    """
    Calculate reduced chi-squared.
    
    Parameters:
    -----------
    y_data : np.ndarray
        Experimental data
    y_fit : np.ndarray
        Fitted data
    n_params : int
        Number of fitting parameters
    uncertainties : np.ndarray, optional
        Data uncertainties
        
    Returns:
    --------
    float
        Reduced chi-squared value
    """
    chi2 = calculate_chi_squared(y_data, y_fit, uncertainties)
    n_data = np.sum(~np.isnan(y_data))
    degrees_of_freedom = max(1, n_data - n_params)
    
    return chi2 / degrees_of_freedom


def estimate_uncertainties_from_residuals(y_data: np.ndarray, y_fit: np.ndarray,
                                        window_size: int = 5) -> np.ndarray:
    """
    Estimate data uncertainties from fit residuals.
    
    Parameters:
    -----------
    y_data : np.ndarray
        Experimental data
    y_fit : np.ndarray
        Fitted data
    window_size : int
        Window size for local uncertainty estimation
        
    Returns:
    --------
    np.ndarray
        Estimated uncertainties
    """
    residuals = y_data - y_fit
    uncertainties = np.zeros_like(y_data)
    
    for i in range(len(y_data)):
        # Define window around point i
        start = max(0, i - window_size // 2)
        end = min(len(y_data), i + window_size // 2 + 1)
        
        # Calculate local standard deviation
        local_residuals = residuals[start:end]
        local_std = np.std(local_residuals[~np.isnan(local_residuals)])
        
        uncertainties[i] = local_std if local_std > 0 else np.std(residuals[~np.isnan(residuals)])
    
    return uncertainties


def find_peaks_and_dips(y_data: np.ndarray, x_data: np.ndarray = None, 
                       prominence: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find peaks and dips in data.
    
    Parameters:
    -----------
    y_data : np.ndarray
        Data values
    x_data : np.ndarray, optional
        X-axis values. If None, uses indices.
    prominence : float, optional
        Minimum prominence for peak detection
        
    Returns:
    --------
    tuple
        (peak_positions, dip_positions)
    """
    if x_data is None:
        x_data = np.arange(len(y_data))
    
    # Find peaks (maxima)
    from scipy.signal import find_peaks
    
    if prominence is None:
        prominence = (np.max(y_data) - np.min(y_data)) * 0.1
    
    peak_indices, _ = find_peaks(y_data, prominence=prominence)
    dip_indices, _ = find_peaks(-y_data, prominence=prominence)
    
    peak_positions = x_data[peak_indices] if len(peak_indices) > 0 else np.array([])
    dip_positions = x_data[dip_indices] if len(dip_indices) > 0 else np.array([])
    
    return peak_positions, dip_positions


def smooth_data(y_data: np.ndarray, window_size: int = 5, method: str = 'gaussian') -> np.ndarray:
    """
    Smooth data using various methods.
    
    Parameters:
    -----------
    y_data : np.ndarray
        Input data
    window_size : int
        Size of smoothing window
    method : str
        Smoothing method: 'gaussian', 'moving_average', 'savgol'
        
    Returns:
    --------
    np.ndarray
        Smoothed data
    """
    if method == 'gaussian':
        from scipy.ndimage import gaussian_filter1d
        sigma = window_size / 3.0  # Standard deviation
        return gaussian_filter1d(y_data, sigma=sigma)
    
    elif method == 'moving_average':
        # Simple moving average
        kernel = np.ones(window_size) / window_size
        return np.convolve(y_data, kernel, mode='same')
    
    elif method == 'savgol':
        from scipy.signal import savgol_filter
        # Savitzky-Golay filter
        poly_order = min(3, window_size - 1)
        if window_size % 2 == 0:
            window_size += 1  # Must be odd
        return savgol_filter(y_data, window_size, poly_order)
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def normalize_data(data: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Normalize data using various methods.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    method : str
        Normalization method: 'minmax', 'zscore', 'robust'
        
    Returns:
    --------
    tuple
        (normalized_data, normalization_params)
    """
    params = {}
    
    if method == 'minmax':
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max == data_min:
            normalized = np.zeros_like(data)
        else:
            normalized = (data - data_min) / (data_max - data_min)
        params = {'min': data_min, 'max': data_max}
    
    elif method == 'zscore':
        data_mean = np.mean(data)
        data_std = np.std(data)
        if data_std == 0:
            normalized = np.zeros_like(data)
        else:
            normalized = (data - data_mean) / data_std
        params = {'mean': data_mean, 'std': data_std}
    
    elif method == 'robust':
        data_median = np.median(data)
        data_mad = np.median(np.abs(data - data_median))  # Median absolute deviation
        if data_mad == 0:
            normalized = np.zeros_like(data)
        else:
            normalized = (data - data_median) / (1.4826 * data_mad)  # 1.4826 for normal distribution
        params = {'median': data_median, 'mad': data_mad}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def confidence_interval(parameter_value: float, parameter_error: float, 
                       confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for a parameter.
    
    Parameters:
    -----------
    parameter_value : float
        Parameter estimate
    parameter_error : float
        Parameter uncertainty (standard error)
    confidence_level : float
        Confidence level (default 0.95 for 95%)
        
    Returns:
    --------
    tuple
        (lower_bound, upper_bound)
    """
    from scipy.stats import t
    
    # For large degrees of freedom, t-distribution approaches normal
    # Using t-distribution with 100 degrees of freedom as approximation
    df = 100
    alpha = 1 - confidence_level
    t_value = t.ppf(1 - alpha/2, df)
    
    margin = t_value * parameter_error
    
    return (parameter_value - margin, parameter_value + margin)