"""
Project settings and configuration.
"""

import numpy as np

# Default plot settings
PLOT_SETTINGS = {
    'figsize': (12, 8),
    'contour_levels': 50,
    'colormap': 'viridis',
    'trajectory_linewidth': 2,
    'trajectory_markersize': 5,
    'mask_alpha': 0.3,
    'mask_color': 'red'
}

# Data processing settings
DATA_SETTINGS = {
    'frequency_units': 'GHz',
    'field_units': 'Oe',  # Oersted - units for magnetic field H
    's_parameter_units': 'dB',
    'smoothing_window': 3  # For trajectory smoothing only, not raw data
}

# Fitting settings
FIT_SETTINGS = {
    'max_iterations': 1000,
    'tolerance': 1e-8,
    'initial_guess_method': 'auto',
    'bounds_sigma': 3.0,  # Standard deviations for parameter bounds
    'r_squared_threshold': 0.8  # Minimum R² for accepting fit
}

# File formats
FILE_FORMATS = {
    'data_extension': '.txt',
    'trajectory_extension': '.json',
    'results_extension': '.json',
    'figure_format': 'png',
    'figure_dpi': 300
}

# Physical constants and default parameter ranges
PHYSICS = {
    # FMR relation: f = (γ/2π) * g * H, where γ/2π ≈ 2.8 MHz/Oe for free electron
    'gamma_gyro': 2.8,  # Gyromagnetic ratio γ/2π in MHz/Oe 
    'g_factor': 2.0,  # Landé g-factor for free electron
    'typical_coupling': 1e-3,  # Typical G value in GHz
    'typical_linewidth': 1e-3,  # Typical gamma in GHz
    'frequency_range': (1.0, 20.0),  # GHz
    'field_range': (0.0, 10000.0)  # Oe (0-1 Tesla ≈ 0-10000 Oe)
}

# GUI settings
GUI_SETTINGS = {
    'click_tolerance': 5,  # pixels
    'trajectory_colors': ['blue', 'red', 'green', 'orange', 'purple'],
    'cavity_color': 'blue',
    'fmr_color': 'red',
    'mask_color': 'gray',
    'point_removal_key': 'd',  # Key to delete points
    'mode_switch_keys': {
        'c': 'cavity',
        'f': 'fmr', 
        'm': 'mask'
    }
}