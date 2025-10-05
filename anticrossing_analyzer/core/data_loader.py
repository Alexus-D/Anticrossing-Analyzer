"""
Data loader module for S-parameter files.

This module handles loading of experimental S-parameter data from text files
with format: first row - frequencies, first column - magnetic fields,
data matrix - S-parameters in dB.
"""

import numpy as np
from typing import Tuple, Optional
import os
from ..config.settings import DATA_SETTINGS


class DataLoader:
    """
    Loads and processes S-parameter data from text files.
    
    Expected file format:
    - First row: frequency values (starting from second column)
    - First column: magnetic field values (starting from second row)
    - Data matrix: S-parameter values in dB
    """
    
    def __init__(self):
        self.frequencies = None
        self.fields = None
        self.s_parameters = None
        self.filename = None
        
    def load_data(self, filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load S-parameter data from text file.
        
        Parameters:
        -----------
        filename : str
            Path to the data file
            
        Returns:
        --------
        tuple
            (frequencies, fields, s_parameters) as numpy arrays
            
        Raises:
        -------
        FileNotFoundError
            If file doesn't exist
        ValueError
            If file format is incorrect
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Data file not found: {filename}")
            
        try:
            # Load the entire file as a 2D array
            data = np.loadtxt(filename)
            
            if data.ndim != 2:
                raise ValueError("Data file must contain a 2D array")
                
            if data.shape[0] < 2 or data.shape[1] < 2:
                raise ValueError("Data file must have at least 2 rows and 2 columns")
            
            # Extract frequencies (first row, excluding first element)
            self.frequencies = data[0, 1:]
            
            # Extract magnetic fields (first column, excluding first element)  
            self.fields = data[1:, 0]
            
            # Extract S-parameter matrix (excluding first row and column)
            self.s_parameters = data[1:, 1:]
            
            self.filename = filename
            
            # Validate dimensions match
            if self.s_parameters.shape != (len(self.fields), len(self.frequencies)):
                raise ValueError("S-parameter matrix dimensions don't match frequency/field arrays")
                
            print(f"Loaded data: {len(self.frequencies)} frequencies, {len(self.fields)} field points")
            print(f"Frequency range: {self.frequencies.min():.3f} - {self.frequencies.max():.3f} {DATA_SETTINGS['frequency_units']}")
            print(f"Field range: {self.fields.min():.0f} - {self.fields.max():.0f} {DATA_SETTINGS['field_units']}")
            print(f"S-parameter range: {self.s_parameters.min():.1f} - {self.s_parameters.max():.1f} {DATA_SETTINGS['s_parameter_units']}")
            
            return self.frequencies, self.fields, self.s_parameters
            
        except Exception as e:
            raise ValueError(f"Error loading data file {filename}: {str(e)}")
    
    def get_spectrum_at_field(self, field_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract S-parameter spectrum at specific magnetic field.
        
        Parameters:
        -----------
        field_value : float
            Magnetic field value
            
        Returns:
        --------
        tuple
            (frequencies, s21_values) at the specified field
        """
        if self.frequencies is None or self.fields is None or self.s_parameters is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Find closest field value
        field_idx = np.argmin(np.abs(self.fields - field_value))
        
        return self.frequencies.copy(), self.s_parameters[field_idx, :].copy()
    
    def get_field_sweep_at_frequency(self, frequency_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract S-parameter vs field at specific frequency.
        
        Parameters:
        -----------
        frequency_value : float
            Frequency value
            
        Returns:
        --------
        tuple
            (fields, s21_values) at the specified frequency
        """
        if self.frequencies is None or self.fields is None or self.s_parameters is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Find closest frequency value
        freq_idx = np.argmin(np.abs(self.frequencies - frequency_value))
        
        return self.fields.copy(), self.s_parameters[:, freq_idx].copy()
    
    def create_test_data(self, output_filename: str = "test_data.txt") -> str:
        """
        Create a simple test data file for development.
        
        Parameters:
        -----------
        output_filename : str
            Name of output file
            
        Returns:
        --------
        str
            Path to created file
        """
        # Create test frequency and field arrays
        frequencies = np.linspace(5.0, 7.0, 101)  # 5-7 GHz
        fields = np.linspace(1000, 3000, 81)  # 1000-3000 Oe (≈ 0.1-0.3 T)
        
        # Create artificial S21 data with simple resonance features
        freq_mesh, field_mesh = np.meshgrid(frequencies, fields)
        
        # Cavity resonance at ~6 GHz (field-independent)
        cavity_freq = 6.0
        cavity_width = 0.05
        cavity_response = -20 / (1 + ((freq_mesh - cavity_freq) / cavity_width)**2)
        
        # FMR resonance (field-dependent): f = gamma * H / (2*pi)
        # Use relation: f_fmr = 0.0028 * H (GHz = 0.0028 * Oe, γ/2π ≈ 2.8 MHz/Oe)
        fmr_freq = 0.0028 * field_mesh  # GHz = 0.0028 * Oe
        fmr_width = 0.03
        fmr_response = -15 / (1 + ((freq_mesh - fmr_freq) / fmr_width)**2)
        
        # Combine responses with some background
        s21_data = -5 + cavity_response + fmr_response + 0.5 * np.random.randn(*freq_mesh.shape)
        
        # Create output array in required format
        output_data = np.zeros((len(fields) + 1, len(frequencies) + 1))
        output_data[0, 0] = 0  # Corner element
        output_data[0, 1:] = frequencies  # First row: frequencies
        output_data[1:, 0] = fields  # First column: fields
        output_data[1:, 1:] = s21_data  # Data matrix
        
        # Save to file
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", output_filename)
        output_path = os.path.normpath(output_path)
        
        np.savetxt(output_path, output_data, fmt='%.6f')
        
        print(f"Test data created: {output_path}")
        return output_path
    
    @property
    def is_loaded(self) -> bool:
        """Check if data is loaded."""
        return all(x is not None for x in [self.frequencies, self.fields, self.s_parameters])
    
    @property
    def shape(self) -> Optional[Tuple[int, int]]:
        """Get shape of loaded data."""
        if self.is_loaded:
            return self.s_parameters.shape
        return None