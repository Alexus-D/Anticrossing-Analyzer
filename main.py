"""
Main entry point for Anticrossing Analyzer.

This file demonstrates basic usage of the data loading and visualization
functionality for Stage 1 of the project.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.contour import ContourSet
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from anticrossing_analyzer.core.data_loader import DataLoader
from anticrossing_analyzer.config.settings import PLOT_SETTINGS, DATA_SETTINGS


class AnticrossingAnalyzer:
    """
    Main class for anticrossing analysis.
    
    Currently implements Stage 1 functionality:
    - Data loading
    - Contour plot visualization
    """
    
    def __init__(self, data_file: str = None):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        data_file : str, optional
            Path to S-parameter data file
        """
        self.data_loader = DataLoader()
        self.data_file = data_file
        
        # Data arrays
        self.frequencies = None
        self.fields = None
        self.s_parameters = None
        
        # Plotting
        self.fig = None
        self.ax = None
        self.contour = None
        
    def load_data(self, filename: str = None) -> bool:
        """
        Load S-parameter data.
        
        Parameters:
        -----------
        filename : str, optional
            Data file path. If None, uses self.data_file
            
        Returns:
        --------
        bool
            True if data loaded successfully
        """
        if filename is None:
            filename = self.data_file
            
        if filename is None:
            print("No data file specified")
            return False
            
        try:
            self.frequencies, self.fields, self.s_parameters = self.data_loader.load_data(filename)
            self.data_file = filename
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def create_test_data(self) -> str:
        """
        Create test data file for development.
        
        Returns:
        --------
        str
            Path to created test file
        """
        return self.data_loader.create_test_data("test_anticrossing_data.txt")
    
    def show_contour_plot(self, save_figure: bool = False) -> None:
        """
        Display contour plot of S-parameter data.
        
        Parameters:
        -----------
        save_figure : bool
            Whether to save the figure to file
        """
        if not self.data_loader.is_loaded:
            print("No data loaded. Use load_data() first.")
            return
            
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=PLOT_SETTINGS['figsize'])
        
        # Create meshgrid for contour plotting
        freq_mesh, field_mesh = np.meshgrid(self.frequencies, self.fields)
        
        # Create contour plot
        self.contour = self.ax.contourf(
            freq_mesh, field_mesh, self.s_parameters,
            levels=PLOT_SETTINGS['contour_levels'],
            cmap=PLOT_SETTINGS['colormap']
        )
        
        # Add colorbar
        cbar = plt.colorbar(self.contour, ax=self.ax)
        cbar.set_label(f'S₂₁ ({DATA_SETTINGS["s_parameter_units"]})', fontsize=12)
        
        # Labels and title
        self.ax.set_xlabel(f'Frequency ({DATA_SETTINGS["frequency_units"]})', fontsize=12)
        self.ax.set_ylabel(f'Magnetic Field ({DATA_SETTINGS["field_units"]})', fontsize=12)
        self.ax.set_title('S₂₁ Parameter vs Frequency and Magnetic Field', fontsize=14)
        
        # Grid
        self.ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Save if requested
        if save_figure:
            filename = os.path.splitext(os.path.basename(self.data_file))[0] + '_contour.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Figure saved as: {filename}")
        
        plt.show()
    
    def show_example_spectra(self, n_spectra: int = 5) -> None:
        """
        Show example S21 spectra at different magnetic fields.
        
        Parameters:
        -----------
        n_spectra : int
            Number of spectra to display
        """
        if not self.data_loader.is_loaded:
            print("No data loaded. Use load_data() first.")
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Select field values evenly spaced across the range
        field_indices = np.linspace(0, len(self.fields)-1, n_spectra, dtype=int)
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_spectra))
        
        for i, field_idx in enumerate(field_indices):
            field_value = self.fields[field_idx]
            s21_spectrum = self.s_parameters[field_idx, :]
            
            ax.plot(self.frequencies, s21_spectrum, 
                   color=colors[i], linewidth=2,
                   label=f'H = {field_value:.1f} {DATA_SETTINGS["field_units"]}')
        
        ax.set_xlabel(f'Frequency ({DATA_SETTINGS["frequency_units"]})', fontsize=12)
        ax.set_ylabel(f'S₂₁ ({DATA_SETTINGS["s_parameter_units"]})', fontsize=12)
        ax.set_title('S₂₁ Spectra at Different Magnetic Fields', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def print_data_info(self) -> None:
        """Print information about loaded data."""
        if not self.data_loader.is_loaded:
            print("No data loaded.")
            return
            
        print("\n" + "="*50)
        print("DATA INFORMATION")
        print("="*50)
        print(f"File: {self.data_file}")
        print(f"Data shape: {self.data_loader.shape}")
        print(f"Frequency range: {self.frequencies.min():.3f} - {self.frequencies.max():.3f} {DATA_SETTINGS['frequency_units']}")
        print(f"Field range: {self.fields.min():.1f} - {self.fields.max():.1f} {DATA_SETTINGS['field_units']}")
        print(f"S21 range: {self.s_parameters.min():.1f} - {self.s_parameters.max():.1f} {DATA_SETTINGS['s_parameter_units']}")
        print("="*50)


def main():
    """
    Main function demonstrating Stage 1 functionality.
    """
    print("Anticrossing Analyzer - Stage 1 Demo")
    print("=====================================")
    
    # Create analyzer instance
    analyzer = AnticrossingAnalyzer()
    
    # Create test data
    print("\n1. Creating test data...")
    test_file = analyzer.create_test_data()
    
    # Load the test data
    print("\n2. Loading test data...")
    if analyzer.load_data(test_file):
        print("✓ Data loaded successfully")
    else:
        print("✗ Failed to load data")
        return
    
    # Print data information
    analyzer.print_data_info()
    
    # Show contour plot
    print("\n3. Displaying contour plot...")
    analyzer.show_contour_plot(save_figure=True)
    
    # Show example spectra
    print("\n4. Displaying example spectra...")
    analyzer.show_example_spectra(n_spectra=5)
    
    print("\n✓ Stage 1 completed successfully!")
    print("\nNext: Implement interactive trajectory editor (Stage 2)")


if __name__ == "__main__":
    main()