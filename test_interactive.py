"""
Test script for interactive trajectory editor (Stage 2).

This script demonstrates the interactive interface functionality
without going through all the previous stages.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from anticrossing_analyzer.core.data_loader import DataLoader
from anticrossing_analyzer.visualization.interactive_interface import create_interactive_editor


def test_interactive_editor():
    """Test the interactive trajectory editor."""
    print("Interactive Trajectory Editor Test")
    print("==================================")
    
    # Create and load test data
    print("1. Creating test data...")
    loader = DataLoader()
    test_file = loader.create_test_data("test_interactive.txt")
    
    print("2. Loading data...")
    frequencies, fields, s_parameters = loader.load_data(test_file)
    
    print("3. Starting interactive editor...")
    print("\nINSTRUCTIONS:")
    print("• Press 'H' for help when the plot window is active")
    print("• Press 'C' for Cavity mode, 'F' for FMR mode, 'M' for Mask mode")
    print("• Left click to add points/define masks")
    print("• Right click to remove closest point")
    print("• Press 'N' to start new FMR trajectory")
    print("• Press 'I' to show trajectory info")
    print("• Close the window when done")
    
    # Create and show interactive interface
    interface = create_interactive_editor(frequencies, fields, s_parameters)
    interface.show()
    
    # Print final summary
    print("\n4. Final summary:")
    interface.get_editor().print_summary()
    
    print("\n✓ Interactive editor test completed!")


if __name__ == "__main__":
    test_interactive_editor()