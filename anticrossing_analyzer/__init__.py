"""
Anticrossing Analyzer - Tool for analyzing FMR anticrossing in S-parameter data.

This package provides tools for interactive trajectory marking and fitting
of theoretical anticrossing models to experimental S-parameter data.
"""

__version__ = "0.1.0"
__author__ = "Alexey Kaminskiy"

from .core.data_loader import DataLoader
from .core.trajectory_editor import TrajectoryEditor
from .core.fitting_engine import FittingEngine
from .core.results_manager import ResultsManager

__all__ = [
    'DataLoader',
    'TrajectoryEditor', 
    'FittingEngine',
    'ResultsManager'
]