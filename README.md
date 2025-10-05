# Anticrossing Analyzer

Tool for analyzing FMR (Ferromagnetic Resonance) anticrossing in experimental S-parameter data.

## Overview

This project provides an interactive tool for analyzing anticrossing between FMR modes and cavity resonator modes in experimental S-parameter measurements. The tool combines user-guided trajectory marking with automatic theoretical model fitting to extract physical parameters like coupling strengths and damping coefficients.

## Features

- **Interactive Trajectory Marking**: Mark mode trajectories directly on contour plots
- **Automatic Anticrossing Detection**: Identify anticrossing regions from trajectory intersections  
- **Theoretical Model Fitting**: Extract physical parameters using rigorous theoretical formulas
- **Multi-mode Support**: Handle arbitrary number of FMR modes interacting with cavity
- **Data Masking**: Exclude problematic data regions from analysis
- **Professional Results Export**: Save results in multiple formats with publication-quality figures

## Project Structure

```
anticrossing_analyzer/
├── __init__.py
├── main.py                          # Main entry point
├── config/
│   ├── __init__.py
│   └── settings.py                  # Project configuration
├── core/
│   ├── __init__.py
│   ├── data_loader.py              # S-parameter data loading
│   ├── trajectory_editor.py        # Interactive trajectory marking
│   ├── fitting_engine.py           # Model fitting engine
│   └── results_manager.py          # Results processing and export
├── models/
│   ├── __init__.py
│   ├── theoretical_functions.py    # Theoretical anticrossing formulas
│   └── model_selector.py           # Model selection logic
├── visualization/
│   ├── __init__.py
│   ├── interactive_interface.py    # GUI for trajectory marking
│   └── results_plotter.py         # Results visualization
└── utils/
    ├── __init__.py
    ├── math_utils.py              # Mathematical utilities
    └── trajectory_utils.py        # Trajectory processing
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Alexus-D/Anticrossing-Analyzer.git
cd Anticrossing-Analyzer
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install numpy scipy matplotlib pandas
```

## Usage

### Basic Usage

```python
from anticrossing_analyzer import AnticrossingAnalyzer

# Load experimental data
analyzer = AnticrossingAnalyzer('your_data.txt')
analyzer.load_data()

# Display contour plot
analyzer.show_contour_plot()

# Start interactive trajectory editor (future feature)
# analyzer.start_interactive_editor()

# Analyze anticrossing (future feature)
# results = analyzer.analyze_with_trajectories()
```

### Data Format

Input data files should be text files with the following format:
- First row: frequency values (GHz)
- First column: magnetic field values (mT)  
- Data matrix: S₂₁ parameter values (dB)

Example:
```
0.0     5.000   5.020   5.040   ...
100.0   -25.1   -24.8   -23.2   ...  
102.5   -26.3   -25.1   -24.1   ...
...
```

## Development Status

### ✅ Stage 1: Basic Structure and Data Adapter (COMPLETED)
- [x] Project structure created
- [x] Configuration system implemented
- [x] S-parameter data loading functionality
- [x] Basic contour plot visualization
- [x] Test data generation

### 🚧 Stage 2: Interactive Trajectory Editor (IN PROGRESS)
- [ ] Interactive GUI for trajectory marking
- [ ] Support for cavity and FMR mode trajectories
- [ ] Data masking functionality
- [ ] Trajectory save/load system

### 📋 Upcoming Stages
- Stage 3: Test data generation
- Stage 4: Theoretical model implementation
- Stage 5: Basic spectrum fitting
- Stage 6: Results visualization
- Stage 7: Multi-mode support
- Stage 8: Results export and reporting

## Physical Model

The tool fits experimental spectra to the theoretical anticrossing formula:

```
S₂₁ ∝ 1 - (iγce) / (ω - ωc + i(γce + γci) + G²/(ω - ωm + iγm))
```

Where:
- ω: frequency
- ωc: cavity resonance frequency  
- ωm: FMR frequency (field-dependent)
- γce, γci: cavity damping parameters
- γm: FMR damping parameter
- G: coupling strength between modes

## Requirements

- Python 3.7+
- NumPy
- SciPy  
- Matplotlib
- Pandas (optional, for data export)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this tool in your research, please cite:

```
[Citation will be added when paper is published]
```

## Contact

For questions and support, please open an issue on GitHub or contact the development team.