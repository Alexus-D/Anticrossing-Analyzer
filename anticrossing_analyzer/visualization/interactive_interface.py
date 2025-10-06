"""
Interactive interface for trajectory marking on contour plots.

This module provides a matplotlib-based GUI for interactive trajectory editing.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Optional, Tuple, Callable, Dict, Any
from matplotlib.backend_bases import MouseEvent, KeyEvent

from ..core.trajectory_editor import TrajectoryEditor, Trajectory, MaskArea
from ..config.settings import PLOT_SETTINGS, GUI_SETTINGS, DATA_SETTINGS


class InteractiveInterface:
    """
    Interactive matplotlib interface for trajectory and mask editing.
    
    Provides mouse and keyboard interaction for marking trajectories and masks
    on contour plots of S-parameter data.
    """
    
    def __init__(self, frequencies: np.ndarray, fields: np.ndarray, s_parameters: np.ndarray):
        """
        Initialize interactive interface.
        
        Parameters:
        -----------
        frequencies : np.ndarray
            Frequency values (x-axis)
        fields : np.ndarray  
            Magnetic field values (y-axis)
        s_parameters : np.ndarray
            S-parameter data matrix (2D)
        """
        self.frequencies = frequencies
        self.fields = fields  
        self.s_parameters = s_parameters
        
        # Trajectory editor
        self.editor = TrajectoryEditor()
        
        # Matplotlib objects
        self.fig = None
        self.ax = None
        self.contour = None
        
        # Interactive state
        self.is_active = False
        self.mask_start_point = None  # For drawing rectangular masks
        self.mask_rectangle = None  # Temporary rectangle for mask preview
        
        # Plotted objects for dynamic updates
        self.trajectory_lines = {}  # trajectory_name -> Line2D object
        self.trajectory_points = {}  # trajectory_name -> scatter object  
        self.mask_patches = []  # List of Rectangle patches
        
        # Event handlers
        self.click_handler = None
        self.key_handler = None
        
        # Callbacks
        self.on_trajectory_change: Optional[Callable] = None
        
    def create_plot(self, figsize: Tuple[int, int] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create the base contour plot.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height)
            
        Returns:
        --------
        tuple
            (figure, axes) objects
        """
        if figsize is None:
            figsize = PLOT_SETTINGS['figsize']
        
        self.fig, self.ax = plt.subplots(figsize=figsize)
        
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
        self.ax.set_title('Interactive Trajectory Editor - Click to add points', fontsize=14)
        
        # Grid
        self.ax.grid(True, alpha=0.3)
        
        # Add mode indicator text
        self._update_title()
        
        return self.fig, self.ax
    
    def activate_interaction(self):
        """Activate mouse and keyboard interaction."""
        if self.fig is None:
            raise ValueError("Must create plot first using create_plot()")
        
        self.is_active = True
        
        # Connect event handlers
        self.click_handler = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.key_handler = self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Make figure focusable for keyboard events
        self.fig.canvas.set_focus_on_draw(True)
        
        print("Interactive mode activated!")
        self._print_help()
    
    def deactivate_interaction(self):
        """Deactivate mouse and keyboard interaction."""
        if self.fig and self.is_active:
            if self.click_handler:
                self.fig.canvas.mpl_disconnect(self.click_handler)
            if self.key_handler:
                self.fig.canvas.mpl_disconnect(self.key_handler)
        
        self.is_active = False
        print("Interactive mode deactivated.")
    
    def _on_click(self, event: MouseEvent):
        """Handle mouse click events."""
        if not self.is_active or event.inaxes != self.ax:
            return
        
        field = event.ydata
        frequency = event.xdata
        
        if field is None or frequency is None:
            return
        
        if event.button == 1:  # Left click
            self._handle_left_click(field, frequency, event)
        elif event.button == 3:  # Right click
            self._handle_right_click(field, frequency, event)
    
    def _handle_left_click(self, field: float, frequency: float, event: MouseEvent):
        """Handle left mouse click."""
        if self.editor.current_mode == 'mask':
            self._handle_mask_click(field, frequency, event)
        else:
            # Add trajectory point
            self.editor.add_trajectory_point(field, frequency)
            self._update_trajectory_display()
            
            if self.on_trajectory_change:
                self.on_trajectory_change()
    
    def _handle_right_click(self, field: float, frequency: float, event: MouseEvent):
        """Handle right mouse click (remove points)."""
        if self.editor.current_mode != 'mask':
            # Remove closest trajectory point
            tolerance = self._calculate_click_tolerance()
            if self.editor.remove_trajectory_point(field, frequency, tolerance):
                self._update_trajectory_display()
                
                if self.on_trajectory_change:
                    self.on_trajectory_change()
    
    def _handle_mask_click(self, field: float, frequency: float, event: MouseEvent):
        """Handle clicks in mask mode (draw rectangular masks)."""
        if self.mask_start_point is None:
            # First click - start rectangle
            self.mask_start_point = (field, frequency)
            print(f"Mask start point: {field:.0f} Oe, {frequency:.3f} GHz")
        else:
            # Second click - complete rectangle
            start_field, start_freq = self.mask_start_point
            
            # Create mask area (ensure proper min/max order)
            field_range = (min(start_field, field), max(start_field, field))
            freq_range = (min(start_freq, frequency), max(start_freq, frequency))
            
            self.editor.add_mask_area(field_range, freq_range)
            self._update_mask_display()
            
            print(f"Mask added: Fields {field_range[0]:.0f}-{field_range[1]:.0f} Oe, "
                  f"Freqs {freq_range[0]:.3f}-{freq_range[1]:.3f} GHz")
            
            # Reset mask drawing state
            self.mask_start_point = None
            if self.mask_rectangle:
                self.mask_rectangle.remove()
                self.mask_rectangle = None
            
            if self.on_trajectory_change:
                self.on_trajectory_change()
    
    def _on_key_press(self, event: KeyEvent):
        """Handle keyboard events."""
        if not self.is_active:
            return
        
        key = event.key.lower()
        
        # Mode switching
        if key in GUI_SETTINGS['mode_switch_keys']:
            new_mode = GUI_SETTINGS['mode_switch_keys'][key]
            self.editor.set_mode(new_mode)
            self._update_title()
            print(f"Switched to {new_mode} mode")
            
            # Reset mask drawing state when switching modes
            if self.mask_start_point:
                self.mask_start_point = None
                if self.mask_rectangle:
                    self.mask_rectangle.remove()
                    self.mask_rectangle = None
        
        # Start new FMR trajectory
        elif key == 'n' and self.editor.current_mode == 'fmr':
            self.editor.start_new_fmr_trajectory()
            print("Started new FMR trajectory")
        
        # Clear all
        elif key == 'ctrl+c':
            self.editor.clear_all()
            self._update_all_displays()
            print("Cleared all trajectories and masks")
        
        # Save/Load
        elif key == 'ctrl+s':
            filename = self.editor.get_default_filename()
            self.editor.save_to_file(filename)
        
        elif key == 'ctrl+o':
            filename = self.editor.get_default_filename()
            if self.editor.load_from_file(filename):
                self._update_all_displays()
        
        # Help
        elif key == 'h':
            self._print_help()
        
        # Summary
        elif key == 'i':
            self.editor.print_summary()
    
    def _update_title(self):
        """Update plot title with current mode information."""
        mode_info = {
            'cavity': "CAVITY mode - Click to add cavity trajectory points",
            'fmr': "FMR mode - Click to add FMR trajectory points (N=new trajectory)",  
            'mask': "MASK mode - Click twice to define rectangular mask area"
        }
        
        title = f"Interactive Trajectory Editor - {mode_info[self.editor.current_mode]}"
        if self.ax:
            self.ax.set_title(title, fontsize=12)
            self.fig.canvas.draw_idle()
    
    def _update_trajectory_display(self):
        """Update display of all trajectories."""
        # Remove old trajectory displays
        for line in self.trajectory_lines.values():
            if line in self.ax.lines:
                line.remove()
        for points in self.trajectory_points.values():
            if points in self.ax.collections:
                points.remove()
        
        self.trajectory_lines.clear()
        self.trajectory_points.clear()
        
        # Draw all trajectories
        for name, trajectory in self.editor.trajectories.items():
            if len(trajectory.points) == 0:
                continue
            
            fields, frequencies = trajectory.get_arrays()
            
            # Draw lines connecting points (if more than 1 point)
            if len(fields) > 1:
                line = self.ax.plot(frequencies, fields, 
                                  color=trajectory.color,
                                  linewidth=PLOT_SETTINGS['trajectory_linewidth'],
                                  label=f'{name} ({trajectory.mode_type})',
                                  alpha=0.8)[0]
                self.trajectory_lines[name] = line
            
            # Draw individual points
            points = self.ax.scatter(frequencies, fields,
                                   color=trajectory.color,
                                   s=PLOT_SETTINGS['trajectory_markersize']**2,
                                   edgecolors='white',
                                   linewidths=1,
                                   zorder=10)
            self.trajectory_points[name] = points
        
        # Update legend if there are trajectories
        if self.trajectory_lines:
            self.ax.legend(loc='upper right', fontsize=10)
        elif hasattr(self.ax, 'legend_') and self.ax.legend_:
            self.ax.legend_.remove()
        
        self.fig.canvas.draw_idle()
    
    def _update_mask_display(self):
        """Update display of all mask areas."""
        # Remove old mask displays
        for patch in self.mask_patches:
            if patch in self.ax.patches:
                patch.remove()
        self.mask_patches.clear()
        
        # Draw all masks
        for mask in self.editor.masks:
            field_range = mask.field_range
            freq_range = mask.freq_range
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (freq_range[0], field_range[0]),  # (x, y) = (freq, field)
                freq_range[1] - freq_range[0],    # width
                field_range[1] - field_range[0],  # height
                linewidth=1,
                edgecolor='red',
                facecolor=mask.color,
                alpha=mask.alpha,
                zorder=5
            )
            
            self.ax.add_patch(rect)
            self.mask_patches.append(rect)
        
        self.fig.canvas.draw_idle()
    
    def _update_all_displays(self):
        """Update display of trajectories and masks."""
        self._update_trajectory_display()
        self._update_mask_display()
    
    def _calculate_click_tolerance(self) -> float:
        """Calculate reasonable click tolerance based on current plot ranges."""
        freq_range = self.frequencies.max() - self.frequencies.min()
        field_range = self.fields.max() - self.fields.min()
        
        # Use small fraction of data ranges
        freq_tol = freq_range * 0.02
        field_tol = field_range * 0.02
        
        # Return combined tolerance (Euclidean distance)
        return np.sqrt(freq_tol**2 + field_tol**2)
    
    def _print_help(self):
        """Print help information."""
        help_text = """
=== INTERACTIVE TRAJECTORY EDITOR HELP ===

MOUSE CONTROLS:
  Left Click:   Add trajectory point (cavity/fmr modes) or define mask area
  Right Click:  Remove closest trajectory point

KEYBOARD SHORTCUTS:
  C:         Switch to Cavity mode
  F:         Switch to FMR mode  
  M:         Switch to Mask mode
  N:         Start new FMR trajectory (in FMR mode)
  H:         Show this help
  I:         Show trajectory summary
  Ctrl+S:    Save trajectories and masks
  Ctrl+O:    Load trajectories and masks
  Ctrl+C:    Clear all trajectories and masks

MODES:
  Cavity:    Draw cavity resonator trajectory (blue)
  FMR:       Draw FMR mode trajectories (colors vary)
  Mask:      Define rectangular areas to exclude from analysis

MASK MODE:
  Click twice to define rectangular mask area
  First click sets start corner, second click completes rectangle

==========================================
"""
        print(help_text)
    
    def get_editor(self) -> TrajectoryEditor:
        """Get the trajectory editor instance."""
        return self.editor
    
    def show(self):
        """Display the interactive plot."""
        if self.fig:
            plt.show()
    
    def save_figure(self, filename: str, dpi: int = 300):
        """Save current figure to file."""
        if self.fig:
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved as: {filename}")


def create_interactive_editor(frequencies: np.ndarray, fields: np.ndarray, 
                            s_parameters: np.ndarray) -> InteractiveInterface:
    """
    Create and setup interactive trajectory editor.
    
    Parameters:
    -----------
    frequencies : np.ndarray
        Frequency values
    fields : np.ndarray
        Magnetic field values
    s_parameters : np.ndarray
        S-parameter data matrix
        
    Returns:
    --------
    InteractiveInterface
        Configured interactive interface
    """
    interface = InteractiveInterface(frequencies, fields, s_parameters)
    interface.create_plot()
    interface.activate_interaction()
    
    return interface