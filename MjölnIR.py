# Copyright (c) [2025] [Mathesh Vaithiyanathan]
# This software is licensed under the MIT License.
# See the LICENSE file for details.

# Versions needed for the program (in case of changes in the future versions, please come back to these versions.
# Matplotlib Version: 3.10.3
# PyQt5 Version: 5.15.10
# Pandas Version: 2.2.3
# NumPy Version: 2.2.5
# PyQtGraph Version: 0.13.7
# Scipy Version: 1.15.3

import re
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QSlider,
    QLabel,
    QGridLayout,
    QPushButton,
    QMessageBox,
    QDialog,
    QLineEdit,
    QSplitter,
    QDialogButtonBox,
    QSpinBox,
    QTextEdit,
    QDoubleSpinBox,
    QMenu,
    QAction,
    QFileDialog,
    QComboBox,
    QTabWidget
)
import resources_rc
import scipy
from PyQt5.QtGui import QFont, QColor, QDoubleValidator, QIntValidator, QIcon
from scipy.interpolate import RectBivariateSpline, UnivariateSpline, interp1d # Added interp1d
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
import pandas as pd
import json
import os
import ctypes
from scipy.interpolate import RegularGridInterpolator
from PyQt5 import QtCore
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import pyqtgraph as pg
import matplotlib

print(f"Matplotlib Version: {matplotlib.__version__}")
print(f"PyQt5 Version: {QtCore.PYQT_VERSION_STR}")
print(f"Pandas Version: {pd.__version__}")
print(f"NumPy Version: {np.__version__}")
print(f"PyQtGraph Version: {pg.__version__}")
print(f"Scipy Version: {scipy.__version__}")

# ---------------------------------------------------------------------------------
def find(in_array, target_value):
    array = in_array
    nearest_index = np.abs(array - target_value).argmin( )
    nearest_value = array [ nearest_index ]
    # print(f"Nearest value to {target_value} is {nearest_value} at index {nearest_index}.")
    return nearest_index

def gaussian(x, amp, pos, fwhm):
    """
    Calculates the value of a Gaussian function.

    Args:
        x (np.array): The input x-values.
        amp (float): Amplitude of the Gaussian.
        pos (float): Position (mean) of the Gaussian.
        fwhm (float): Full Width at Half Maximum of the Gaussian.

    Returns:
        np.array: The y-values of the Gaussian.
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
    return amp * np.exp(-(x - pos) ** 2 / (2 * sigma ** 2))

def multi_gaussian(x, *params):
    """
    Calculates the sum of multiple Gaussian functions.
    params should be a flat list of [amp, pos, fwhm] for each Gaussian.
    """
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):  # 3 parameters per Gaussian
        amp, pos, fwhm = params [ i:i + 3 ]
        y += gaussian(x, amp, pos, fwhm)
    return y


def lorentzian(x, amplitude, mean, fwhm):
    """
    1D Lorentzian function for fitting.
    fwhm: Full Width at Half Maximum
    gamma: half-width at half-maximum (HWHM)
    """
    gamma = fwhm / 2.0
    return amplitude * (gamma ** 2 / ((x - mean) ** 2 + gamma ** 2))


def multi_lorentzian(x, *params):
    """
    Sum of multiple 1D Lorentzian functions.
    params should be a flat list of [amplitude, mean, fwhm] for each Lorentzian.
    """
    y_sum = np.zeros_like(x, dtype = float)
    for i in range(0, len(params), 3):  # 3 parameters per Lorentzian
        amplitude, mean, fwhm = params [ i:i + 3 ]
        y_sum += lorentzian(x, amplitude, mean, fwhm)
    return y_sum


class PeakFinderDialog(QDialog):
    """
    A dialog window for setting parameters for the 2D peak finding algorithm.
    """

    def __init__(self, parent=None, initial_params=None):
        super( ).__init__(parent)
        self.setWindowTitle("Find Peaks Parameters")
        self.setGeometry(200, 200, 400, 250)
        self.params = initial_params if initial_params else {
            'neighborhood_size': 4,
            'positive_threshold': 0.0001,
            'negative_threshold': -0.0001,
            'smooth_sigma': 0.0
        }

        self.init_ui( )

    def init_ui(self):
        layout = QVBoxLayout( )

        h_layout_n = QHBoxLayout( )
        h_layout_n.addWidget(QLabel("Neighborhood Size:"))
        self.neighborhood_size_input = QLineEdit(self)
        self.neighborhood_size_input.setText(str(self.params [ 'neighborhood_size' ]))
        self.neighborhood_size_input.setValidator(QIntValidator(1, 100))
        h_layout_n.addWidget(self.neighborhood_size_input)
        layout.addLayout(h_layout_n)

        h_layout_pt = QHBoxLayout( )
        h_layout_pt.addWidget(QLabel("Positive Threshold:"))
        self.positive_threshold_input = QLineEdit(self)
        self.positive_threshold_input.setText(str(self.params [ 'positive_threshold' ]))
        self.positive_threshold_input.setValidator(QDoubleValidator(-1.0, 1.0, 5))
        h_layout_pt.addWidget(self.positive_threshold_input)
        layout.addLayout(h_layout_pt)

        h_layout_nt = QHBoxLayout( )
        h_layout_nt.addWidget(QLabel("Negative Threshold:"))
        self.negative_threshold_input = QLineEdit(self)
        self.negative_threshold_input.setText(str(self.params [ 'negative_threshold' ]))
        self.negative_threshold_input.setValidator(QDoubleValidator(-1.0, 1.0, 5))
        h_layout_nt.addWidget(self.negative_threshold_input)
        layout.addLayout(h_layout_nt)

        h_layout_ss = QHBoxLayout( )
        h_layout_ss.addWidget(QLabel("Smooth Sigma:"))
        self.smooth_sigma_input = QLineEdit(self)
        self.smooth_sigma_input.setText(str(self.params [ 'smooth_sigma' ]))
        self.smooth_sigma_input.setValidator(QDoubleValidator(0.0, 10.0, 2))
        h_layout_ss.addWidget(self.smooth_sigma_input)
        layout.addLayout(h_layout_ss)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_parameters(self):
        """
        Retrieves the parameters entered.
        """
        try:
            self.params [ 'neighborhood_size' ] = int(self.neighborhood_size_input.text( ))
            self.params [ 'positive_threshold' ] = float(self.positive_threshold_input.text( ))
            self.params [ 'negative_threshold' ] = float(self.negative_threshold_input.text( ))
            self.params [ 'smooth_sigma' ] = float(self.smooth_sigma_input.text( ))
            return self.params
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numbers for all parameters.")
            return None


class ColorPickerDialog(QDialog):
    """
    A dialog window for selecting a color from a predefined list.
    """

    def __init__(self, parent=None, initial_color_name='Black'):
        super( ).__init__(parent)
        self.setWindowTitle("Select Color")
        self.setGeometry(300, 300, 250, 100)

        self.color_map = {
            'Black': 'k',
            'Red': 'r',
            'Blue': 'b',
            'Green': 'g',
            'Magenta': 'm',
            'Cyan': 'c',
            'Yellow': 'y',
            'White': 'w',
            'Orange': (255, 165, 0),
            'Purple': (128, 0, 128)
        }
        self.reverse_color_map = { v: k for k, v in self.color_map.items( ) }
        for name, rgb in self.color_map.items( ):
            if isinstance(rgb, tuple):
                self.reverse_color_map [ str(rgb) ] = name

        layout = QVBoxLayout( )

        h_layout = QHBoxLayout( )
        h_layout.addWidget(QLabel("Select Color:"))
        self.color_combo = QComboBox(self)
        self.color_combo.addItems(list(self.color_map.keys( )))

        initial_display_name = 'Black'
        if initial_color_name in self.reverse_color_map:
            initial_display_name = self.reverse_color_map [ initial_color_name ]
        elif isinstance(initial_color_name, tuple):
            for name, color_val in self.color_map.items( ):
                if color_val == initial_color_name:
                    initial_display_name = name
                    break

        idx = self.color_combo.findText(initial_display_name)
        if idx != -1:
            self.color_combo.setCurrentIndex(idx)

        h_layout.addWidget(self.color_combo)
        layout.addLayout(h_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_color_name(self):
        """Returns the pyqtgraph-compatible color representation (string or tuple)."""
        selected_text = self.color_combo.currentText( )
        return self.color_map [ selected_text ]

class LineThicknessDialog(QDialog):
    """
    A dialog window for setting the line thickness of plots.
    """

    def __init__(self, parent=None, initial_thickness=2):
        super( ).__init__(parent)
        self.setWindowTitle("Change Line Thickness")
        self.setGeometry(300, 300, 250, 100)

        layout = QVBoxLayout( )

        h_layout = QHBoxLayout( )
        h_layout.addWidget(QLabel("Line Thickness (px):"))
        self.thickness_input = QSpinBox(self)
        self.thickness_input.setRange(1, 10)
        self.thickness_input.setValue(initial_thickness)
        h_layout.addWidget(self.thickness_input)
        layout.addLayout(h_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_thickness(self):
        return self.thickness_input.value( )

class PlotParametersDialog(QDialog):
    def __init__(self, parent=None):
        super( ).__init__(parent)
        self.setWindowTitle("Plot Parameters")
        self.setGeometry(300, 300, 400, 300)
        self.setModal(True)

        self.result_params = None
        self.init_ui( )

    def init_ui(self):
        layout = QVBoxLayout( )

        self.start_wn_input = self._create_input_row(layout, "Start:", "1900")
        self.stop_wn_input = self._create_input_row(layout, "End:", "2100")
        self.shift_index_input = self._create_input_row(layout, "Shift Index (pixels):", "0", is_int = True)

        self.npoints_input = self._create_input_row(layout, "NPoints (Interpolation):", "500", is_int = True)
        self.interp_method_combo = self._create_combo_row(layout, "Interpolation Method:",
                                                          [ "linear", "cubic", "nearest" ])

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept_parameters)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def _create_input_row(self, layout, label_text, default_value, is_int=False):
        h_layout = QHBoxLayout( )
        label = QLabel(label_text)
        input_field = QLineEdit(self)
        input_field.setText(default_value)
        if is_int:
            input_field.setValidator(QIntValidator( ))
        else:
            input_field.setValidator(QDoubleValidator( ))
        h_layout.addWidget(label)
        h_layout.addWidget(input_field)
        layout.addLayout(h_layout)
        return input_field

    def _create_combo_row(self, layout, label_text, options):
        h_layout = QHBoxLayout( )
        label = QLabel(label_text)
        combo_box = QComboBox(self)
        combo_box.addItems(options)
        h_layout.addWidget(label)
        h_layout.addWidget(combo_box)
        layout.addLayout(h_layout)
        return combo_box

    def accept_parameters(self):
        try:
            start_wn = float(self.start_wn_input.text( ))
            stop_wn = float(self.stop_wn_input.text( ))
            shift_index = int(self.shift_index_input.text( ))
            npoints = int(self.npoints_input.text( ))
            interp_method = self.interp_method_combo.currentText( )

            self.result_params = {
                'start_wn': start_wn,
                'stop_wn': stop_wn,
                'shift_index': shift_index,
                'npoints': npoints,
                'interp_method': interp_method,
            }
            self.accept( )
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numerical values for all fields.")

class EditNamesDialog(QDialog):
    """
    A dialog window for editing the axis labels of the plots.
    """

    def __init__(self, parent=None, current_labels=None):
        super( ).__init__(parent)
        self.setWindowTitle("Edit Axis Names")
        self.setGeometry(200, 200, 1000, 550)

        self.labels = current_labels if current_labels else {
            'signal_bottom': 'Probe wavenumber [cm\u207B\u00B9]',
            'signal_left': 'Pump wavenumber [cm\u207B\u00B9]',
            'x_slice_bottom': 'Pump wavenumber [cm\u207B\u00B9]',
            'x_slice_left': 'ΔOD',
            'y_slice_bottom': 'Probe wavenumber [cm\u207B\u00B9]',
            'y_slice_left': 'ΔOD'
        }
        self.init_ui( )

    def init_ui(self):
        layout = QVBoxLayout( )
        form_layout = QGridLayout( )

        dialog_font = QFont("Times New Roman")
        dialog_font.setPointSize(12)

        label_signal_bottom = QLabel("2D Plot - X-axis:")
        label_signal_bottom.setFont(dialog_font)
        form_layout.addWidget(label_signal_bottom, 0, 0)
        self.signal_bottom_input = QLineEdit(self)
        self.signal_bottom_input.setText(self.labels [ 'signal_bottom' ])
        self.signal_bottom_input.setFont(dialog_font)
        form_layout.addWidget(self.signal_bottom_input, 0, 1)

        label_signal_left = QLabel("2D Plot - Y-axis:")
        label_signal_left.setFont(dialog_font)
        form_layout.addWidget(label_signal_left, 1, 0)
        self.signal_left_input = QLineEdit(self)
        self.signal_left_input.setText(self.labels [ 'signal_left' ])
        self.signal_left_input.setFont(dialog_font)
        form_layout.addWidget(self.signal_left_input, 1, 1)

        label_x_slice_bottom = QLabel("X-Slice Plot - X-axis:")
        label_x_slice_bottom.setFont(dialog_font)
        form_layout.addWidget(label_x_slice_bottom, 2, 0)
        self.x_slice_bottom_input = QLineEdit(self)
        self.x_slice_bottom_input.setText(self.labels [ 'x_slice_bottom' ])
        self.x_slice_bottom_input.setFont(dialog_font)
        form_layout.addWidget(self.x_slice_bottom_input, 2, 1)

        label_x_slice_left = QLabel("X-Slice Plot - Y-axis:")
        label_x_slice_left.setFont(dialog_font)
        form_layout.addWidget(label_x_slice_left, 3, 0)
        self.x_slice_left_input = QLineEdit(self)
        self.x_slice_left_input.setText(self.labels [ 'x_slice_left' ])
        self.x_slice_left_input.setFont(dialog_font)
        form_layout.addWidget(self.x_slice_left_input, 3, 1)

        label_y_slice_bottom = QLabel("Y-Slice Plot - X-axis:")
        label_y_slice_bottom.setFont(dialog_font)
        form_layout.addWidget(label_y_slice_bottom, 4, 0)
        self.y_slice_bottom_input = QLineEdit(self)
        self.y_slice_bottom_input.setText(self.labels [ 'y_slice_bottom' ])
        self.y_slice_bottom_input.setFont(dialog_font)
        form_layout.addWidget(self.y_slice_bottom_input, 4, 1)

        label_y_slice_left = QLabel("Y-Slice Plot - Y-axis:")
        label_y_slice_left.setFont(dialog_font)
        form_layout.addWidget(label_y_slice_left, 5, 0)
        self.y_slice_left_input = QLineEdit(self)
        self.y_slice_left_input.setText(self.labels [ 'y_slice_left' ])
        self.y_slice_left_input.setFont(dialog_font)
        form_layout.addWidget(self.y_slice_left_input, 5, 1)

        layout.addLayout(form_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_names(self):
        """
        Retrieves the updated axis names from the input fields.
        """
        return {
            'signal_bottom': self.signal_bottom_input.text( ),
            'signal_left': self.signal_left_input.text( ),
            'x_slice_bottom': self.x_slice_bottom_input.text( ),
            'x_slice_left': self.x_slice_left_input.text( ),
            'y_slice_bottom': self.y_slice_bottom_input.text( ),
            'y_slice_left': self.y_slice_left_input.text( )
        }

#----------------------------------------------------------------------------------------------------

class GaussianFitterApp(QWidget):
    """
    PyQt Application for interactive Gaussian or Lorentzian fitting.
    Allows users to visually select initial peak guesses and fit them to data
    based on the chosen function type.
    """

    def __init__(self, parent=None, x_data=None, y_data=None, fitting_function_type="Gaussian", xlabel="X-axis",
                 ylabel="Y-axis",
                 slice_axis_name="", slice_value=None, slice_unit="", is_spline_corrected=False):
        """
        Initializes the FitterApp.

        Args:
            parent (QWidget): The parent widget for this dialog.
            x_data (np.array): The x-values of the input data.
            y_data (np.array): The y-values of the input data.
            fitting_function_type (str): "Gaussian" or "Lorentzian".
            xlabel (str): The label for the x-axis of the fitter plot.
            ylabel (str): The label for the y-axis of the fitter plot.
            slice_axis_name (str): The name of the axis that was sliced (e.g., "Probe wavenumber").
            slice_value (float): The value at which the slice was taken.
            slice_unit (str): The unit of the slice value.
            is_spline_corrected (bool): True if the data is spline corrected, False otherwise.
        """
        super( ).__init__(parent)
        self.xga = x_data if x_data is not None else np.array([ ])
        self.yga = y_data if y_data is not None else np.array([ ])

        # Store original data for interpolation purposes
        self._original_x_data = np.copy(x_data) if x_data is not None else np.array([])
        self._original_y_data = np.copy(y_data) if y_data is not None else np.array([])

        self.fitting_function_type = fitting_function_type.capitalize( )
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.slice_axis_name = slice_axis_name
        self.slice_value = slice_value
        self.slice_unit = slice_unit
        self.is_spline_corrected = is_spline_corrected  # Store the flag

        # State variable to control if clicks are for guessing or normal plot interaction
        self.is_guessing_mode_active = False

        title_parts = [ f"{self.fitting_function_type}:" ]
        if self.slice_axis_name and self.slice_value is not None:
            title_parts.append(f"{self.slice_axis_name} = {self.slice_value:.1f}{self.slice_unit}")

        # Add spline correction status to the title
        if self.is_spline_corrected:
            title_parts.append("(Spline Corrected)")
        else:
            title_parts.append("(Original Data)")

        self.setWindowTitle(" ".join(title_parts))
        self.setObjectName(" ".join(title_parts))
        # self.setGeometry(100, 100, 1500, 1000)

        self.init_ui( )
        self.init_fitter_variables( )
        self.update_plot( )

    def init_ui(self):
        """
        Sets up the main user interface components with resizable splitter.
        """
        # Main layout for the QWidget
        self.main_layout = QVBoxLayout(self)


        # 1. Create Matplotlib Figure and Canvas FIRST
        self.fig, self.ax = plt.subplots(figsize = (15, 10))
        self.canvas = FigureCanvas(self.fig)

        # 2. Create Navigation Toolbar (requires canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # 3. Create Text Display Widget
        self.params_text_edit = QTextEdit( )
        self.params_text_edit.setReadOnly(True)
        self.params_text_edit.setMinimumHeight(100)
        self.params_text_edit.setStyleSheet("font-family: Consolas; font-size: 10pt;")

        # 4. Create Control Buttons
        self.start_guess_button = QPushButton("Start Initial Guess")
        self.fit_button = QPushButton(f"Fit {self.fitting_function_type}")
        self.clear_button = QPushButton("Clear Guesses")
        self.close_button = QPushButton("Close Tab")

        # 5. Create Info Label
        self.info_label = QLabel(
            f"Use the toolbar for zoom/pan. Click 'Start Initial Guess' to define {self.fitting_function_type} parameters.")
        self.info_label.setWordWrap(True)

        # 6. Create Splitter for resizable plot/text areas
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(self.canvas)  # Top: Plot
        self.splitter.addWidget(self.params_text_edit)  # Bottom: Text results
        self.splitter.setSizes([ 700, 300 ])  # Initial size ratio (70/30)

        # 7. Create Control Panel Layout
        self.control_layout = QHBoxLayout( )
        self.control_layout.addWidget(self.start_guess_button)
        self.control_layout.addWidget(self.fit_button)
        self.control_layout.addWidget(self.clear_button)
        self.control_layout.addWidget(self.info_label)
        self.control_layout.addWidget(self.close_button)  # Add Close button to layout
        self.control_layout.setSpacing(10)

        # 8. Assemble Main Layout
        self.main_layout.addWidget(self.toolbar)  # Matplotlib toolbar
        self.main_layout.addLayout(self.control_layout)  # Button controls
        self.main_layout.addWidget(self.splitter)  # Resizable plot/text area

        # 9. Connect signals
        self.start_guess_button.clicked.connect(self._toggle_guessing_mode)
        self.fit_button.clicked.connect(self.on_fit)
        self.clear_button.clicked.connect(self.on_clear_guesses)
        self.close_button.clicked.connect(self._close_tab)  # Connect Close button to custom close method

        # 10. Connect Matplotlib events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def init_fitter_variables(self):
        """
        Initializes variables used for interactive fitting.
        """
        self.amp = None
        self.pos = None
        self.fwhm = None
        self.temp_line = None
        self.fixed_peaks = [ ]  # Stores (amp, pos, fwhm) for each peak
        self.fitted_params = None
        self.fitted_errors = None

        # Store original limits for resetting view
        self.original_xlim = self.ax.get_xlim( )
        self.original_ylim = self.ax.get_ylim( )

    def _get_single_function(self):
        """Returns the single peak function based on the chosen type."""
        if self.fitting_function_type == "Gaussian":
            return gaussian
        elif self.fitting_function_type == "Lorentzian":
            return lorentzian
        else:
            raise ValueError("Invalid fitting function type selected.")

    def _get_multi_function(self):
        """Returns the multi-peak function based on the chosen type."""
        if self.fitting_function_type == "Gaussian":
            return multi_gaussian
        elif self.fitting_function_type == "Lorentzian":
            return multi_lorentzian
        else:
            raise ValueError("Invalid fitting function type selected.")

    def _toggle_guessing_mode(self):
        """
        Toggles the interactive guessing mode for placing peaks.
        """
        self.is_guessing_mode_active = not self.is_guessing_mode_active

        if self.is_guessing_mode_active:
            # Reset current temporary guess
            self.amp = None
            self.pos = None
            self.fwhm = None
            self.temp_line = None
            self.start_guess_button.setText("Stop Initial Guess")
            self.info_label.setText("Guessing Mode: ON. Click on the plot for peak position and amplitude.")
            self.update_plot( )
        else:
            self.start_guess_button.setText("Start Initial Guess")
            self.info_label.setText(
                "Guessing Mode: OFF. Use the toolbar for zoom/pan. Click 'Start Initial Guess' to define parameters.")
            # Clear any active temporary guess visualization
            self.amp = None
            self.pos = None
            self.fwhm = None
            self.temp_line = None
            self.update_plot( )  # Redraw to remove temporary line

    def on_click(self, event):
        """
        Handles mouse button press events on the Matplotlib canvas.
        Used to define initial peak parameters when guessing mode is active.
        """
        if event.inaxes != self.ax:  # Ensure click is within the plot area
            return

        if not self.is_guessing_mode_active:
            # If guessing mode is off, allow default toolbar interaction (zoom/pan)
            return

        if event.button == 1:  # Left click
            if self.amp is None:  # First click to set amplitude and position
                self.amp = event.ydata
                self.pos = event.xdata
                # Initial FWHM is a tenth of the x-range, or a small value if range is zero
                self.fwhm = (self.xga [ -1 ] - self.xga [ 0 ]) / 10 if (self.xga [ -1 ] - self.xga [ 0 ]) != 0 else 1.0
                self.start_x = event.xdata  # Store starting x-position for FWHM calculation
                self.info_label.setText("Drag mouse to adjust FWHM, then click again to fix.")
            else:  # Second click to fix the peak guess
                self.fixed_peaks.append((self.amp, self.pos, self.fwhm))  # Store 3 parameters
                self.amp = None
                self.pos = None
                self.fwhm = None
                self.temp_line = None
                self.info_label.setText(
                    f"{self.fitting_function_type} {len(self.fixed_peaks)} fixed. Click for next, or 'Stop Initial Guess'.")
            self.update_plot( )
        elif event.button == 3:  # Right click to cancel current selection
            if self.amp is not None:
                self.amp = None
                self.pos = None
                self.fwhm = None
                self.temp_line = None
                self.info_label.setText("Current peak selection cancelled. Click to start new guess.")
                self.update_plot( )

    def on_motion(self, event):
        """
        Handles mouse motion events on the Matplotlib canvas.
        Used to dynamically adjust the FWHM of the temporary peak when guessing mode is active.
        """
        if event.inaxes != self.ax or self.amp is None or not self.is_guessing_mode_active:
            return

        # Calculate FWHM based on distance from start_x
        self.fwhm = 2 * abs(event.xdata - self.start_x)
        if self.fwhm < 0.001:  # Prevent FWHM from becoming zero or too small
            self.fwhm = 0.001
        self.update_plot( )

    def apply_interpolation_settings(self, method, multiplier):
        """
        Applies interpolation to the current slice data and updates the plot.
        Resets fit/guess points as data changes.
        """
        # Clear existing guesses/fits as they are based on old data points
        self.on_clear_guesses()

        if method == "None":
            self.xga = np.copy(self._original_x_data)
            self.yga = np.copy(self._original_y_data)
            print(f"GaussianFitterApp: Reverted to original data (len={len(self.xga)})")
        else:
            try:
                target_n_points = int(len(self._original_x_data) * multiplier)
                if target_n_points < 2: # Ensure at least 2 points for interpolation
                    target_n_points = 2

                # Create a new, denser x-axis for interpolation
                x_interp = np.linspace(self._original_x_data.min(), self._original_x_data.max(), target_n_points)

                # Create the interpolation function
                # Use fill_value="extrapolate" to handle cases where x_interp might slightly exceed original bounds
                f_interp = interp1d(self._original_x_data, self._original_y_data, kind=method, fill_value="extrapolate")

                # Interpolate y values
                y_interp = f_interp(x_interp)

                self.xga = x_interp
                self.yga = y_interp
                print(f"GaussianFitterApp: Applied {method} interpolation with x{multiplier} (new len={len(self.xga)})")

            except ValueError as e:
                QMessageBox.warning(self, "Interpolation Error",
                                    f"Could not apply {method} interpolation: {e}")
                # Fallback to original data on error
                self.xga = np.copy(self._original_x_data)
                self.yga = np.copy(self._original_y_data)
                print(f"GaussianFitterApp: Error applying interpolation, reverted to original data.")
            except Exception as e:
                 QMessageBox.warning(self, "Interpolation Error",
                                    f"An unexpected error occurred during interpolation: {e}")
                 # Fallback to original data on error
                 self.xga = np.copy(self._original_x_data)
                 self.yga = np.copy(self._original_y_data)
                 print(f"GaussianFitterApp: Unexpected error, reverted to original data.")

        self.update_plot() # Redraw the plot with new data


    def on_fit(self):
        """
        Performs the fitting using scipy.optimize.curve_fit.
        Triggered by the "Fit" button.
        """
        if not self.fixed_peaks:
            QMessageBox.warning(self, "No Peaks",
                                f"Please fix at least one {self.fitting_function_type} guess before fitting.")
            return

        # If guessing mode is active, turn it off before fitting
        if self.is_guessing_mode_active:
            self.is_guessing_mode_active = False
            self.start_guess_button.setText("Start Initial Guess")
            self.info_label.setText("Guessing Mode: OFF. Fitting in progress...")

        params = np.array(self.fixed_peaks).flatten( )

        # Define bounds for parameters: [amp, pos, fwhm]
        # amp: can be negative or positive
        # pos: within x_data range
        # fwhm: must be positive (e.g., > 0.001)
        lower_bounds = [ ]
        upper_bounds = [ ]
        for _ in range(len(self.fixed_peaks)):
            lower_bounds.extend([ -np.inf, self.xga.min( ), 0.001 ])
            upper_bounds.extend([ np.inf, self.xga.max( ), np.inf ])

        try:
            self.fitted_params, pcov = curve_fit(self._get_multi_function( ), self.xga, self.yga, p0 = params,
                                                 bounds = (lower_bounds, upper_bounds))
            self.fitted_errors = np.sqrt(np.diag(pcov))
            self.update_plot( )
            self.display_fitted_parameters( )
            self.info_label.setText("Fitting complete. See fitted parameters below.")
        except RuntimeError as e:
            QMessageBox.critical(self, "Fitting Error",
                                 f"Failed to fit {self.fitting_function_type}: {e}. Adjust peaks and try again.")
            self.info_label.setText("Fitting failed. Adjust guesses and try again.")
        except ValueError as e:
            QMessageBox.critical(self, "Fitting Error",
                                 f"Fitting input error: {e}. Check data and initial parameters.")
            self.info_label.setText("Fitting failed due to input error. Check data.")

    def on_clear_guesses(self):
        """
        Clears all fixed peak guesses and resets the fitter state.
        """
        self.fixed_peaks = [ ]
        self.amp = None
        self.pos = None
        self.fwhm = None
        self.temp_line = None
        self.fitted_params = None
        self.fitted_errors = None
        self.params_text_edit.clear( )
        self.info_label.setText("All peak guesses cleared. Click 'Start Initial Guess' to define new ones.")
        self.update_plot( )

    def display_fitted_parameters(self):
        """
        Formats and displays the fitted peak parameters and their errors
        in the QTextEdit widget.
        """
        if self.fitted_params is None:
            self.params_text_edit.clear( )
            return

        output_text = f"Fitted {self.fitting_function_type} Parameters:\n"
        output_text += "--------------------------------------------------\n"
        for i in range(0, len(self.fitted_params), 3):  # Iterate by 3 parameters
            amp, pos, fwhm = self.fitted_params [ i:i + 3 ]
            amp_err, pos_err, fwhm_err = self.fitted_errors [ i:i + 3 ]

            output_text += (f"Peak {i // 3 + 1}:\n"
                            f"  Amplitude (Amp): {amp:.4g} ± {amp_err:.2g}\n"
                            f"  Position (Pos):  {pos:.4g} ± {pos_err:.2g}\n"
                            f"  FWHM:            {fwhm:.4g} ± {fwhm_err:.2g}\n")
            output_text += "--------------------------------------------------\n"
        self.params_text_edit.setText(output_text)

    def update_plot(self):
        """
        Clears the plot and redraws all elements:
        - Original data
        - Fixed peak guesses (if not fitted)
        - Temporary adjusting peak (if active)
        - Fitted curve and individual fitted components (if fitted)
        """
        self.ax.clear( )
        self.ax.plot(self.xga, self.yga, 'b-', label = 'Data')
        legend_entries = [ 'Data' ]

        if self.fitted_params is not None:
            # Plot fitted curve
            self.ax.plot(self.xga, self._get_multi_function( )(self.xga, *self.fitted_params),
                         'g-', label = 'Fitted Curve', linewidth = 2)
            legend_entries.append('Fitted Curve')

            # Plot components with formatted legend
            for i in range(0, len(self.fitted_params), 3):  # Iterate by 3 parameters
                amp, pos, fwhm = self.fitted_params [ i:i + 3 ]
                # Ensure errors are within bounds of fitted_errors array
                amp_err, pos_err, fwhm_err = (
                    self.fitted_errors [ i ], self.fitted_errors [ i + 1 ], self.fitted_errors [ i + 2 ]
                ) if i + 2 < len(self.fitted_errors) else (0, 0, 0)  # Fallback for safety

                label = (f'{self.fitting_function_type [ :5 ]} {i // 3 + 1}')  # Use 'Ga' or 'Lo' prefix
                # f'Amp= {amp:.3g} ± {amp_err:.2g}\n'
                # f'Pos = {pos:.1f} ± {pos_err:.2g}\n'
                # f'FWHM = {fwhm:.3g} ± {fwhm_err:.2g}')

                self.ax.plot(self.xga, self._get_single_function( )(self.xga, amp, pos, fwhm),
                             '--', alpha = 0.7, linewidth = 1.5, label = label)
                # legend_entries.append(label) # Add individual labels only if desired in legend

        else:
            # Plot initial guesses
            for i, (amp, pos, fwhm) in enumerate(self.fixed_peaks):  # Unpack 3 values
                label = f'Initial Guess {i + 1}' if i == 0 else ""
                self.ax.plot(self.xga, self._get_single_function( )(self.xga, amp, pos, fwhm),
                             'r--', alpha = 0.5, label = label)
                if i == 0:
                    legend_entries.append('Initial Guesses')

        # Temporary adjusting peak
        if self.amp is not None:
            # Ensure fwhm is not zero to avoid division by zero
            if self.fwhm == 0: self.fwhm = 0.001
            self.temp_line, = self.ax.plot(self.xga,
                                           self._get_single_function( )(self.xga, self.amp, self.pos, self.fwhm),
                                           'g--', alpha = 0.5, label = 'Adjusting Width')
            if 'Adjusting Width' not in legend_entries:  # Add only once
                legend_entries.append('Adjusting Width')

        # Use the passed xlabel and ylabel
        self.ax.set_ylabel(self.ylabel, fontsize = 14)
        self.ax.set_xlabel(self.xlabel, fontsize = 14)
        self.ax.grid(True)
        self.ax.legend(loc = 'best', fontsize = 14)
        self.ax.tick_params(axis = 'both', which = 'major', labelsize = 12)

        # Restore original view if not fitted yet
        if self.fitted_params is None:
            pass

        self.canvas.draw( )

    def get_fitter_state(self):
        """
        Returns the current state of the fitter for saving.
        """
        return {
            'x_data': self.xga.tolist( ),
            'y_data': self.yga.tolist( ),
            'original_x_data': self._original_x_data.tolist(), # Save original data
            'original_y_data': self._original_y_data.tolist(), # Save original data
            'fitting_function_type': self.fitting_function_type,
            'xlabel': self.xlabel,
            'ylabel': self.ylabel,
            'slice_axis_name': self.slice_axis_name,
            'slice_value': self.slice_value,
            'slice_unit': self.slice_unit,
            'fixed_peaks': self.fixed_peaks,
            'fitted_params': self.fitted_params.tolist( ) if self.fitted_params is not None else None,
            'fitted_errors': self.fitted_errors.tolist( ) if self.fitted_errors is not None else None,
            'tab_title': self.objectName( ),  # Save the tab title
            'is_spline_corrected': self.is_spline_corrected  # Save the spline status
        }

    def set_fitter_state(self, state):
        """
        Sets the state of the fitter from loaded data.
        """
        self.xga = np.array(state [ 'x_data' ])
        self.yga = np.array(state [ 'y_data' ])
        self._original_x_data = np.array(state.get('original_x_data', self.xga)) # Load original data, fallback to current
        self._original_y_data = np.array(state.get('original_y_data', self.yga)) # Load original data, fallback to current
        self.fitting_function_type = state [ 'fitting_function_type' ]
        self.xlabel = state [ 'xlabel' ]
        self.ylabel = state [ 'ylabel' ]
        self.slice_axis_name = state [ 'slice_axis_name' ]
        self.slice_value = state [ 'slice_value' ]
        self.slice_unit = state [ 'slice_unit' ]
        self.fixed_peaks = state [ 'fixed_peaks' ]
        self.fitted_params = np.array(state [ 'fitted_params' ]) if state [ 'fitted_params' ] is not None else None
        self.fitted_errors = np.array(state [ 'fitted_errors' ]) if state [ 'fitted_errors' ] is not None else None
        self.is_spline_corrected = state.get('is_spline_corrected', False)  # Load the spline status

        # Update the object name (which becomes the tab title)
        title_parts = [ f"{self.fitting_function_type}:" ]
        if self.slice_axis_name and self.slice_value is not None:
            title_parts.append(f"{self.slice_axis_name} = {self.slice_value:.1f}{self.slice_unit}")

        if self.is_spline_corrected:
            title_parts.append("(Spline Corrected)")
        else:
            title_parts.append("(Original Data)")

        self.setObjectName(" ".join(title_parts))
        self.setWindowTitle(" ".join(title_parts))

        self.update_plot( )
        self.display_fitted_parameters( )

    def _close_tab(self):
        """
        Handles closing the tab by explicitly removing it from the parent QTabWidget.
        """
        # The parent of GaussianFitterApp is the QTabWidget
        parent_tab_widget = self.parent( )
        if parent_tab_widget and isinstance(parent_tab_widget, QTabWidget):
            tab_index = parent_tab_widget.indexOf(self)
            if tab_index != -1:
                parent_tab_widget.removeTab(tab_index)
        self.deleteLater( )  # Ensure the widget is properly deleted


def signal_fitter_wrapper(parent, plot_data_item, is_x_slice, fitting_function_type, xlabel, ylabel, slice_axis_name,
                          slice_value, slice_unit, is_spline_corrected):
    """
    Wrapper function to extract data from pyqtgraph plot and launch
    the Interactive Fitter (GaussianFitterApp).
    plot_data_item: PlotDataItem holding x and y data.
    is_x_slice (bool): True if data is from X-slice, False for Y-slice.
    fitting_function_type (str): "Gaussian" or "Lorentzian".
    xlabel (str): The x-axis label for the fitter plot.
    ylabel (str): The y-axis label for the fitter plot.
    slice_axis_name (str): The name of the axis that was sliced (e.g., "Probe wavenumber").
    slice_value (float): The value at which the slice was taken.
    slice_unit (str): The unit of the slice value.
    is_spline_corrected (bool): True if the data is spline corrected, False otherwise.
    """
    x_data_full = plot_data_item.getData( ) [ 0 ]
    y_data_full = plot_data_item.getData( ) [ 1 ]

    # Get the current view range from the plot_data_item's ViewBox
    view_range = plot_data_item.getViewBox( ).viewRange( )
    xlim_view = view_range [ 0 ]
    ylim_view = view_range [ 1 ]

    # Filter data to only include points within the visible x-range
    mask = np.logical_and(x_data_full >= xlim_view [ 0 ], x_data_full <= xlim_view [ 1 ])
    x_data_filtered = x_data_full [ mask ]
    y_data_filtered = y_data_full [ mask ]

    if len(x_data_filtered) < 3:  # Need at least 3 points for Gaussian/Lorentzian fit (amp, pos, fwhm)
        QMessageBox.warning(None, "Fit Error",
                            "Not enough data points in visible range for fitting (need at least 3). Zoom in or adjust data.")
        return None  # Return None if no window is created

    # Pass parent to GaussianFitterApp
    fitter_app = GaussianFitterApp(parent, x_data_filtered, y_data_filtered, fitting_function_type, xlabel, ylabel,
                                   slice_axis_name, slice_value, slice_unit, is_spline_corrected)
    return fitter_app  # Return the instance so the caller can keep a reference


class Plotting_contour_and_interpoaltion:
    def __init__(self, probe_wn, pump_wn, data):
        self.pump = pump_wn
        self.probe = probe_wn
        self.data = data
        self.diagonal_x_line = None
        self.diagonal = None

    def choose_region(self, start, stop, shiftindex_left=0):
        if self.probe is None or self.pump is None or self.data is None:
            raise ValueError("Data not initialized. Cannot choose region.")

        probe_start_idx = find(self.probe, start) + shiftindex_left
        probe_stop_idx = find(self.probe, stop) + shiftindex_left
        pump_start_idx = find(self.pump, start)
        pump_stop_idx = find(self.pump, stop)

        probe_start_idx = max(0, probe_start_idx)
        probe_stop_idx = min(len(self.probe), probe_stop_idx)
        pump_start_idx = max(0, pump_start_idx)
        pump_stop_idx = min(len(self.pump), pump_stop_idx)

        self.probe = self.probe [ probe_start_idx:probe_stop_idx ]
        self.pump = self.pump [ pump_start_idx:pump_stop_idx ]
        self.data = self.data [ pump_start_idx:pump_stop_idx, probe_start_idx:probe_stop_idx ]

        return self.probe, self.pump, self.data

    def interpolate(self, npoints, method='cubic'):
        """
        To take a diagonal from the data, we interpoalte the data to make a square.
        """
        if self.probe is None or self.pump is None or self.data is None:
            raise ValueError("Data not initialized. Cannot interpolate.")
        if not self.probe.size > 1 or not self.pump.size > 1:
            raise ValueError(
                "Not enough data points for interpolation. Ensure probe and pump arrays have at least 2 points.")

        interpolator = RegularGridInterpolator((self.pump, self.probe), self.data, method = method)

        new_probe = np.linspace(np.min(self.probe), np.max(self.probe), npoints)
        new_pump = np.linspace(np.min(self.pump), np.max(self.pump), npoints)

        new_pump_mesh, new_probe_mesh = np.meshgrid(new_pump, new_probe, indexing = 'ij')
        points_to_interpolate = np.array([ new_pump_mesh.ravel( ), new_probe_mesh.ravel( ) ]).T

        interpolated_data_flat = interpolator(points_to_interpolate)
        self.data = interpolated_data_flat.reshape(npoints, npoints)

        self.probe = new_probe
        self.pump = new_pump

        return self.probe, self.pump, self.data

    def plot_diagonal_and_contour(self, contour_widget, diagonal_widget, image_item, diagonal_slice_curve,
                                  dashed_diagonal_line,
                                  initial_min_level=None, initial_max_level=None,
                                  initial_diagonal_slice_linewidth=2, initial_dashed_diagonal_line_linewidth=2,
                                  initial_dashed_diagonal_line_color='k',
                                  initial_contour_xlabel='Probe wavenumber  [cm<sup>-1</sup>]',
                                  initial_contour_ylabel='Pump wavenumber [cm<sup>-1</sup>]',
                                  initial_diagonal_xlabel='Probe wavenumber [cm<sup>-1</sup>]',
                                  initial_diagonal_ylabel='ΔOD'):
        # Clear existing content
        contour_widget.clear( )
        diagonal_widget.clear( )

        # Disable auto-range before plotting (this is also done in __init__ for robustness)
        contour_widget.enableAutoRange(False)
        diagonal_widget.enableAutoRange(False)

        # Calculate diagonal
        # Ensure probe and data have compatible dimensions for diagonal slice
        if self.probe.shape [ 0 ] != self.data.shape [ 0 ] or self.probe.shape [ 0 ] != self.data.shape [ 1 ]:
            min_dim = min(self.data.shape)
            self.diagonal = np.diagonal(self.data [ :min_dim, :min_dim ])
            self.diagonal_x_line = self.probe [ :min_dim ]
        else:
            self.diagonal = np.diagonal(self.data)
            self.diagonal_x_line = self.probe

        # Contour Plot
        # Set image data and its bounding rectangle
        image_item.setImage(self.data.T)
        image_item.setRect(pg.QtCore.QRectF(self.probe [ 0 ], self.pump [ 0 ],
                                            self.probe [ -1 ] - self.probe [ 0 ],
                                            self.pump [ -1 ] - self.pump [ 0 ]))
        contour_widget.addItem(image_item)

        # Plot the diagonal dashed line on the contour plot
        dashed_diagonal_line.setData(self.probe, self.pump)
        dashed_diagonal_line.setPen(
            pg.mkPen(initial_dashed_diagonal_line_color, width = initial_dashed_diagonal_line_linewidth,
                     style = QtCore.Qt.DashLine))
        contour_widget.addItem(dashed_diagonal_line)

        # Create colormap from Matplotlib's 'seismic' change if you want anything else in here.
        cmap = plt.colormaps [ 'seismic' ]
        colors = cmap(np.linspace(0, 1, 256))
        colors_pyqtgraph = [ (int(c [ 0 ] * 255), int(c [ 1 ] * 255), int(c [ 2 ] * 255)) for c in colors ]
        pos = np.linspace(0., 1., len(colors_pyqtgraph))
        pg_cmap = pg.ColorMap(pos, colors_pyqtgraph)

        # Apply colormap to the image item
        image_item.setLookupTable(pg_cmap.getLookupTable( ))

        # Set initial levels based on provided arguments or data min/max
        if initial_min_level is not None and initial_max_level is not None:
            image_item.setLevels([ initial_min_level, initial_max_level ])
        else:
            image_item.setLevels([ np.min(self.data), np.max(self.data) ])

        # Set plot properties for the contour plot
        contour_widget.setLabel('left', initial_contour_ylabel)
        contour_widget.setLabel('bottom', initial_contour_xlabel)
        contour_widget.showGrid(x = True, y = True, alpha = 0.3)

        # Manually set the initial view range for the contour plot with padding=0
        contour_widget.setXRange(self.probe [ 0 ], self.probe [ -1 ], padding = 0)
        contour_widget.setYRange(self.pump [ 0 ], self.pump [ -1 ], padding = 0)

        # Diagonal Plot
        diagonal_slice_curve.setData(self.diagonal_x_line, -self.diagonal)
        diagonal_slice_curve.setPen(pg.mkPen('b', width = initial_diagonal_slice_linewidth))
        diagonal_widget.addItem(diagonal_slice_curve)

        diagonal_widget.setLabel('bottom', initial_diagonal_xlabel)
        diagonal_widget.setLabel('left', initial_diagonal_ylabel)
        diagonal_widget.addLegend( )
        diagonal_widget.showGrid(x = True, y = True, alpha = 0.3)

        # Manually set the initial view range for the diagonal plot
        diagonal_widget.setXRange(self.probe [ 0 ], self.probe [ -1 ])
        y_min, y_max = np.min(-self.diagonal), np.max(-self.diagonal)
        y_padding = (y_max - y_min) * 0.1
        diagonal_widget.setYRange(y_min - y_padding, y_max + y_padding)

        # Link x-axes of the two plots for synchronized zooming/panning
        contour_widget.setXLink(diagonal_widget)


class SetTextDialog(QDialog):
    """
    A generic dialog for setting text (e.g., plot title or axis label).
    """

    def __init__(self, parent=None, title="Set Text", initial_text=""):
        super( ).__init__(parent)
        self.setWindowTitle(title)
        self.setGeometry(300, 300, 350, 100)

        layout = QVBoxLayout( )

        h_layout = QHBoxLayout( )
        self.text_input = QLineEdit(self)
        self.text_input.setText(initial_text)
        h_layout.addWidget(QLabel("New Text:"))
        h_layout.addWidget(self.text_input)
        layout.addLayout(h_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_text(self):
        return self.text_input.text( )


class PlotWindow(QWidget):
    def __init__(self, processed_probe, processed_pump, processed_data,
                 initial_min_level=None, initial_max_level=None,
                 initial_diagonal_slice_linewidth=2, initial_dashed_diagonal_line_linewidth=2,
                 initial_dashed_diagonal_line_color='k',
                 initial_contour_title='2D IR Spectrum', initial_diagonal_title='Diagonal slice',
                 initial_contour_xlabel='Probe wavenumber [cm<sup>-1</sup>]',
                 initial_contour_ylabel='Pump wavenumber [cm<sup>-1</sup>]',
                 initial_diagonal_xlabel='Probe wavenumber [cm<sup>-1</sup>]',
                 initial_diagonal_ylabel='ΔOD', parent=None, is_spline_corrected=False):
        super( ).__init__(parent)

        # Determine title based on spline correction
        title_suffix = "(Spline Corrected)" if is_spline_corrected else "(Original Data)"
        self.setWindowTitle(f"Diagonal slice plotter {title_suffix}")
        self.setObjectName(f"Diagonal Plot {title_suffix}")  # Set object name for tab title

        self.main_layout = QVBoxLayout(self)

        self.contour_plot_widget = pg.PlotWidget(autoRange = False)
        self.diagonal_plot_widget = pg.PlotWidget(autoRange = False)

        self.tick_font = QFont( )
        self.tick_font.setPointSize(10)

        self.contour_plot_widget.getPlotItem( ).getAxis('bottom').tickFont = self.tick_font
        self.contour_plot_widget.getPlotItem( ).getAxis('left').tickFont = self.tick_font
        self.contour_plot_widget.getPlotItem( ).getAxis('bottom').setHeight(60)
        self.contour_plot_widget.getPlotItem( ).getAxis('left').setWidth(80)

        self.diagonal_plot_widget.getPlotItem( ).getAxis('bottom').tickFont = self.tick_font
        self.diagonal_plot_widget.getPlotItem( ).getAxis('left').tickFont = self.tick_font
        self.diagonal_plot_widget.getPlotItem( ).getAxis('bottom').setHeight(60)
        self.diagonal_plot_widget.getPlotItem( ).getAxis('left').setWidth(80)

        self.contour_plot_widget.getPlotItem( ).getAxis('bottom').setStyle(tickTextOffset = 10)
        self.contour_plot_widget.getPlotItem( ).getAxis('left').setStyle(tickTextOffset = 10)
        self.diagonal_plot_widget.getPlotItem( ).getAxis('bottom').setStyle(tickTextOffset = 10)
        self.diagonal_plot_widget.getPlotItem( ).getAxis('left').setStyle(tickTextOffset = 10)

        self.label_title_font = QFont( )
        self.label_title_font.setPointSize(10)

        self.contour_plot_widget.getPlotItem( ).getAxis('bottom').label.setFont(self.label_title_font)
        self.contour_plot_widget.getPlotItem( ).getAxis('left').label.setFont(self.label_title_font)

        self.diagonal_plot_widget.getPlotItem( ).getAxis('bottom').label.setFont(self.label_title_font)
        self.diagonal_plot_widget.getPlotItem( ).getAxis('left').label.setFont(self.label_title_font)

        self.image_item = pg.ImageItem( )

        self.diagonal_slice_curve = pg.PlotDataItem(pen = pg.mkPen('b', width = initial_diagonal_slice_linewidth))
        self.dashed_diagonal_line = pg.PlotDataItem(
            pen = pg.mkPen(initial_dashed_diagonal_line_color, width = initial_dashed_diagonal_line_linewidth,
                           style = QtCore.Qt.DashLine))
        self.current_dashed_line_color = initial_dashed_diagonal_line_color

        self.contour_plot_widget.setBackground('w')
        self.diagonal_plot_widget.setBackground('w')

        self.contour_plot_widget.setAspectLocked(True)

        self.contour_plot_widget.getViewBox( ).setMouseMode(pg.ViewBox.RectMode)

        self.diagonal_plot_widget.getViewBox( ).setMouseMode(pg.ViewBox.RectMode)

        self.main_layout.addWidget(self.contour_plot_widget, 5)
        self.main_layout.addWidget(self.diagonal_plot_widget, 2)

        level_input_layout = QHBoxLayout( )
        self.min_level_label = QLabel("Contour Min:")
        self.min_level_input = QLineEdit(self)
        self.min_level_input.setValidator(QDoubleValidator( ))
        self.min_level_input.editingFinished.connect(self._update_contour_levels)

        self.max_level_label = QLabel("Contour Max:")
        self.max_level_input = QLineEdit(self)
        self.max_level_input.setValidator(QDoubleValidator( ))
        self.max_level_input.editingFinished.connect(self._update_contour_levels)

        self.close_button = QPushButton("Close Tab")  # Add Close button
        self.close_button.clicked.connect(self._close_tab)  # Connect Close button to custom close method

        level_input_layout.addWidget(self.min_level_label)
        level_input_layout.addWidget(self.min_level_input)
        level_input_layout.addSpacing(20)
        level_input_layout.addWidget(self.max_level_label)
        level_input_layout.addWidget(self.max_level_input)
        level_input_layout.addStretch(1)
        level_input_layout.addWidget(self.close_button)  # Add Close button to layout

        self.main_layout.addLayout(level_input_layout, 1)

        # Add cursor position label - CORRECTED LINE
        self.cursor_pos_label = QLabel("Cursor: (X: -, Y: -)")
        self.cursor_pos_label.setFont(self.label_title_font)
        self.main_layout.addWidget(self.cursor_pos_label, alignment = Qt.AlignBottom | Qt.AlignLeft)

        # Connect mouse move signals for both plots to their specific handlers
        self.contour_plot_widget.scene( ).sigMouseMoved.connect(self._update_contour_cursor_pos)
        self.diagonal_plot_widget.scene( ).sigMouseMoved.connect(self._update_diagonal_cursor_pos)

        self.data_plotter = Plotting_contour_and_interpoaltion(processed_probe, processed_pump, processed_data)

        self.data_plotter.plot_diagonal_and_contour(self.contour_plot_widget,
                                                    self.diagonal_plot_widget,
                                                    self.image_item,
                                                    self.diagonal_slice_curve,
                                                    self.dashed_diagonal_line,
                                                    initial_min_level = initial_min_level,
                                                    initial_max_level = initial_max_level,
                                                    initial_diagonal_slice_linewidth = initial_diagonal_slice_linewidth,
                                                    initial_dashed_diagonal_line_linewidth = initial_dashed_diagonal_line_linewidth,
                                                    initial_dashed_diagonal_line_color = initial_dashed_diagonal_line_color,
                                                    initial_contour_xlabel = initial_contour_xlabel,
                                                    initial_contour_ylabel = initial_contour_ylabel,
                                                    initial_diagonal_xlabel = initial_diagonal_xlabel,
                                                    initial_diagonal_ylabel = initial_diagonal_ylabel)

        self.min_level_input.setText(f"{self.image_item.levels [ 0 ]:.4g}")
        self.max_level_input.setText(f"{self.image_item.levels [ 1 ]:.4g}")

        self._add_linewidth_context_menu(self.diagonal_plot_widget, self.diagonal_slice_curve,
                                         "Diagonal Slice Line Thickness")
        self._add_linewidth_context_menu(self.contour_plot_widget, self.dashed_diagonal_line,
                                         "Dashed Diagonal Line Thickness")

        self._add_label_context_menu(self.contour_plot_widget, "Contour Plot Labels")
        self._add_label_context_menu(self.diagonal_plot_widget, "Diagonal Plot Labels")

        self._add_dashed_line_color_context_menu(self.contour_plot_widget, self.dashed_diagonal_line,
                                                 "Change Dashed Line Color")

    def _update_contour_cursor_pos(self, evt):
        """
        Updates the cursor position label for the contour plot in the child window.
        """
        pos = evt
        if self.contour_plot_widget.sceneBoundingRect( ).contains(pos):
            mousePoint = self.contour_plot_widget.plotItem.vb.mapSceneToView(pos)
            x_val_display = mousePoint.x( )
            y_val_display = mousePoint.y( )
            self.cursor_pos_label.setText(f"Contour Cursor: (X: {x_val_display:.2f}, Y: {y_val_display:.2f})")
        else:
            # If mouse leaves this plot, clear or set generic cursor text
            self.cursor_pos_label.setText("Cursor: (X: -, Y: -)")

    def _update_diagonal_cursor_pos(self, evt):
        """
        Updates the cursor position label for the diagonal plot in the child window.
        """
        pos = evt
        if self.diagonal_plot_widget.sceneBoundingRect( ).contains(pos):
            mousePoint = self.diagonal_plot_widget.plotItem.vb.mapSceneToView(pos)
            x_val_display = mousePoint.x( )
            y_val_display = mousePoint.y( )
            self.cursor_pos_label.setText(f"Diagonal Cursor: (X: {x_val_display:.2f}, Y: {y_val_display:.2g})")
        else:
            # If mouse leaves this plot, clear or set generic cursor text
            self.cursor_pos_label.setText("Cursor: (X: -, Y: -)")

    def _update_contour_levels(self):
        """
        Reads values from min/max level input fields and updates the contour plot's levels.
        """
        try:
            min_level = float(self.min_level_input.text( ))
        except ValueError:
            min_level = np.min(self.data_plotter.data)
            self.min_level_input.setText(f"{min_level:.4g}")

        try:
            max_level = float(self.max_level_input.text( ))
        except ValueError:
            max_level = np.max(self.data_plotter.data)
            self.max_level_input.setText(f"{max_level:.4g}")

        if min_level > max_level:
            min_level, max_level = max_level, min_level
            self.min_level_input.setText(f"{min_level:.4g}")
            self.max_level_input.setText(f"{max_level:.4g}")

        self.image_item.setLevels((min_level, max_level))

    def _add_linewidth_context_menu(self, plot_widget, plot_item, action_text):
        """
        Adds a 'Change Line Thickness' action to the plot_widget's context menu.
        """
        vb_menu = plot_widget.getPlotItem( ).getViewBox( ).menu
        change_linewidth_action = vb_menu.addAction(action_text)
        change_linewidth_action.triggered.connect(lambda: self._show_linewidth_dialog(plot_item))

    def _show_linewidth_dialog(self, plot_item):
        """
        Opens the LineThicknessDialog and applies the new linewidth to the specified plot_item.
        """
        current_thickness = plot_item.opts [ 'pen' ].width( ) if plot_item.opts.get('pen') else 2
        dialog = LineThicknessDialog(self, initial_thickness = current_thickness)
        if dialog.exec_( ) == QDialog.Accepted:
            new_thickness = dialog.get_thickness( )
            current_pen = plot_item.opts [ 'pen' ]
            new_pen = pg.mkPen(current_pen.color( ), width = new_thickness, style = current_pen.style( ))
            plot_item.setPen(new_pen)

    def _add_label_context_menu(self, plot_widget, menu_title):
        """
        Adds a submenu for editing plot labels to the plot_widget's context menu.
        """
        vb_menu = plot_widget.getPlotItem( ).getViewBox( ).menu
        label_menu = vb_menu.addMenu(menu_title)

        set_xlabel_action = label_menu.addAction("Set X Label")
        set_xlabel_action.triggered.connect(lambda: self._show_set_xlabel_dialog(plot_widget))

        set_ylabel_action = label_menu.addAction("Set Y Label")
        set_ylabel_action.triggered.connect(lambda: self._show_set_ylabel_dialog(plot_widget))

    def _show_set_xlabel_dialog(self, plot_widget):
        """
        Opens a dialog to set the plot's X-axis label.
        """
        current_xlabel = plot_widget.getPlotItem( ).getAxis('bottom').labelText
        dialog = SetTextDialog(self, title = "Set X-axis Label", initial_text = current_xlabel)
        if dialog.exec_( ) == QDialog.Accepted:
            new_xlabel = dialog.get_text( )
            plot_widget.setLabel('bottom', new_xlabel, font = self.label_title_font)

    def _show_set_ylabel_dialog(self, plot_widget):
        """
        Opens a dialog to set the plot's Y-axis label.
        """
        current_ylabel = plot_widget.getPlotItem( ).getAxis('left').labelText
        dialog = SetTextDialog(self, title = "Set Y-axis Label", initial_text = current_ylabel)
        if dialog.exec_( ) == QDialog.Accepted:
            new_ylabel = dialog.get_text( )
            plot_widget.setLabel('left', new_ylabel, font = self.label_title_font)

    def _add_dashed_line_color_context_menu(self, plot_widget, plot_item, action_text):
        """
        Adds a 'Change Dashed Line Color' action to the plot_widget's context menu.
        """
        vb_menu = plot_widget.getPlotItem( ).getViewBox( ).menu
        change_color_action = vb_menu.addAction(action_text)
        change_color_action.triggered.connect(lambda: self._show_dashed_line_color_dialog(plot_item))

    def _show_dashed_line_color_dialog(self, plot_item):
        """
        Opens the ColorPickerDialog and applies the new color to the dashed diagonal line.
        """
        current_pen_color = plot_item.opts [ 'pen' ].color( )
        current_color_name = 'k'
        if current_pen_color.isValid( ):
            for name, val in ColorPickerDialog(self).color_map.items( ):
                if isinstance(val, str) and QColor(val) == current_pen_color:
                    current_color_name = val
                    break
                elif isinstance(val, tuple) and QColor(*val) == current_pen_color:
                    current_color_name = val
                    break
            else:
                current_color_name = current_pen_color.getRgb( ) [ :3 ]

        dialog = ColorPickerDialog(self, initial_color_name = current_color_name)
        if dialog.exec_( ) == QDialog.Accepted:
            new_color = dialog.get_color_name( )
            current_pen = plot_item.opts [ 'pen' ]
            new_pen = pg.mkPen(new_color, width = current_pen.width( ), style = current_pen.style( ))
            plot_item.setPen(new_pen)
            self.current_dashed_line_color = new_color

    def _close_tab(self):
        """
        Handles closing the tab by explicitly removing it from the parent QTabWidget.
        """
        # The parent of GaussianFitterApp is the QTabWidget
        parent_tab_widget = self.parent( )
        if parent_tab_widget and isinstance(parent_tab_widget, QTabWidget):
            tab_index = parent_tab_widget.indexOf(self)
            if tab_index != -1:
                parent_tab_widget.removeTab(tab_index)
        self.deleteLater( )  # Ensure the widget is properly deleted


class ClickableTextItem(pg.TextItem):
    """
    A custom TextItem that can respond to right-click events
    to show a context menu for removing itself and its associated scatter item.
    """

    def __init__(self, text="", color='magenta', anchor=(1, 1), parent_app=None, associated_scatter=None,
                 is_peak_label=False):
        if isinstance(color, str):
            q_color = QColor(color)
        elif isinstance(color, tuple) and len(color) in [ 3, 4 ]:
            if all(isinstance(c, float) and 0.0 <= c <= 1.0 for c in color):
                q_color = QColor(*(int(c * 255) for c in color))
            else:
                q_color = QColor(*color)  # Corrected from Color to QColor
        else:
            q_color = color

        super( ).__init__(text = text, color = q_color, anchor = anchor)
        self.setFlag(self.GraphicsItemFlag.ItemIsMovable, True)
        self.parent_app = parent_app
        self.associated_scatter = associated_scatter
        self.self_ref = self
        self.is_peak_label = is_peak_label
        self._stored_text = text
        self._stored_color = q_color

    def text(self):
        return self._stored_text

    def get_rgb_f_color(self):
        return self._stored_color.getRgbF( )

    def mousePressEvent(self, ev):
        """
        Handles mouse press events for the TextItem.
        If right-clicked, shows a context menu to remove the marker.
        """
        if ev.button( ) == Qt.RightButton:
            self.menu = QMenu( )
            action_text = "Remove Peak" if self.is_peak_label else "Remove Marker"
            remove_action = self.menu.addAction(action_text)
            remove_action.triggered.connect(lambda: self._remove_self_callback(self.associated_scatter, self.self_ref))
            self.menu.exec_(ev.screenPos( ))
            ev.accept( )
        else:
            super( ).mousePressEvent(ev)

    def _remove_self_callback(self, scatter_item, text_item):
        """
        Callback to trigger the removal of this specific marker/peak in the parent application.
        """
        if self.parent_app:
            self.parent_app.remove_plot_item_pair(scatter_item, text_item)


class SignalPlotterApp(QMainWindow):
    """
    A PyQt5 application to display a 2D signal plot and its
    X and Y axis slices.
    Sliders are provided to control the slice positions interactively.
    It shows cursor coordinates only in a single label at the bottom-left.
    'Hold' and 'Clear' buttons for slice plots to add persistent slice curves
    and clear them.
    'Fit' buttons for slice plots to launch an interactive signal fitter.
    'Find Peaks' button with interactive parameters for peak detection.
    """

    def __init__(self):
        super( ).__init__( )
        self.base_title = "MjölnIR"
        self._current_project_file = None
        self._data_modified = False
        self._update_window_title( )

        self.setGeometry(100, 100, 2500, 1500)
        self.base_font_size = 12
        self.axis_label_font_size = 12
        self.axis_scale_font_size = 12
        self.central_widget = QWidget( )

        self.setCentralWidget(self.central_widget)
        self.main_layout = QGridLayout(self.central_widget)
        self.held_x_slices_count = 0
        self.held_y_slices_count = 0
        self.placed_markers = [ ]
        self.marker_font_size = 10
        self.last_right_click_data_pos = None
        self.detected_peaks_items = [ ]
        self.show_peak_labels = True
        self.peak_finder_params = {
            'neighborhood_size': 10,
            'positive_threshold': 0.1,
            'negative_threshold': -0.1,
            'smooth_sigma': 0.0
        }
        self.plot_colors = [
            (255, 0, 0),
            (0, 0, 255),
            (0, 200, 0),
            (255, 165, 0),
            (128, 0, 128),
            (0, 255, 255),
            (255, 0, 255),
            (150, 75, 0),
            (0, 128, 128),
            (255, 255, 0)
        ]
        self._initial_raw_x_values = None
        self._initial_raw_y_values = None
        self._initial_raw_signal_data = None
        self.current_x_values = None
        self.current_y_values = None
        self.current_signal_data = None
        self.is_spline_corrected = False

        self.x_values_interp = None
        self.y_values_interp = None
        self.signal_data_interp = None

        self.x_dim = 0
        self.y_dim = 0
        self.data_loaded = False

        self.x_legend_font_size = 14
        self.y_legend_font_size = 14

        self.axis_labels = {
            'signal_bottom': 'Probe wavenumber [cm\u207B\u00B9]',
            'signal_left': 'Pump wavenumber [cm\u207B\u00B9]',
            'x_slice_bottom': 'Pump wavenumber [cm\u207B\u00B9]',
            'x_slice_left': 'ΔOD',
            'y_slice_bottom': 'Probe wavenumber [cm\u207B\u00B9]',
            'y_slice_left': 'ΔOD'
        }

        self.current_slice_linewidth = 2
        self.x_slice_legend_unit = "cm^-1"
        self.y_slice_legend_unit = "cm^-1"

        # Interpolation settings for main window slices
        self._current_interp_method = "None"
        self._current_interp_multiplier = 1
        self._original_x_slice_data_main = None # Stores original x-data for x-slice plot
        self._original_y_slice_data_main = None # Stores original y-data for x-slice plot
        self._original_y_slice_data_main_y = None # Stores original x-data for y-slice plot
        self._original_y_slice_data_main_x = None # Stores original y-data for y-slice plot


        # Modified to handle two separate diagonal plot instances
        self.original_diagonal_plot_widget_instance = None
        self.spline_corrected_diagonal_plot_widget_instance = None
        self.saved_original_diagonal_plot_state = None
        self.saved_spline_corrected_diagonal_plot_state = None

        # List to hold references to opened fitter tabs
        self.active_fitter_tabs = [ ]

        self.init_ui( )

        self.update_plots( )

        self.x_unit_input.textChanged.connect(self._set_data_modified)
        self.y_unit_input.textChanged.connect(self._set_data_modified)
        self.min_level_input.textChanged.connect(self._set_data_modified)
        self.max_level_input.textChanged.connect(self._set_data_modified)
        self.marker_font_size_input.valueChanged.connect(self._update_marker_font_size)

    def _set_data_modified(self):
        if not self._data_modified:
            self._data_modified = True
            self._update_window_title( )

    def _update_window_title(self):
        """Updates the main window title based on project file and modified status."""
        title = self.base_title
        if self._current_project_file:
            project_name = os.path.basename(self._current_project_file)
            title += f" - {project_name}"
        else:
            title += " - Unsaved"

        if self._data_modified and self._current_project_file:
            title += " (Unsaved Changes)"
        elif self._data_modified and not self._current_project_file:
            title += " (Unsaved)"

        self.setWindowTitle(title)

    def _load_data_into_plots(self, x_raw, y_raw, z_raw):
        """
        Loads the 2D data into the application, performs interpolation,
        and updates all plots and slider ranges.
        """
        self._initial_raw_x_values = x_raw.copy( )
        self._initial_raw_y_values = y_raw.copy( )
        self._initial_raw_signal_data = z_raw.copy( )

        self.current_x_values = x_raw.copy( )
        self.current_y_values = y_raw.copy( )
        self.current_signal_data = z_raw.copy( )
        self.is_spline_corrected = False

        print(f"Original data shape: {self.current_signal_data.shape}")
        print(f"Original X range: {self.current_x_values.min( ):.2f} to {self.current_x_values.max( ):.2f}")
        print(f"Original Y range: {self.current_y_values.min( ):.2f} to {self.current_y_values.max( ):.2f}")

        self.data_loaded = True
        self._refresh_all_plots( )
        self._set_data_modified( )

    def _refresh_all_plots(self, preserve_contour_levels=False, min_level=None, max_level=None,
                           preserve_plot_ranges=False, signal_xlim=None, signal_ylim=None,
                           x_slice_xlim=None, x_slice_ylim=None, y_slice_xlim=None, y_slice_ylim=None):
        """
        Refreshes all plots and UI elements based on the current_signal_data.
        This method is called after any data modification (load, spline, revert).
        Can preserve contour levels and plot ranges if specified.
        """
        if not self.data_loaded:
            self.update_plots( )
            return

        self.new_x_resolution = 1000
        self.new_y_resolution = 1000

        self.x_values_interp = np.linspace(self.current_x_values.min( ), self.current_x_values.max( ),
                                           self.new_x_resolution)
        self.y_values_interp = np.linspace(self.current_y_values.min( ), self.current_y_values.max( ),
                                           self.new_y_resolution)

        interp_func = RectBivariateSpline(self.current_y_values, self.current_x_values, self.current_signal_data)
        self.signal_data_interp = interp_func(self.y_values_interp, self.x_values_interp)

        self.x_dim = self.new_x_resolution
        self.y_dim = self.new_y_resolution

        self.x_slider.setRange(0, self.x_dim - 1)
        self.y_slider.setRange(0, self.y_dim - 1)

        self.image_item.setImage(self.signal_data_interp.T)
        self.image_item.setRect(pg.QtCore.QRectF(
            self.x_values_interp [ 0 ],
            self.y_values_interp [ 0 ],
            self.x_values_interp [ -1 ] - self.x_values_interp [ 0 ],
            self.y_values_interp [ -1 ] - self.y_values_interp [ 0 ]
        ))

        # Apply contour levels
        if preserve_contour_levels and min_level is not None and max_level is not None:
            self.min_level_input.setText(f"{min_level:.2f}")
            self.max_level_input.setText(f"{max_level:.2f}")
            self.image_item.setLevels((min_level, max_level))
        else:
            self.min_level_input.setText(f"{self.signal_data_interp.min( ):.2f}")
            self.max_level_input.setText(f"{self.signal_data_interp.max( ):.2f}")
            self.image_item.setLevels((self.signal_data_interp.min( ), self.signal_data_interp.max( )))

        # Apply plot ranges
        if preserve_plot_ranges and signal_xlim is not None and signal_ylim is not None:
            self.signal_plot_widget.setXRange(signal_xlim [ 0 ], signal_xlim [ 1 ], padding = 0)
            self.signal_plot_widget.setYRange(signal_ylim [ 0 ], signal_ylim [ 1 ], padding = 0)
        else:
            self.signal_plot_widget.setXRange(self.x_values_interp.min( ), self.x_values_interp.max( ), padding = 0)
            self.signal_plot_widget.setYRange(self.y_values_interp.min( ), self.y_values_interp.max( ), padding = 0)

        if preserve_plot_ranges and x_slice_xlim is not None and x_slice_ylim is not None:
            self.x_slice_plot_widget.setXRange(x_slice_xlim [ 0 ], x_slice_xlim [ 1 ], padding = 0.05)
            self.x_slice_plot_widget.setYRange(x_slice_ylim [ 0 ], x_slice_ylim [ 1 ], padding = 0.05)
        else:
            self.x_slice_plot_widget.setXRange(self.current_y_values.min( ), self.current_y_values.max( ),
                                               padding = 0.05)
            self.x_slice_plot_widget.setYRange(self.current_signal_data.min( ), self.current_signal_data.max( ),
                                               padding = 0.05)

        if preserve_plot_ranges and y_slice_xlim is not None and y_slice_ylim is not None:
            self.y_slice_plot_widget.setXRange(y_slice_xlim [ 0 ], y_slice_xlim [ 1 ], padding = 0.05)
            self.y_slice_plot_widget.setYRange(y_slice_ylim [ 0 ], y_slice_ylim [ 1 ], padding = 0.05)
        else:
            self.y_slice_plot_widget.setXRange(self.current_x_values.min( ), self.current_x_values.max( ),
                                               padding = 0.05)
            self.y_slice_plot_widget.setYRange(self.current_signal_data.min( ), self.current_signal_data.max( ),
                                               padding = 0.05)

        # Capture original slice data for main window plots after 2D data is set
        x_idx_interp = self.x_slider.value()
        y_idx_interp = self.y_slider.value()

        x_pos_val_interp = self.x_values_interp[x_idx_interp]
        y_pos_val_interp = self.y_values_interp[y_idx_interp]

        original_y_idx = np.argmin(np.abs(self.current_y_values - y_pos_val_interp))
        self._original_y_slice_data_main_x = np.copy(self.current_x_values)
        self._original_y_slice_data_main_y = np.copy(self.current_signal_data[original_y_idx, :])

        original_x_idx = np.argmin(np.abs(self.current_x_values - x_pos_val_interp))
        self._original_x_slice_data_main = np.copy(self.current_y_values)
        self._original_x_slice_data_main_y = np.copy(self.current_signal_data[:, original_x_idx])


        self.clear_all_markers( )
        self.clear_detected_peaks( )
        self.clear_x_slice_plots( )
        self.clear_y_slice_plots( )

        self.update_plots( ) # This will now use the interpolated data based on _current_interp_method/_multiplier

        # Update Original Data Diagonal Plot if it's open and its state is saved
        if self.original_diagonal_plot_widget_instance and self.saved_original_diagonal_plot_state:
            # Re-process data for the original diagonal plot based on original raw data
            original_data_processor = Plotting_contour_and_interpoaltion(
                self._initial_raw_x_values, self._initial_raw_y_values, self._initial_raw_signal_data
            )
            current_diagonal_params = self.saved_original_diagonal_plot_state [ 'params' ]
            original_data_processor.choose_region(current_diagonal_params [ 'start_wn' ],
                                                  current_diagonal_params [ 'stop_wn' ],
                                                  current_diagonal_params [ 'shift_index' ])
            original_data_processor.interpolate(current_diagonal_params [ 'npoints' ],
                                                  current_diagonal_params [ 'interp_method' ])

            # Update the existing instance's data
            self.original_diagonal_plot_widget_instance.data_plotter.probe = original_data_processor.probe
            self.original_diagonal_plot_widget_instance.data_plotter.pump = original_data_processor.pump
            self.original_diagonal_plot_widget_instance.data_plotter.data = original_data_processor.data
            self.original_diagonal_plot_widget_instance.data_plotter.plot_diagonal_and_contour(
                self.original_diagonal_plot_widget_instance.contour_plot_widget,
                self.original_diagonal_plot_widget_instance.diagonal_plot_widget,
                self.original_diagonal_plot_widget_instance.image_item,
                self.original_diagonal_plot_widget_instance.diagonal_slice_curve,
                self.original_diagonal_plot_widget_instance.dashed_diagonal_line,
                initial_min_level = float(self.original_diagonal_plot_widget_instance.min_level_input.text( )),
                # Ensure float conversion
                initial_max_level = float(self.original_diagonal_plot_widget_instance.max_level_input.text( )),
                # Ensure float conversion
                initial_diagonal_slice_linewidth =
                self.original_diagonal_plot_widget_instance.diagonal_slice_curve.opts [ 'pen' ].width( ),
                initial_dashed_diagonal_line_linewidth =
                self.original_diagonal_plot_widget_instance.dashed_diagonal_line.opts [ 'pen' ].width( ),
                initial_dashed_diagonal_line_color = self.original_diagonal_plot_widget_instance.current_dashed_line_color,
                initial_contour_xlabel = self.original_diagonal_plot_widget_instance.contour_plot_widget.getPlotItem( ).getAxis(
                    'bottom').labelText,
                initial_contour_ylabel = self.original_diagonal_plot_widget_instance.contour_plot_widget.getPlotItem( ).getAxis(
                    'left').labelText,
                initial_diagonal_xlabel = self.original_diagonal_plot_widget_instance.diagonal_plot_widget.getPlotItem( ).getAxis(
                    'bottom').labelText,
                initial_diagonal_ylabel = self.original_diagonal_plot_widget_instance.diagonal_plot_widget.getPlotItem( ).getAxis(
                    'left').labelText
            )
            self.original_diagonal_plot_widget_instance.min_level_input.setText(
                f"{self.original_diagonal_plot_widget_instance.image_item.levels [ 0 ]:.4g}")
            self.original_diagonal_plot_widget_instance.max_level_input.setText(
                f"{self.original_diagonal_plot_widget_instance.image_item.levels [ 1 ]:.4g}")
            # Ensure the saved state is also updated with the new data
            self.saved_original_diagonal_plot_state [ 'probe' ] = original_data_processor.probe.tolist( )
            self.saved_original_diagonal_plot_state [ 'pump' ] = original_data_processor.pump.tolist( )
            self.saved_original_diagonal_plot_state [ 'data' ] = original_data_processor.data.tolist( )

        # Update Spline Corrected Data Diagonal Plot if it's open and its state is saved
        if self.spline_corrected_diagonal_plot_widget_instance and self.saved_spline_corrected_diagonal_plot_state:
            # Re-process data for the spline corrected diagonal plot based on spline-corrected raw data
            spline_corrected_raw_data = self.spline_baseline_correction(
                self._initial_raw_signal_data, self._initial_raw_x_values
            )
            spline_data_processor = Plotting_contour_and_interpoaltion(
                self._initial_raw_x_values, self._initial_raw_y_values, spline_corrected_raw_data
            )
            current_diagonal_params = self.saved_spline_corrected_diagonal_plot_state [ 'params' ]
            spline_data_processor.choose_region(current_diagonal_params [ 'start_wn' ],
                                                current_diagonal_params [ 'stop_wn' ],
                                                current_diagonal_params [ 'shift_index' ])
            spline_data_processor.interpolate(current_diagonal_params [ 'npoints' ],
                                              current_diagonal_params [ 'interp_method' ])

            # Update the existing instance's data
            self.spline_corrected_diagonal_plot_widget_instance.data_plotter.probe = spline_data_processor.probe
            self.spline_corrected_diagonal_plot_widget_instance.data_plotter.pump = spline_data_processor.pump
            self.spline_corrected_diagonal_plot_widget_instance.data_plotter.data = spline_data_processor.data
            self.spline_corrected_diagonal_plot_widget_instance.data_plotter.plot_diagonal_and_contour(
                self.spline_corrected_diagonal_plot_widget_instance.contour_plot_widget,
                self.spline_corrected_diagonal_plot_widget_instance.diagonal_plot_widget,
                self.spline_corrected_diagonal_plot_widget_instance.image_item,
                self.spline_corrected_diagonal_plot_widget_instance.diagonal_slice_curve,
                self.spline_corrected_diagonal_plot_widget_instance.dashed_diagonal_line,
                initial_min_level = float(self.spline_corrected_diagonal_plot_widget_instance.min_level_input.text( )),
                # Ensure float conversion
                initial_max_level = float(self.spline_corrected_diagonal_plot_widget_instance.max_level_input.text( )),
                # Ensure float conversion
                initial_diagonal_slice_linewidth =
                self.spline_corrected_diagonal_plot_widget_instance.diagonal_slice_curve.opts [ 'pen' ].width( ),
                initial_dashed_diagonal_line_linewidth =
                self.spline_corrected_diagonal_plot_widget_instance.dashed_diagonal_line.opts [ 'pen' ].width( ),
                initial_dashed_diagonal_line_color = self.spline_corrected_diagonal_plot_widget_instance.current_dashed_line_color,
                initial_contour_xlabel = self.spline_corrected_diagonal_plot_widget_instance.contour_plot_widget.getPlotItem( ).getAxis(
                    'bottom').labelText,
                initial_contour_ylabel = self.spline_corrected_diagonal_plot_widget_instance.contour_plot_widget.getPlotItem( ).getAxis(
                    'left').labelText,
                initial_diagonal_xlabel = self.spline_corrected_diagonal_plot_widget_instance.diagonal_plot_widget.getPlotItem( ).getAxis(
                    'bottom').labelText,
                initial_diagonal_ylabel = self.spline_corrected_diagonal_plot_widget_instance.diagonal_plot_widget.getPlotItem( ).getAxis(
                    'left').labelText
            )
            self.spline_corrected_diagonal_plot_widget_instance.min_level_input.setText(
                f"{self.spline_corrected_diagonal_plot_widget_instance.image_item.levels [ 0 ]:.4g}")
            self.spline_corrected_diagonal_plot_widget_instance.max_level_input.setText(
                f"{self.spline_corrected_diagonal_plot_widget_instance.image_item.levels [ 1 ]:.4g}")
            # Ensure the saved state is also updated with the new data
            self.saved_spline_corrected_diagonal_plot_state [ 'probe' ] = spline_data_processor.probe.tolist( )
            self.saved_spline_corrected_diagonal_plot_state [ 'pump' ] = spline_data_processor.pump.tolist( )
            self.saved_spline_corrected_diagonal_plot_state [ 'data' ] = spline_data_processor.data.tolist( )

        self._update_spline_button_text( )

    def init_ui(self):
        """
        Initializes all the UI components: plots, sliders, and labels.
        """
        self.menu_bar = self.menuBar( )
        self.file_menu = self.menu_bar.addMenu("&File")

        import_data_action = QAction("&Import Data...", self)
        import_data_action.setShortcut("Ctrl+I")
        import_data_action.setStatusTip("Import data from a file")
        import_data_action.triggered.connect(self.on_import_data_action_triggered)
        self.file_menu.addAction(import_data_action)

        save_project_action = QAction("&Save Project", self)
        save_project_action.setShortcut("Ctrl+S")
        save_project_action.setStatusTip("Save current project state")
        save_project_action.triggered.connect(self._save_project)
        self.file_menu.addAction(save_project_action)

        save_project_as_action = QAction("Save Project &As...", self)
        save_project_as_action.setShortcut("Ctrl+Shift+S")
        save_project_as_action.setStatusTip("Save current project with a new name")
        save_project_as_action.triggered.connect(self._save_project_as)
        self.file_menu.addAction(save_project_as_action)

        load_project_action = QAction("&Load Project...", self)
        load_project_action.setShortcut("Ctrl+L")
        load_project_action.setStatusTip("Load a previously saved project")
        load_project_action.triggered.connect(self._load_project)
        self.file_menu.addAction(load_project_action)

        exit_action = QAction("&Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        self.file_menu.addAction(exit_action)

        self.edit_menu = self.menu_bar.addMenu("&Edit")
        edit_names_action = QAction("Edit &Names...", self)
        edit_names_action.setShortcut("Ctrl+N")
        edit_names_action.setStatusTip("Edit axis labels for all plots")
        edit_names_action.triggered.connect(self._show_edit_names_dialog)
        self.edit_menu.addAction(edit_names_action)

        # Removed Interpolation Controls from menu bar

        label_font = QFont("Times New Roman")
        label_font.setPointSize(self.base_font_size)

        axis_scale_font = QFont("Times New Roman")
        axis_scale_font.setPointSize(self.axis_scale_font_size)

        axis_label_font = QFont("Times New Roman")
        axis_label_font.setPointSize(self.axis_label_font_size)

        tick_length = 10
        tick_text_offset = 5

        self.tab_widget = QTabWidget( )
        self.main_plot_tab = QWidget( )
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.addTab(self.main_plot_tab, "Main Plots")
        self.tab_widget.tabCloseRequested.connect(self.close_tab)

        self.main_plot_tab_layout = QGridLayout(self.main_plot_tab)

        self.signal_plot_widget = pg.PlotWidget( )
        self.signal_plot_widget.setLabel('bottom', self.axis_labels [ 'signal_bottom' ],
                                         **{ 'font-size': f'{self.axis_label_font_size}pt' })
        self.signal_plot_widget.setLabel('left', self.axis_labels [ 'signal_left' ],
                                         **{ 'font-size': f'{self.axis_label_font_size}pt' })
        # self.signal_plot_widget.setAspectLocked(True)
        self.signal_plot_widget.setBackground('w')
        self.signal_plot_widget.getViewBox( ).enableAutoRange(axis = pg.ViewBox.XYAxes, enable = False)
        self.signal_plot_widget.getViewBox( ).setMouseMode(pg.ViewBox.RectMode)
        self.signal_plot_widget.getViewBox( ).setMouseEnabled(x = True, y = True)

        bottom_axis = self.signal_plot_widget.getAxis('bottom')
        left_axis = self.signal_plot_widget.getAxis('left')

        bottom_axis.setTickFont(axis_scale_font)
        left_axis.setTickFont(axis_scale_font)

        bottom_axis.showMinorTicks = True
        left_axis.showMinorTicks = True
        bottom_axis.tickLength = tick_length
        left_axis.tickLength = tick_length
        bottom_axis.setStyle(tickTextOffset = tick_text_offset)
        left_axis.setStyle(tickTextOffset = tick_text_offset)

        self.image_item = pg.ImageItem( )
        self.signal_plot_widget.addItem(self.image_item)

        colors_seismic_base = [
            (0, 0, 128, 255),
            (0, 0, 255, 255),
            (0, 255, 255, 255),
            (255, 255, 255, 255),
            (255, 255, 0, 255),
            (255, 0, 0, 255),
            (128, 0, 0, 255)
        ]
        positions_seismic_base = np.linspace(0.0, 0.99, len(colors_seismic_base))
        final_colors = colors_seismic_base + [ (255, 255, 255, 255) ]
        final_positions = np.append(positions_seismic_base, 1.0)
        self.seismic_colormap = pg.ColorMap(final_positions, final_colors)
        self.image_item.setLookupTable(self.seismic_colormap.getLookupTable( ))

        self.cursor_x_line = pg.InfiniteLine(angle = 90, movable = False,
                                             pen = pg.mkPen('k', width = 2, style = Qt.DotLine))
        self.cursor_y_line = pg.InfiniteLine(angle = 0, movable = False,
                                             pen = pg.mkPen('k', width = 2, style = Qt.DotLine))
        self.signal_plot_widget.addItem(self.cursor_x_line)
        self.signal_plot_widget.addItem(self.cursor_y_line)
        self.cursor_x_line.setVisible(False)
        self.cursor_y_line.setVisible(False)

        self.signal_plot_widget.scene( ).sigMouseMoved.connect(self.update_signal_cursor_pos)
        self.signal_plot_widget.scene( ).sigMouseClicked.connect(self.on_signal_plot_clicked)

        vb_menu = self.signal_plot_widget.getPlotItem( ).getViewBox( ).menu
        self.place_marker_action = vb_menu.addAction("Place Marker")
        self.place_marker_action.triggered.connect(self._place_marker_from_context_menu)
        self.clear_markers_action = vb_menu.addAction("Clear All Markers")
        self.clear_markers_action.triggered.connect(self.clear_all_markers)
        self.find_peaks_action = vb_menu.addAction("Find Peaks...")
        self.find_peaks_action.triggered.connect(self._show_peak_finder_dialog)
        self.clear_detected_peaks_action = vb_menu.addAction("Clear Detected Peaks")
        self.clear_detected_peaks_action.triggered.connect(self.clear_detected_peaks)
        self.toggle_peak_labels_action = vb_menu.addAction("Toggle Peak Labels")
        self.toggle_peak_labels_action.triggered.connect(self._toggle_peak_labels_visibility)

        self.x_slice_plot_widget = pg.PlotWidget( )
        self.x_slice_plot_widget.setLabel('bottom', self.axis_labels [ 'x_slice_bottom' ],
                                          **{ 'font-size': f'{self.axis_label_font_size}pt' })
        self.x_slice_plot_widget.setLabel('left', self.axis_labels [ 'x_slice_left' ],
                                          **{ 'font-size': f'{self.axis_label_font_size}pt' })
        self.x_slice_plot_widget.setBackground('w')
        self.x_slice_plot_widget.getViewBox( ).enableAutoRange(axis = pg.ViewBox.XYAxes, enable = False)
        self.x_slice_plot_widget.getViewBox( ).setMouseMode(pg.ViewBox.RectMode)
        self.x_slice_plot_widget.getViewBox( ).setMouseEnabled(x = True, y = True)

        x_slice_bottom_axis = self.x_slice_plot_widget.getAxis('bottom')
        x_slice_left_axis = self.x_slice_plot_widget.getAxis('left')

        x_slice_bottom_axis.setTickFont(axis_scale_font)
        x_slice_left_axis.setTickFont(axis_scale_font)

        x_slice_bottom_axis.showMinorTicks = True
        x_slice_left_axis.showMinorTicks = True
        x_slice_bottom_axis.tickLength = tick_length
        x_slice_left_axis.tickLength = tick_length
        x_slice_bottom_axis.setStyle(tickTextOffset = tick_text_offset)
        x_slice_left_axis.setStyle(tickTextOffset = tick_text_offset)
        self.x_slice_curve = self.x_slice_plot_widget.plot(pen = pg.mkPen('b', width = self.current_slice_linewidth))
        self.x_slice_legend = self.x_slice_plot_widget.addLegend( )
        self._apply_legend_font_size(self.x_slice_legend, self.x_legend_font_size)

        x_vb_menu = self.x_slice_plot_widget.getPlotItem( ).getViewBox( ).menu
        self.change_x_linewidth_action = x_vb_menu.addAction("Change Line Thickness")
        self.change_x_linewidth_action.triggered.connect(lambda: self._show_linewidth_dialog(self.x_slice_plot_widget))

        self.x_slice_plot_widget.scene( ).sigMouseMoved.connect(self.update_x_slice_cursor_pos)

        self.x_hold_button = QPushButton("Hold")
        self.x_hold_button.setFont(label_font)
        self.x_hold_button.clicked.connect(self.hold_x_slice_plot)
        self.x_hold_button.setMinimumHeight(30)

        self.x_unit_input = QLineEdit(self)
        self.x_unit_input.setFont(label_font)
        self.x_unit_input.setPlaceholderText("Enter X-slice unit (e.g., cm^-1)")
        self.x_unit_input.setText(self.x_slice_legend_unit)
        self.x_unit_input.setFixedWidth(150)

        self.x_clear_button = QPushButton("Clear")
        self.x_clear_button.setFont(label_font)
        self.x_clear_button.clicked.connect(self.clear_x_slice_plots)
        self.x_clear_button.setMinimumHeight(30)

        # Add QComboBox for fitting function selection for X-slice
        self.x_fit_function_selector = QComboBox(self)
        self.x_fit_function_selector.addItem("Gaussian")
        self.x_fit_function_selector.addItem("Lorentzian")
        self.x_fit_function_selector.setMinimumHeight(30)
        self.x_fit_function_selector.setFont(label_font)

        self.x_fit_button = QPushButton("Fit")
        self.x_fit_button.setFont(label_font)
        self.x_fit_button.setMinimumHeight(30)
        # Modified to pass the selected fitting function type and slice info
        self.x_fit_button.clicked.connect(
            lambda: self._open_fitter_tab(
                self.x_slice_curve,
                True,
                self.x_fit_function_selector.currentText( ),
                self.axis_labels [ 'x_slice_bottom' ],  # Slice plot's X-axis label
                self.axis_labels [ 'x_slice_left' ],  # Slice plot's Y-axis label
                self._strip_html_tags(self.axis_labels [ 'signal_bottom' ]),
                # Slice axis name (e.g., "Probe wavenumber")
                self.x_values_interp [ self.x_slider.value( ) ],  # Slice value
                self._format_unit_for_display(self.x_unit_input.text( )),  # Slice unit, formatted
                self.is_spline_corrected  # Pass spline status
            )
        )

        self.y_slice_plot_widget = pg.PlotWidget( )
        self.y_slice_plot_widget.setLabel('bottom', self.axis_labels [ 'y_slice_bottom' ],
                                          **{ 'font-size': f'{self.axis_label_font_size}pt' })
        self.y_slice_plot_widget.setLabel('left', self.axis_labels [ 'y_slice_left' ],
                                          **{ 'font-size': f'{self.axis_label_font_size}pt' })
        self.y_slice_plot_widget.setBackground('w')
        self.y_slice_plot_widget.getViewBox( ).enableAutoRange(axis = pg.ViewBox.XYAxes, enable = False)
        self.y_slice_plot_widget.getViewBox( ).setMouseMode(pg.ViewBox.RectMode)
        self.y_slice_plot_widget.getViewBox( ).setMouseEnabled(x = True, y = True)

        y_slice_bottom_axis = self.y_slice_plot_widget.getAxis('bottom')
        y_slice_left_axis = self.y_slice_plot_widget.getAxis('left')

        y_slice_bottom_axis.setTickFont(axis_scale_font)
        y_slice_left_axis.setTickFont(axis_scale_font)

        y_slice_bottom_axis.showMinorTicks = True
        y_slice_left_axis.showMinorTicks = True
        y_slice_bottom_axis.tickLength = tick_length
        y_slice_left_axis.tickLength = tick_length
        y_slice_bottom_axis.setStyle(tickTextOffset = tick_text_offset)
        y_slice_left_axis.setStyle(tickTextOffset = tick_text_offset)

        self.y_slice_curve = self.y_slice_plot_widget.plot(pen = pg.mkPen('r', width = self.current_slice_linewidth))
        self.y_slice_legend = self.y_slice_plot_widget.addLegend( )
        self._apply_legend_font_size(self.y_slice_legend, self.y_legend_font_size)

        y_vb_menu = self.y_slice_plot_widget.getPlotItem( ).getViewBox( ).menu
        self.change_y_linewidth_action = y_vb_menu.addAction("Change Line Thickness...")
        self.change_y_linewidth_action.triggered.connect(lambda: self._show_linewidth_dialog(self.y_slice_plot_widget))

        self.y_slice_plot_widget.scene( ).sigMouseMoved.connect(self.update_y_slice_cursor_pos)

        self.y_hold_button = QPushButton("Hold")
        self.y_hold_button.setFont(label_font)
        self.y_hold_button.clicked.connect(self.hold_y_slice_plot)
        self.y_hold_button.setMinimumHeight(30)

        self.y_unit_input = QLineEdit(self)
        self.y_unit_input.setFont(label_font)
        self.y_unit_input.setPlaceholderText("Enter Y-slice unit (e.g., cm^-1)")
        self.y_unit_input.setText(self.y_slice_legend_unit)
        self.y_unit_input.setFixedWidth(150)

        self.y_clear_button = QPushButton("Clear")
        self.y_clear_button.setFont(label_font)
        self.y_clear_button.clicked.connect(self.clear_y_slice_plots)
        self.y_clear_button.setMinimumHeight(30)

        # Add QComboBox for fitting function selection for Y-slice
        self.y_fit_function_selector = QComboBox(self)
        self.y_fit_function_selector.addItem("Gaussian")
        self.y_fit_function_selector.addItem("Lorentzian")
        self.y_fit_function_selector.setMinimumHeight(30)
        self.y_fit_function_selector.setFont(label_font)

        self.y_fit_button = QPushButton("Fit")
        self.y_fit_button.setFont(label_font)
        self.y_fit_button.setMinimumHeight(30)
        # Modified to pass the selected fitting function type and slice info
        self.y_fit_button.clicked.connect(
            lambda: self._open_fitter_tab(
                self.y_slice_curve,
                False,
                self.y_fit_function_selector.currentText( ),
                self.axis_labels [ 'y_slice_bottom' ],  # Slice plot's X-axis label
                self.axis_labels [ 'y_slice_left' ],  # Slice plot's Y-axis label
                self._strip_html_tags(self.axis_labels [ 'signal_left' ]),  # Slice axis name (e.g., "Pump wavenumber")
                self.y_values_interp [ self.y_slider.value( ) ],  # Slice value
                self._format_unit_for_display(self.y_unit_input.text( )),  # Slice unit, formatted
                self.is_spline_corrected  # Pass spline status
            )
        )

        self.x_slider = QSlider(Qt.Horizontal)
        self.x_slider.setTickPosition(QSlider.TicksBelow)
        self.x_slider.valueChanged.connect(self.update_plots)
        self.x_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: #e0e0e0;
                height: 10px;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 1px solid #3498db;
                width: 20px;
                margin: -5px 0;
                border-radius: 10px;
            }
            QSlider::sub-page:horizontal {
                background: #2980b9;
                border-radius: 5px;
            }
        """)

        self.y_slider = QSlider(Qt.Horizontal)
        self.y_slider.setTickPosition(QSlider.TicksBelow)
        self.y_slider.valueChanged.connect(self.update_plots)
        self.y_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: #e0e0e0;
                height: 10px;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 1px solid #3498db;
                width: 20px;
                margin: -5px 0;
                border-radius: 10px;
            }
            QSlider::sub-page:horizontal {
                background: #2980b9;
                border-radius: 5px;
            }
        """)

        self.min_level_input = QLineEdit(self)
        self.min_level_input.setFont(label_font)
        self.min_level_input.setValidator(QDoubleValidator( ))
        self.min_level_input.editingFinished.connect(self.update_plots)
        self.min_level_input.setFixedWidth(100)

        self.max_level_input = QLineEdit(self)
        self.max_level_input.setFont(label_font)
        self.max_level_input.setValidator(QDoubleValidator( ))
        self.max_level_input.editingFinished.connect(self.update_plots)
        self.max_level_input.setFixedWidth(100)

        self.marker_font_size_input = QSpinBox(self)
        self.marker_font_size_input.setFont(label_font)
        self.marker_font_size_input.setRange(6, 30)
        self.marker_font_size_input.setValue(self.marker_font_size)

        self.x_label = QLabel(f"X Slice Position: -")
        self.x_label.setFont(label_font)
        self.y_label = QLabel(f"Y Slice Position: -")
        self.y_label.setFont(label_font)
        self.min_level_label = QLabel("Contour min:")
        self.min_level_label.setFont(label_font)
        self.max_level_label = QLabel("Contour max:")
        self.max_level_label.setFont(label_font)
        self.marker_font_size_label = QLabel("Marker Font Size:")
        self.marker_font_size_label.setFont(label_font)
        self.cursor_pos_label = QLabel("Cursor: (X: -, Y: -)")
        self.cursor_pos_label.setFont(label_font)

        self.clear_all_markers_button = QPushButton("Clear All Markers")
        self.clear_all_markers_button.setFont(label_font)
        self.clear_all_markers_button.setMinimumHeight(30)
        self.clear_all_markers_button.setFixedWidth(300)
        self.clear_all_markers_button.clicked.connect(self.clear_all_markers)
        self.show_diagonal_button = QPushButton("Extract Diagonal")
        self.show_diagonal_button.setFont(label_font)
        self.show_diagonal_button.setMinimumHeight(30)
        self.show_diagonal_button.clicked.connect(self._show_diagonal_plot)

        self.spline_baseline_button = QPushButton("Apply Spline Baseline")
        self.spline_baseline_button.setFont(label_font)
        self.spline_baseline_button.setMinimumHeight(30)
        self.spline_baseline_button.clicked.connect(self._toggle_spline_correction)

        # --- Interpolation Controls (moved from menu bar) ---
        self.interp_method_label = QLabel("Interpolate slices")
        self.interp_method_label.setFont(label_font)
        self.interp_method_combo = QComboBox(self)
        self.interp_method_combo.addItems(["None", "linear", "cubic", "nearest"])
        self.interp_method_combo.setFont(label_font)
        self.interp_method_combo.setMinimumHeight(30)
        self.interp_method_combo.currentIndexChanged.connect(self._apply_interpolation_to_all_plots)

        self.interp_multiplier_label = QLabel("")
        self.interp_multiplier_label.setFont(label_font)
        self.interp_multiplier_combo = QComboBox(self)
        self.interp_multiplier_combo.addItems(["x1", "x2", "x3", "x5"])
        self.interp_multiplier_combo.setFont(label_font)
        self.interp_multiplier_combo.setMinimumHeight(30)
        self.interp_multiplier_combo.currentIndexChanged.connect(self._apply_interpolation_to_all_plots)
        # --- End Interpolation Controls ---


        self.main_plot_tab_layout.addWidget(self.signal_plot_widget, 0, 0, 2, 2)

        x_slice_controls_layout = QHBoxLayout( )
        x_slice_controls_layout.addWidget(self.x_hold_button)
        x_slice_controls_layout.addWidget(self.x_unit_input)
        x_slice_controls_layout.addWidget(self.x_clear_button)
        x_slice_controls_layout.addWidget(QLabel("Fit Type:"))  # Label for combo box
        x_slice_controls_layout.addWidget(self.x_fit_function_selector)  # Add combo box
        x_slice_controls_layout.addWidget(self.x_fit_button)
        x_slice_controls_layout.setSpacing(10)

        x_slice_plot_and_buttons_layout = QVBoxLayout( )
        x_slice_plot_and_buttons_layout.addWidget(self.x_slice_plot_widget)
        x_slice_plot_and_buttons_layout.addLayout(x_slice_controls_layout)

        self.main_plot_tab_layout.addLayout(x_slice_plot_and_buttons_layout, 0, 2, 1, 1)

        y_slice_controls_layout = QHBoxLayout( )
        y_slice_controls_layout.addWidget(self.y_hold_button)
        y_slice_controls_layout.addWidget(self.y_unit_input)
        y_slice_controls_layout.addWidget(self.y_clear_button)
        y_slice_controls_layout.addWidget(QLabel("Fit Type:"))  # Label for combo box
        y_slice_controls_layout.addWidget(self.y_fit_function_selector)  # Add combo box
        y_slice_controls_layout.addWidget(self.y_fit_button)
        y_slice_controls_layout.setSpacing(10)

        y_slice_plot_and_buttons_layout = QVBoxLayout( )
        y_slice_plot_and_buttons_layout.addWidget(self.y_slice_plot_widget)
        y_slice_plot_and_buttons_layout.addLayout(y_slice_controls_layout)

        self.main_plot_tab_layout.addLayout(y_slice_plot_and_buttons_layout, 1, 2, 1, 1)

        slider_controls_layout = QVBoxLayout( )
        slider_controls_layout.addWidget(self.x_label)
        slider_controls_layout.addWidget(self.x_slider)
        slider_controls_layout.addWidget(self.y_label)
        slider_controls_layout.addWidget(self.y_slider)

        level_inputs_layout = QHBoxLayout( )
        level_inputs_layout.addWidget(self.min_level_label)
        level_inputs_layout.addWidget(self.min_level_input)
        level_inputs_layout.addSpacing(20)
        level_inputs_layout.addWidget(self.max_level_label)
        level_inputs_layout.addWidget(self.max_level_input)
        level_inputs_layout.addSpacing(20)
        level_inputs_layout.addWidget(self.marker_font_size_label)
        level_inputs_layout.addWidget(self.marker_font_size_input)
        level_inputs_layout.addStretch(1)
        level_inputs_layout.addWidget(self.spline_baseline_button)
        level_inputs_layout.addWidget(self.show_diagonal_button)
        level_inputs_layout.addWidget(self.clear_all_markers_button)

        # Add interpolation controls to the level_inputs_layout
        level_inputs_layout.addWidget(self.interp_method_label)
        level_inputs_layout.addWidget(self.interp_method_combo)
        level_inputs_layout.addWidget(self.interp_multiplier_label)
        level_inputs_layout.addWidget(self.interp_multiplier_combo)


        slider_controls_layout.addLayout(level_inputs_layout)
        slider_controls_layout.addWidget(self.cursor_pos_label)

        self.main_plot_tab_layout.addLayout(slider_controls_layout, 2, 0, 1, 3)
        self.main_plot_tab_layout.setColumnStretch(0, 2)
        self.main_plot_tab_layout.setColumnStretch(1, 2)
        self.main_plot_tab_layout.setColumnStretch(2, 3)

        self.main_plot_tab_layout.setRowStretch(0, 3)
        self.main_plot_tab_layout.setRowStretch(1, 3)
        self.main_plot_tab_layout.setRowStretch(2, 1)
        self.main_plot_tab_layout.setRowStretch(3, 0)

        self.main_layout.addWidget(self.tab_widget, 0, 0, 1, 1)

        self._update_spline_button_text( )

    def _get_interpolated_1d_data(self, original_x, original_y, method, multiplier):
        """
        Helper function to perform 1D interpolation.
        Returns interpolated x and y data.
        """
        if method == "None" or len(original_x) < 2:
            return np.copy(original_x), np.copy(original_y)

        try:
            target_n_points = int(len(original_x) * multiplier)
            if target_n_points < 2:
                target_n_points = 2

            x_interp = np.linspace(original_x.min(), original_x.max(), target_n_points)
            f_interp = interp1d(original_x, original_y, kind=method, fill_value="extrapolate")
            y_interp = f_interp(x_interp)
            return x_interp, y_interp
        except Exception as e:
            print(f"Error during 1D interpolation ({method}, x{multiplier}): {e}")
            return np.copy(original_x), np.copy(original_y) # Fallback to original on error

    def _apply_interpolation_to_all_plots(self):
        """
        Applies the selected interpolation settings to main window slice plots
        and all open GaussianFitterApp instances.
        """
        self._current_interp_method = self.interp_method_combo.currentText()
        self._current_interp_multiplier = int(self.interp_multiplier_combo.currentText().replace('x', ''))

        # Apply to main window slice plots
        self.update_plots() # This will now call _get_interpolated_1d_data internally

        # Apply to all open GaussianFitterApp instances (tabs)
        print(f"Applying interpolation to fitter tabs: Method={self._current_interp_method}, Multiplier={self._current_interp_multiplier}")
        for tab_info in self.active_fitter_tabs:
            fitter_app_instance = tab_info['widget']
            if isinstance(fitter_app_instance, GaussianFitterApp):
                fitter_app_instance.apply_interpolation_settings(self._current_interp_method, self._current_interp_multiplier)
        self._set_data_modified()

    def close_tab(self, index):
        """
        Handles closing of tabs in the QTabWidget.
        Modified to make 'Main Plots' tab non-closable.
        """
        # Get the text of the tab at the requested index
        tab_name = self.tab_widget.tabText(index)

        # --- START MODIFICATION ---
        # Check if the tab being requested to close is the "Main Plots" tab
        if tab_name == "Main Plots":
            QMessageBox.information(self, "Cannot Close Tab", "The 'Main Plots' tab cannot be closed.")
            return  # Stop the function here, preventing the tab from being removed
        # --- END MODIFICATION ---

        widget = self.tab_widget.widget(index)
        if widget is not None:
            # Disconnect destroyed signal before deleting to prevent double-handling
            if widget == self.original_diagonal_plot_widget_instance:
                try:
                    widget.destroyed.disconnect(self._clear_diagonal_plot_instance_ref)
                except TypeError:
                    pass  # Already disconnected
                self.original_diagonal_plot_widget_instance = None
                self.saved_original_diagonal_plot_state = None
            elif widget == self.spline_corrected_diagonal_plot_widget_instance:
                try:
                    widget.destroyed.disconnect(self._clear_diagonal_plot_instance_ref)
                except TypeError:
                    pass  # Already disconnected
                self.spline_corrected_diagonal_plot_widget_instance = None
                self.saved_spline_corrected_diagonal_plot_state = None

            # Remove from active fitter tabs list
            for i, tab_info in enumerate(list(self.active_fitter_tabs)):
                if tab_info [ 'widget' ] == widget:
                    self.active_fitter_tabs.pop(i)
                    break

            widget.deleteLater( )  # Schedule for deletion
            self._set_data_modified( )  # This function will now only be called if a tab is actually removed

        self.tab_widget.removeTab(index)  # This line will now only execute for closable tabs

        # Optional: Select a different tab if the current one was closed
        if self.tab_widget.count( ) > 0:
            self.tab_widget.setCurrentIndex(max(0, index - 1))

        print(f"Tab '{tab_name}' at index {index} closed.")

    def _apply_legend_font_size(self, legend_item, new_size):
        """
        Applies a new font size to all labels within a given LegendItem.
        """
        if legend_item:
            for sample, label_item in legend_item.items:
                font = label_item.font( )
                font.setPointSize(new_size)
                label_item.setFont(font)

    def _show_linewidth_dialog(self, plot_widget):
        """
        Opens the LineThicknessDialog and applies the new linewidth to all plots
        in the given plot_widget (x_slice or y_slice).
        """
        dialog = LineThicknessDialog(self, initial_thickness = self.current_slice_linewidth)
        if dialog.exec_( ) == QDialog.Accepted:
            new_thickness = dialog.get_thickness( )
            self.current_slice_linewidth = new_thickness
            if plot_widget == self.x_slice_plot_widget:
                current_pen = self.x_slice_curve.opts [ 'pen' ]
                self.x_slice_curve.setPen(
                    pg.mkPen(current_pen.color( ), width = new_thickness, style = current_pen.style( )))
            elif plot_widget == self.y_slice_plot_widget:
                current_pen = self.y_slice_curve.opts [ 'pen' ]
                self.y_slice_curve.setPen(
                    pg.mkPen(current_pen.color( ), width = new_thickness, style = current_pen.style( )))

            for item in plot_widget.listDataItems( ):
                if (plot_widget == self.x_slice_plot_widget and item == self.x_slice_curve) or \
                        (plot_widget == self.y_slice_plot_widget and item == self.y_slice_curve):
                    continue

                if item.opts.get('pen') is not None:
                    old_pen = item.opts [ 'pen' ]
                    new_pen = pg.mkPen(old_pen.color( ), width = new_thickness, style = old_pen.style( ))
                    item.setPen(new_pen)
            self._set_data_modified( )

    def update_plots(self):
        """
        Updates the slice plots and the slice indicator lines on the 2D plot
        based on the current slider positions, and updates contour levels.
        Only updates if data is loaded.
        """
        if not self.data_loaded:
            self.image_item.setImage(np.zeros((2, 2)).T)
            self.image_item.setRect(pg.QtCore.QRectF(0, 0, 1, 1))
            self.signal_plot_widget.setXRange(0, 1, padding = 0)
            self.signal_plot_widget.setYRange(0, 1, padding = 0)

            self.x_slice_curve.setData([ ], [ ])
            self.y_slice_curve.setData([ ], [ ])
            self.x_slice_plot_widget.setXRange(0, 1, padding = 0.05)
            self.x_slice_plot_widget.setYRange(0, 1, padding = 0.05)
            self.y_slice_plot_widget.setXRange(0, 1, padding = 0.05)
            self.y_slice_plot_widget.setYRange(0, 1, padding = 0.05)

            self.x_label.setText(f"X Slice Position: -")
            self.y_label.setText(f"Y Slice Position: -")
            self.cursor_x_line.setVisible(False)
            self.cursor_y_line.setVisible(False)
            self.min_level_input.setText("")
            self.max_level_input.setText("")
            self.marker_font_size_input.setValue(self.marker_font_size)
            self.x_slider.setEnabled(False)
            self.y_slider.setEnabled(False)
            self.min_level_input.setEnabled(False)
            self.max_level_input.setEnabled(False)
            self.marker_font_size_input.setEnabled(False)
            self.x_hold_button.setEnabled(False)
            self.x_unit_input.setEnabled(False)
            self.x_clear_button.setEnabled(False)
            self.x_fit_button.setEnabled(False)
            self.x_fit_function_selector.setEnabled(False)  # Disable fit function selector
            self.y_hold_button.setEnabled(False)
            self.y_unit_input.setEnabled(False)
            self.y_clear_button.setEnabled(False)
            self.y_fit_button.setEnabled(False)
            self.y_fit_function_selector.setEnabled(False)  # Disable fit function selector
            self.clear_all_markers_button.setEnabled(False)
            self.show_diagonal_button.setEnabled(False)
            self.place_marker_action.setEnabled(False)
            self.clear_markers_action.setEnabled(False)
            self.find_peaks_action.setEnabled(False)
            self.clear_detected_peaks_action.setEnabled(False)
            self.toggle_peak_labels_action.setEnabled(False)
            self.spline_baseline_button.setEnabled(False)
            # Disable interpolation controls
            self.interp_method_combo.setEnabled(False)
            self.interp_multiplier_combo.setEnabled(False)
            return

        self.x_slider.setEnabled(True)
        self.y_slider.setEnabled(True)
        self.min_level_input.setEnabled(True)
        self.max_level_input.setEnabled(True)
        self.marker_font_size_input.setEnabled(True)
        self.x_hold_button.setEnabled(True)
        self.x_unit_input.setEnabled(True)
        self.x_clear_button.setEnabled(True)
        self.x_fit_button.setEnabled(True)
        self.x_fit_function_selector.setEnabled(True)  # Enable fit function selector
        self.y_hold_button.setEnabled(True)
        self.y_unit_input.setEnabled(True)
        self.y_clear_button.setEnabled(True)
        self.y_fit_button.setEnabled(True)
        self.y_fit_function_selector.setEnabled(True)  # Enable fit function selector
        self.clear_all_markers_button.setEnabled(True)
        self.show_diagonal_button.setEnabled(True)
        self.place_marker_action.setEnabled(True)
        self.clear_markers_action.setEnabled(True)
        self.find_peaks_action.setEnabled(True)
        self.clear_detected_peaks_action.setEnabled(True)
        self.toggle_peak_labels_action.setEnabled(True)
        self.spline_baseline_button.setEnabled(True)
        # Enable interpolation controls
        self.interp_method_combo.setEnabled(True)
        self.interp_multiplier_combo.setEnabled(True)


        x_idx_interp = self.x_slider.value( )
        y_idx_interp = self.y_slider.value( )

        x_pos_val_interp = self.x_values_interp [ x_idx_interp ]
        y_pos_val_interp = self.y_values_interp [ y_idx_interp ]

        self.x_label.setText(f"X Slice Position: {x_pos_val_interp:.1f}")
        self.y_label.setText(f"Y Slice Position: {y_pos_val_interp:.1f}")

        # Get the original slice data based on the current slider position
        original_y_idx = np.argmin(np.abs(self.current_y_values - y_pos_val_interp))
        original_y_slice_x_data = self.current_x_values
        original_y_slice_y_data = self.current_signal_data[original_y_idx, :]

        original_x_idx = np.argmin(np.abs(self.current_x_values - x_pos_val_interp))
        original_x_slice_x_data = self.current_y_values
        original_x_slice_y_data = self.current_signal_data[:, original_x_idx]

        # Apply interpolation to the current slice data
        interp_x_slice_x, interp_x_slice_y = self._get_interpolated_1d_data(
            original_x_slice_x_data, original_x_slice_y_data,
            self._current_interp_method, self._current_interp_multiplier
        )
        interp_y_slice_x, interp_y_slice_y = self._get_interpolated_1d_data(
            original_y_slice_x_data, original_y_slice_y_data,
            self._current_interp_method, self._current_interp_multiplier
        )

        self.y_slice_curve.setData(interp_y_slice_x, interp_y_slice_y)
        self.x_slice_curve.setData(interp_x_slice_x, interp_x_slice_y)


        vb_y_slice = self.y_slice_plot_widget.getViewBox( )
        x_auto_y_slice, y_auto_y_slice = vb_y_slice.autoRangeEnabled( )
        if x_auto_y_slice:
            self.y_slice_plot_widget.setXRange(interp_y_slice_x.min( ), interp_y_slice_x.max( ),
                                               padding = 0.05)
        if y_auto_y_slice:
            self.y_slice_plot_widget.setYRange(interp_y_slice_y.min( ), interp_y_slice_y.max( ),
                                               padding = 0.05)

        vb_x_slice = self.x_slice_plot_widget.getViewBox( )
        x_auto_x_slice, y_auto_x_slice = vb_x_slice.autoRangeEnabled( )
        if x_auto_x_slice:
            self.x_slice_plot_widget.setXRange(interp_x_slice_x.min( ), interp_x_slice_x.max( ),
                                               padding = 0.05)
        if y_auto_x_slice:
            self.x_slice_plot_widget.setYRange(interp_x_slice_y.min( ), interp_x_slice_y.max( ),
                                               padding = 0.05)

        try:
            min_level = float(self.min_level_input.text( ))
        except ValueError:
            min_level = self.signal_data_interp.min( )

        try:
            max_level = float(self.max_level_input.text( ))
        except ValueError:
            max_level = self.signal_data_interp.max( )

        if min_level > max_level:
            min_level, max_level = max_level, min_level

        self.image_item.setLevels((min_level, max_level))

        self.cursor_x_line.setPos(x_pos_val_interp)
        self.cursor_y_line.setPos(y_pos_val_interp)
        self.cursor_x_line.setVisible(True)
        self.cursor_y_line.setVisible(True)


    def update_signal_cursor_pos(self, evt):
        """
        Updates the cursor lines and the cursor position label for the 2D signal plot.
        """
        if not self.data_loaded:
            self.cursor_pos_label.setText("Cursor: (X: -, Y: -)")
            return

        pos = evt
        if self.signal_plot_widget.sceneBoundingRect( ).contains(pos):
            mousePoint = self.signal_plot_widget.plotItem.vb.mapSceneToView(pos)
            x_val_display = mousePoint.x( )
            y_val_display = mousePoint.y( )

            self.cursor_x_line.setVisible(True)
            self.cursor_y_line.setVisible(True)

            self.cursor_x_line.setPos(x_val_display)
            self.cursor_y_line.setPos(y_val_display)

            self.cursor_pos_label.setText(f"Cursor: (X: {x_val_display:.2f}, Y: {y_val_display:.2f})")
        else:
            self.cursor_x_line.setVisible(False)
            self.cursor_y_line.setVisible(False)
            self.cursor_pos_label.setText("Cursor: (X: -, Y: -)")

    def on_signal_plot_clicked(self, event):
        """
        Handles mouse clicks on the 2D signal plot.
        If right-clicked, stores the position for context menu action.
        """
        if not self.data_loaded: return
        if event.button( ) == Qt.RightButton:
            mousePoint = self.signal_plot_widget.plotItem.vb.mapSceneToView(event.scenePos( ))
            self.last_right_click_data_pos = mousePoint

    def _place_marker_from_context_menu(self):
        """
        Places a marker at the last stored right-click position.
        Called when the "Place Marker" context menu action is triggered.
        """
        if not self.data_loaded: return
        if self.last_right_click_data_pos:
            x_coord = self.last_right_click_data_pos.x( )
            y_coord = self.last_right_click_data_pos.y( )

            scatter_item = pg.PlotDataItem([ x_coord ], [ y_coord ], symbol = 'x', size = 10, pen = 'magenta',
                                           brush = 'magenta')
            self.signal_plot_widget.addItem(scatter_item)

            marker_text = f"({x_coord:.1f}, {y_coord:.1f})"
            text_item = ClickableTextItem(text = marker_text, color = 'magenta', anchor = (1, 1),
                                          parent_app = self, associated_scatter = scatter_item, is_peak_label = False)
            text_item.setPos(x_coord, y_coord)

            font = QFont("Times New Roman", self.marker_font_size)
            text_item.setFont(font)

            self.signal_plot_widget.addItem(text_item)
            self.placed_markers.append((scatter_item, text_item))
            self.last_right_click_data_pos = None
            self._set_data_modified( )

    def remove_plot_item_pair(self, scatter_item_to_remove, text_item_to_remove):
        """
        Removes a specific scatter and text item pair from the 2D plot and
        from the tracking lists (placed_markers or detected_peaks_items).
        This method is called by the ClickableTextItem's context menu.
        """
        removed = False
        if scatter_item_to_remove:
            self.signal_plot_widget.removeItem(scatter_item_to_remove)
            removed = True
        if text_item_to_remove:
            self.signal_plot_widget.removeItem(text_item_to_remove)
            removed = True

        for i, (scatter, text) in enumerate(list(self.placed_markers)):
            if scatter == scatter_item_to_remove and text == text_item_to_remove:
                self.placed_markers.pop(i)
                break

        for i, (scatter, text) in enumerate(list(self.detected_peaks_items)):
            if scatter == scatter_item_to_remove and text == text_item_to_remove:
                self.detected_peaks_items.pop(i)
                break

        if removed:
            self._set_data_modified( )

    def clear_all_markers(self):
        """
        Removes all placed markers (text and scatter items) from the 2D plot.
        """
        if self.placed_markers:
            for item_tuple in self.placed_markers:
                scatter_item, text_item = item_tuple
                if scatter_item:
                    self.signal_plot_widget.removeItem(scatter_item)
                if text_item:
                    self.signal_plot_widget.removeItem(text_item)
            self.placed_markers.clear( )
            self._set_data_modified( )

    def update_x_slice_cursor_pos(self, evt):
        """
        Updates the cursor position label for the X-axis slice plot.
        Note: X-slice plot's x-axis is current Y-axis values.
        """
        if not self.data_loaded: return
        pos = evt
        if self.x_slice_plot_widget.sceneBoundingRect( ).contains(pos):
            mousePoint = self.x_slice_plot_widget.plotItem.vb.mapSceneToView(pos)
            y_val_current = mousePoint.x( )
            amp_val = mousePoint.y( )
            self.cursor_pos_label.setText(f"X-Slice Cursor: ({y_val_current:.2f},{amp_val:.2g})")
        else:
            self.cursor_pos_label.setText("Cursor: (X: -, Y: -)")

    def update_y_slice_cursor_pos(self, evt):
        """
        Updates the cursor position label for the Y-axis slice plot.
        Note: Y-slice plot's x-axis is current X-axis values.
        """
        if not self.data_loaded: return
        pos = evt
        if self.y_slice_plot_widget.sceneBoundingRect( ).contains(pos):
            mousePoint = self.y_slice_plot_widget.plotItem.vb.mapSceneToView(pos)
            x_val_current = mousePoint.x( )
            amp_val = mousePoint.y( )
            self.cursor_pos_label.setText(f"Y-Slice Cursor: ({x_val_current:.2f}, {amp_val:.2g})")
        else:
            self.cursor_pos_label.setText("Cursor: (X: -, Y: -)")

    def _format_unit_for_display(self, unit_string):
        """
        Converts common plain-text unit notations (e.g., cm^-1) to Unicode
        superscripts/subscripts for better display in plot legends.
        """
        unit_string = unit_string.replace("^-1", "\u207B\u00B9")
        unit_string = unit_string.replace("^-2", "\u207B\u00B2")
        unit_string = unit_string.replace("^2", "\u00B2")
        unit_string = unit_string.replace("^3", "\u00B3")
        unit_string = unit_string.replace("^-3", "\u207B\u00B3")
        unit_string = unit_string.replace("^-4", "\u207B\u2074")
        unit_string = unit_string.replace("_1", "\u2081")
        unit_string = unit_string.replace("_2", "\u2082")
        unit_string = unit_string.replace("_3", "\u2083")
        return unit_string

    def _strip_html_tags(self, text):
        """
        Removes HTML-like tags (like <sup>, <sub>) and bracketed content from a string.
        """
        # Remove <sup> and <sub> tags and their content
        text = re.sub(r'<sup[^>]*>.*?</sup>', '', text)
        text = re.sub(r'<sub[^>]*>.*?</sub>', '', text)
        # Remove any remaining angle brackets (e.g., from [cm^-1] if not handled by regex)
        text = text.replace('<sup>', '').replace('</sup>', '')
        text = text.replace('<sub>', '').replace('</sub>', '')
        # Remove the brackets and their content
        text = re.sub(r'\[.*?\]', '', text).strip( )
        return text

    def hold_x_slice_plot(self):
        """
        Takes the current X-axis slice data and adds it as a new, persistent curve
        to the X-axis slice plot.
        """
        if not self.data_loaded: return
        x_idx_interp = self.x_slider.value( )
        x_pos_val = self.x_values_interp [ x_idx_interp ]

        self.held_x_slices_count += 1
        x_data = self.x_slice_curve.getData( ) [ 0 ]
        y_data = self.x_slice_curve.getData( ) [ 1 ]
        color_index = (self.held_x_slices_count - 1) % len(self.plot_colors)
        color = self.plot_colors [ color_index ]
        pen = pg.mkPen(color, width = self.current_slice_linewidth, style = Qt.SolidLine)

        unit_text = self.x_unit_input.text( ).strip( )
        formatted_unit_text = self._format_unit_for_display(unit_text)

        if formatted_unit_text:
            name = f'{x_pos_val:.0f} {formatted_unit_text}'
        else:
            name = f'{x_pos_val:.0f}'

        self.x_slice_plot_widget.plot(x_data, y_data, pen = pen, name = name)
        self._set_data_modified( )

    def clear_x_slice_plots(self):
        """
        Clears all held X-axis slice plots, resetting the plot to only show the live slice.
        """
        items_to_remove = [ item for item in self.x_slice_plot_widget.listDataItems( ) if item != self.x_slice_curve ]
        if items_to_remove:
            for item in items_to_remove:
                self.x_slice_plot_widget.removeItem(item)
            self.held_x_slices_count = 0
            self._set_data_modified( )

    def hold_y_slice_plot(self):
        """
        Takes the current Y-axis slice data and adds it as a new, persistent curve
        to the Y-axis slice plot.
        """
        if not self.data_loaded: return
        y_idx_interp = self.y_slider.value( )
        y_pos_val = self.y_values_interp [ y_idx_interp ]

        self.held_y_slices_count += 1
        x_data = self.y_slice_curve.getData( ) [ 0 ]
        y_data = self.y_slice_curve.getData( ) [ 1 ]

        color_index = (self.held_y_slices_count - 1) % len(self.plot_colors)
        color = self.plot_colors [ color_index ]
        pen = pg.mkPen(color, width = self.current_slice_linewidth, style = Qt.SolidLine)

        unit_text = self.y_unit_input.text( ).strip( )
        formatted_unit_text = self._format_unit_for_display(unit_text)

        if formatted_unit_text:
            name = f'{y_pos_val:.0f} {formatted_unit_text}'
        else:
            name = f'{y_pos_val:.0f}'

        self.y_slice_plot_widget.plot(x_data, y_data, pen = pen, name = name)
        self._set_data_modified( )

    def clear_y_slice_plots(self):
        """
        Clears all held Y-axis slice plots, resetting the plot to only show the live slice.
        """
        items_to_remove = [ item for item in self.y_slice_plot_widget.listDataItems( ) if item != self.y_slice_curve ]
        if items_to_remove:
            for item in items_to_remove:
                self.y_slice_plot_widget.removeItem(item)
            self.held_y_slices_count = 0
            self._set_data_modified( )

    def spline_baseline_correction(self, data, probe_wn):
        """
        Applies spline baseline correction to 2D data along the probe (x) dimension.
        data: 2D numpy array (pump x probe)
        probe_wn: 1D numpy array of probe wavenumbers (x-axis values)
        Uses default s (smoothing factor) for UnivariateSpline.
        """
        baseline = np.zeros_like(data, dtype = float)
        for i in range(data.shape [ 0 ]):
            if len(probe_wn) >= 2:
                try:
                    valid_mask = np.isfinite(data [ i, : ])
                    if np.sum(valid_mask) >= 2:
                        # Use default s value
                        spline = UnivariateSpline(probe_wn [ valid_mask ], data [ i, valid_mask ])
                        baseline [ i, : ] = spline(probe_wn)
                    else:
                        print(f"Not enough valid data points for spline in row {i}. Skipping spline for this row.")
                        baseline [ i, : ] = 0.0
                except Exception as e:
                    print(f"Error during spline calculation for row {i}: {e}")
                    baseline [ i, : ] = 0.0
            else:
                print(f"Not enough data points for spline in row {i}. Skipping spline for this row.")
                baseline [ i, : ] = 0.0

        corrected_data = data - baseline
        return corrected_data

    def _toggle_spline_correction(self):
        """
        Toggles the application/removal of spline baseline correction.
        """
        if not self.data_loaded:
            QMessageBox.warning(self, "No Data", "Please import data before applying spline baseline.")
            return

        # Safely get current contour levels
        current_min_level = float(self.min_level_input.text( )) if self.min_level_input.text( ) else None
        current_max_level = float(self.max_level_input.text( )) if self.max_level_input.text( ) else None

        # Safely get current plot ranges
        current_signal_xlim = None
        current_signal_ylim = None
        if self.signal_plot_widget.plotItem.vb:
            signal_view_range = self.signal_plot_widget.plotItem.vb.viewRange( )
            if signal_view_range and len(signal_view_range) == 2:
                current_signal_xlim = signal_view_range [ 0 ]
                current_signal_ylim = signal_view_range [ 1 ]

        current_x_slice_xlim = None
        current_x_slice_ylim = None
        if self.x_slice_plot_widget.plotItem.vb:
            x_slice_view_range = self.x_slice_plot_widget.plotItem.vb.viewRange( )
            if x_slice_view_range and len(x_slice_view_range) == 2:
                current_x_slice_xlim = x_slice_view_range [ 0 ]
                current_x_slice_ylim = x_slice_view_range [ 1 ]

        current_y_slice_xlim = None
        current_y_slice_ylim = None
        if self.y_slice_plot_widget.plotItem.vb:
            y_slice_view_range = self.y_slice_plot_widget.plotItem.vb.viewRange( )
            if y_slice_view_range and len(y_slice_view_range) == 2:
                current_y_slice_xlim = y_slice_view_range [ 0 ]
                current_y_slice_ylim = y_slice_view_range [ 1 ]

        if not self.is_spline_corrected:
            try:
                # Call without s_value
                corrected_data = self.spline_baseline_correction(
                    self._initial_raw_signal_data, self._initial_raw_x_values
                )
                self.current_signal_data = corrected_data
                self.is_spline_corrected = True
                self._set_data_modified( )
                self._refresh_all_plots(preserve_contour_levels = True,
                                        min_level = current_min_level,
                                        max_level = current_max_level,
                                        preserve_plot_ranges = True,
                                        signal_xlim = current_signal_xlim,
                                        signal_ylim = current_signal_ylim,
                                        x_slice_xlim = current_x_slice_xlim,
                                        x_slice_ylim = current_x_slice_ylim,
                                        y_slice_xlim = current_y_slice_xlim,
                                        y_slice_ylim = current_y_slice_ylim)
            except Exception as e:
                print(f"Failed to apply spline baseline: {e}")  # Print to console instead of QMessageBox
                self.current_signal_data = self._initial_raw_signal_data.copy( )  # Revert on error
                self.is_spline_corrected = False  # Ensure state is correct
                self._refresh_all_plots( )  # Refresh to original state
        else:
            self.current_signal_data = self._initial_raw_signal_data.copy( )
            self.is_spline_corrected = False
            self._set_data_modified( )
            self._refresh_all_plots(preserve_contour_levels = True,
                                    min_level = current_min_level,
                                    max_level = current_max_level,
                                    preserve_plot_ranges = True,
                                    signal_xlim = current_signal_xlim,
                                    signal_ylim = current_signal_ylim,
                                    x_slice_xlim = current_x_slice_xlim,
                                    x_slice_ylim = current_x_slice_ylim,
                                    y_slice_xlim = current_y_slice_xlim,
                                    y_slice_ylim = current_y_slice_ylim)

        self._update_spline_button_text( )

    def _update_spline_button_text(self):
        """
        Updates the text of the spline baseline button based on the current state.
        """
        if self.is_spline_corrected:
            self.spline_baseline_button.setText("Revert to Original")
        else:
            self.spline_baseline_button.setText("Use Spline Baseline")

    def find_peaks_second_derivative(self, z, neighborhood_size=4, positive_threshold=0.0001,
                                     negative_threshold=-0.0001, smooth_sigma=0):
        """
        Find peaks and valleys using the second derivative (Laplacian).
        """
        if not self.data_loaded: return [ ], [ ]
        if smooth_sigma > 0:
            z = gaussian_filter(z, sigma = smooth_sigma)

        grad_x = np.gradient(z, axis = 1)
        grad_y = np.gradient(z, axis = 0)
        grad_xx = np.gradient(grad_x, axis = 1)
        grad_yy = np.gradient(grad_y, axis = 0)
        laplacian = grad_xx + grad_yy

        positive_peaks_mask = (laplacian < negative_threshold)

        negative_peaks_mask = (laplacian > positive_threshold)

        local_max = (maximum_filter(z, size = neighborhood_size) == z)
        positive_peaks_mask &= local_max

        local_min = (minimum_filter(z, size = neighborhood_size) == z)
        negative_peaks_mask &= local_min

        positive_peaks_indices = np.where(positive_peaks_mask)
        negative_peaks_indices = np.where(negative_peaks_mask)

        return positive_peaks_indices, negative_peaks_indices

    def _show_peak_finder_dialog(self):
        """
        Opens the PeakFinderDialog to get parameters and then finds/displays peaks.
        """
        if not self.data_loaded:
            QMessageBox.warning(self, "No Data", "Please import data before finding peaks.")
            return
        dialog = PeakFinderDialog(self, initial_params = self.peak_finder_params)
        if dialog.exec_( ) == QDialog.Accepted:
            params = dialog.get_parameters( )
            if params:
                self.peak_finder_params = params
                self.find_and_display_peaks(
                    neighborhood_size = params [ 'neighborhood_size' ],
                    positive_threshold = params [ 'positive_threshold' ],
                    negative_threshold = params [ 'negative_threshold' ],
                    smooth_sigma = params [ 'smooth_sigma' ]
                )
                self._set_data_modified( )

    def find_and_display_peaks(self, neighborhood_size, positive_threshold, negative_threshold, smooth_sigma):
        """
        Finds peaks and displays them on the 2D plot using the given parameters.
        """
        if not self.data_loaded: return
        self.clear_detected_peaks( )

        positive_peaks_indices, negative_peaks_indices = self.find_peaks_second_derivative(
            self.signal_data_interp,
            neighborhood_size = neighborhood_size,
            positive_threshold = positive_threshold,
            negative_threshold = negative_threshold,
            smooth_sigma = smooth_sigma
        )

        peaks_found = False
        if len(positive_peaks_indices [ 0 ]) > 0 or len(negative_peaks_indices [ 0 ]) > 0:
            peaks_found = True

        for i in range(len(positive_peaks_indices [ 0 ])):
            row_idx, col_idx = positive_peaks_indices [ 0 ] [ i ], positive_peaks_indices [ 1 ] [ i ]
            x, y = self.x_values_interp [ col_idx ], self.y_values_interp [ row_idx ]

            scatter_item = pg.PlotDataItem(
                x = [ x ], y = [ y ],
                symbol = 'x', size = 10, pen = pg.mkPen('green', width = 2), brush = None
            )
            self.signal_plot_widget.addItem(scatter_item)

            text = f"({x:.1f}, {y:.1f})"
            text_item = ClickableTextItem(text = text, color = 'green', anchor = (0.5, 0),
                                          parent_app = self, associated_scatter = scatter_item, is_peak_label = True)
            text_item.setPos(x, y)
            text_item.setFont(QFont("Times New Roman", self.marker_font_size))
            text_item.setVisible(self.show_peak_labels)
            self.signal_plot_widget.addItem(text_item)

            self.detected_peaks_items.append((scatter_item, text_item))

        for i in range(len(negative_peaks_indices [ 0 ])):
            row_idx, col_idx = negative_peaks_indices [ 0 ] [ i ], negative_peaks_indices [ 1 ] [ i ]
            x, y = self.x_values_interp [ col_idx ], self.y_values_interp [ row_idx ]

            scatter_item = pg.PlotDataItem(
                x = [ x ], y = [ y ],
                symbol = 'o', size = 10, pen = pg.mkPen('orange', width = 2), brush = None
            )
            self.signal_plot_widget.addItem(scatter_item)

            text = f"({x:.1f}, {y:.1f})"
            text_item = ClickableTextItem(text = text, color = 'orange', anchor = (0.5, 1),
                                          parent_app = self, associated_scatter = scatter_item,
                                          is_peak_label = True)
            text_item.setPos(x, y)
            text_item.setFont(QFont("Times New Roman", self.marker_font_size))
            text_item.setVisible(self.show_peak_labels)
            self.signal_plot_widget.addItem(text_item)

            self.detected_peaks_items.append((scatter_item, text_item))

        if not peaks_found:
            QMessageBox.information(self, "No Peaks Found", "No peaks or valleys found with the current parameters.")
        else:
            self._set_data_modified( )

    def clear_detected_peaks(self):
        """
        Clears all detected peak markers and their labels from the 2D plot.
        """
        if self.detected_peaks_items:
            for item_tuple in self.detected_peaks_items:
                scatter_item, text_item = item_tuple
                if scatter_item:
                    self.signal_plot_widget.removeItem(scatter_item)
                if text_item:
                    self.signal_plot_widget.removeItem(text_item)
            self.detected_peaks_items.clear( )
            self._set_data_modified( )

    def _toggle_peak_labels_visibility(self):
        """
        Toggles the visibility of all detected peak labels.
        """
        self.show_peak_labels = not self.show_peak_labels
        for item_tuple in self.detected_peaks_items:
            _, text_item = item_tuple
            if text_item:
                text_item.setVisible(self.show_peak_labels)

        if self.show_peak_labels:
            self.toggle_peak_labels_action.setText("Hide Peak Labels")
        else:
            self.toggle_peak_labels_action.setText("Show Peak Labels")

    def _show_edit_names_dialog(self):
        """
        Opens the EditNamesDialog to allow the user to edit axis labels.
        """
        dialog = EditNamesDialog(self, current_labels = self.axis_labels)
        if dialog.exec_( ) == QDialog.Accepted:
            new_labels = dialog.get_names( )
            if any(self.axis_labels [ key ] != new_labels [ key ] for key in self.axis_labels):
                self.axis_labels.update(new_labels)
                self._apply_axis_labels( )
                self._set_data_modified( )

    def _apply_axis_labels(self):
        """
        Applies the currently stored axis labels to all plots.
        This method is called after data load and after editing names.
        """
        self.signal_plot_widget.setLabel('bottom', self.axis_labels [ 'signal_bottom' ],
                                         **{ 'font-size': f'{self.axis_label_font_size}pt' })
        self.signal_plot_widget.setLabel('left', self.axis_labels [ 'signal_left' ],
                                         **{ 'font-size': f'{self.axis_label_font_size}pt' })

        self.x_slice_plot_widget.setLabel('bottom', self.axis_labels [ 'x_slice_bottom' ],
                                          **{ 'font-size': f'{self.axis_label_font_size}pt' })
        self.x_slice_plot_widget.setLabel('left', self.axis_labels [ 'x_slice_left' ],
                                          **{ 'font-size': f'{self.axis_label_font_size}pt' })

        self.y_slice_plot_widget.setLabel('bottom', self.axis_labels [ 'y_slice_bottom' ],
                                          **{ 'font-size': f'{self.axis_label_font_size}pt' })
        self.y_slice_plot_widget.setLabel('left', self.axis_labels [ 'y_slice_left' ],
                                          **{ 'font-size': f'{self.axis_label_font_size}pt' })

    def on_import_data_action_triggered(self):
        """
        Slot for the "Import Data..." menu action.
        Opens a file dialog, reads the selected data, and updates the plots.
        """
        if self._data_modified:
            reply = QMessageBox.question(self, 'Save Changes',
                                         "You have unsaved changes. Do you want to save them before importing new data?",
                                         QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                                         QMessageBox.Save)
            if reply == QMessageBox.Save:
                save_successful = self._save_project( )
                if not save_successful:
                    return
            elif reply == QMessageBox.Cancel:
                return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            "",
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                df = pd.read_csv(file_path, header = None)

                if df.shape [ 0 ] < 2 or df.shape [ 1 ] < 2:
                    raise ValueError("Data file must have at least 2 rows and 2 columns for X, Y, Z extraction.")

                y_values = df.iloc [ 1:, 0 ].values.astype(float)
                x_values = df.iloc [ 0, 1: ].values.astype(float)
                z_data = df.iloc [ 1:, 1: ].values.astype(float)

                self._load_data_into_plots(x_values, y_values, z_data)

                QMessageBox.information(self, "File Processed",
                                        f"File '{file_path}' loaded and data parsed successfully.\n"
                                        f"X-values shape: {x_values.shape}\n"
                                        f"Y-values shape: {y_values.shape}\n"
                                        f"Z-data shape: {z_data.shape}")
                print(f"Data file selected and parsed: {file_path}")
                self._current_project_file = None
                self._data_modified = True
                self._update_window_title( )

            except ValueError as ve:
                QMessageBox.critical(self, "Data Error", f"Error in data structure: {ve}")
                print(f"Error in data structure for {file_path}: {ve}")
                self.data_loaded = False
                self.update_plots( )
                self._data_modified = False
                self._update_window_title( )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to process file: {e}")
                print(f"Error processing file {file_path}: {e}")
                self.data_loaded = False
                self.update_plots( )
                self._data_modified = False
                self._update_window_title( )
        else:
            QMessageBox.information(self, "Action", "No data file selected.")
            print("No data file selected.")

    def _show_diagonal_plot(self):
        """
        Opens a dialog to get parameters for the diagonal plot and then displays it in new tabs.
        It will attempt to open both original and spline-corrected diagonal plots.
        """
        if not self.data_loaded:
            QMessageBox.warning(self, "No Data", "Please import data before showing diagonal plot.")
            return

        param_dialog = PlotParametersDialog(self)
        if param_dialog.exec_( ) == QDialog.Accepted:
            params = param_dialog.result_params
            if params:
                # --- Process Original Data Diagonal Plot ---
                original_data_processor = Plotting_contour_and_interpoaltion(
                    self._initial_raw_x_values, self._initial_raw_y_values, self._initial_raw_signal_data
                )
                original_data_processor.choose_region(params [ 'start_wn' ], params [ 'stop_wn' ],
                                                      params [ 'shift_index' ])
                original_data_processor.interpolate(params [ 'npoints' ], params [ 'interp_method' ])

                original_plot_state = {
                    'probe': original_data_processor.probe.tolist( ),
                    'pump': original_data_processor.pump.tolist( ),
                    'data': original_data_processor.data.tolist( ),
                    'params': params,
                    'min_level': float(self.min_level_input.text( )) if self.min_level_input.text( ) else None,
                    'max_level': float(self.max_level_input.text( )) if self.max_level_input.text( ) else None,
                    'diagonal_slice_linewidth': 2,
                    'dashed_diagonal_line_linewidth': 2,
                    'dashed_diagonal_line_color': 'k',
                    'contour_xlabel': 'Probe wavenumber [cm<sup>-1</sup>]',
                    'contour_ylabel': 'Pump wavenumber [cm<sup>-1</sup>]',
                    'diagonal_xlabel': 'Probe wavenumber [cm<sup>-1</sup>]',
                    'diagonal_ylabel': 'ΔOD',
                    'is_spline_corrected': False  # Explicitly mark as original
                }
                self.saved_original_diagonal_plot_state = original_plot_state
                self._open_diagonal_plot_tab(original_plot_state, False)

                # --- Process Spline Corrected Data Diagonal Plot ---
                # Ensure spline corrected data is available for this calculation
                spline_corrected_raw_data = self.spline_baseline_correction(
                    self._initial_raw_signal_data, self._initial_raw_x_values
                )
                spline_data_processor = Plotting_contour_and_interpoaltion(
                    self._initial_raw_x_values, self._initial_raw_y_values, spline_corrected_raw_data
                )
                spline_data_processor.choose_region(params [ 'start_wn' ], params [ 'stop_wn' ],
                                                    params [ 'shift_index' ])
                spline_data_processor.interpolate(params [ 'npoints' ], params [ 'interp_method' ])

                spline_corrected_plot_state = {
                    'probe': spline_data_processor.probe.tolist( ),
                    'pump': spline_data_processor.pump.tolist( ),
                    'data': spline_data_processor.data.tolist( ),
                    'params': params,
                    'min_level': float(self.min_level_input.text( )) if self.min_level_input.text( ) else None,
                    'max_level': float(self.max_level_input.text( )) if self.max_level_input.text( ) else None,
                    'diagonal_slice_linewidth': 2,
                    'dashed_diagonal_line_linewidth': 2,
                    'dashed_diagonal_line_color': 'k',
                    'contour_xlabel': 'Probe wavenumber [cm<sup>-1</sup>]',
                    'contour_ylabel': 'Pump wavenumber [cm<sup>-1</sup>]',
                    'diagonal_xlabel': 'Probe wavenumber [cm<sup>-1</sup>]',
                    'diagonal_ylabel': 'ΔOD',
                    'is_spline_corrected': True  # Explicitly mark as spline corrected
                }
                self.saved_spline_corrected_diagonal_plot_state = spline_corrected_plot_state
                self._open_diagonal_plot_tab(spline_corrected_plot_state, True)

                self._set_data_modified( )
            else:
                QMessageBox.critical(self, "Error", "Failed to retrieve plot parameters for diagonal plot.")
        else:
            QMessageBox.information(self, "Cancelled", "Diagonal plot parameters selection cancelled.")

    def _open_fitter_tab(self, plot_data_item, is_x_slice, fitting_function_type, xlabel, ylabel, slice_axis_name,
                         slice_value, slice_unit, is_spline_corrected):  # ADDED is_spline_corrected here
        """
        Calls signal_fitter_wrapper, adds the returned widget as a new tab, and manages its lifecycle.
        """
        fitter_widget = signal_fitter_wrapper(self, plot_data_item, is_x_slice, fitting_function_type, xlabel, ylabel,
                                              slice_axis_name, slice_value, slice_unit, is_spline_corrected)
        if fitter_widget:
            # Add the widget as a new tab
            tab_index = self.tab_widget.addTab(fitter_widget, fitter_widget.objectName( ))
            self.tab_widget.setCurrentIndex(tab_index)

            # Store a reference to the widget and its tab index
            self.active_fitter_tabs.append({ 'widget': fitter_widget, 'tab_index': tab_index })

            # Connect the widget's destroyed signal to remove the tab
            fitter_widget.destroyed.connect(
                lambda: self._remove_fitter_tab(fitter_widget)
            )
            self._set_data_modified( )  # Mark project as modified

    def _remove_fitter_tab(self, fitter_widget_to_remove):
        """
        Removes a fitter tab from the QTabWidget and cleans up its reference.
        """
        for i, tab_info in enumerate(list(self.active_fitter_tabs)):
            if tab_info [ 'widget' ] == fitter_widget_to_remove:
                self.tab_widget.removeTab(tab_info [ 'tab_index' ])
                self.active_fitter_tabs.pop(i)
                self._set_data_modified( )  # Mark project as modified
                break
        # Re-index remaining tabs if necessary (QTabWidget handles this internally, but good to be aware)
        # We don't need to manually re-index tab_index in active_fitter_tabs as we remove by widget reference.

    def _clear_diagonal_plot_instance_ref(self, destroyed_widget):
        """Slot to clear the reference when a diagonal plot window is destroyed."""
        if destroyed_widget == self.original_diagonal_plot_widget_instance:
            self.original_diagonal_plot_widget_instance = None
            self.saved_original_diagonal_plot_state = None
            print("Original diagonal plot instance reference and saved state cleared by destroyed signal.")
        elif destroyed_widget == self.spline_corrected_diagonal_plot_widget_instance:
            self.spline_corrected_diagonal_plot_widget_instance = None
            self.saved_spline_corrected_diagonal_plot_state = None
            print("Spline corrected diagonal plot instance reference and saved state cleared by destroyed signal.")

    def _open_diagonal_plot_tab(self, plot_state, is_spline_corrected_for_tab):
        """
        Opens or brings to front the diagonal plot tab with the given state.
        Ensures any old instance is properly deleted and its reference cleared.
        """
        target_instance = None
        if is_spline_corrected_for_tab:
            target_instance = self.spline_corrected_diagonal_plot_widget_instance
        else:
            target_instance = self.original_diagonal_plot_widget_instance

        is_instance_valid = False
        if target_instance is not None:
            try:
                _ = target_instance.objectName( )
                is_instance_valid = True
            except RuntimeError:
                if is_spline_corrected_for_tab:
                    self.spline_corrected_diagonal_plot_widget_instance = None
                else:
                    self.original_diagonal_plot_widget_instance = None
                is_instance_valid = False

        if not is_instance_valid:
            new_instance = PlotWindow(
                np.array(plot_state [ 'probe' ]),
                np.array(plot_state [ 'pump' ]),
                np.array(plot_state [ 'data' ]),
                initial_min_level = plot_state.get('min_level'),
                initial_max_level = plot_state.get('max_level'),
                initial_diagonal_slice_linewidth = plot_state.get('diagonal_slice_linewidth', 2),
                initial_dashed_diagonal_line_linewidth = plot_state.get('dashed_diagonal_line_linewidth', 2),
                initial_dashed_diagonal_line_color = plot_state.get('dashed_diagonal_line_color', 'k'),
                initial_contour_xlabel = plot_state.get('contour_xlabel', 'Probe wavenumber [cm<sup>-1</sup>]'),
                initial_contour_ylabel = plot_state.get('contour_ylabel', 'Pump wavenumber [cm<sup>-1</sup>]'),
                initial_diagonal_xlabel = plot_state.get('diagonal_xlabel', 'Probe wavenumber [ cm<sup>-1</sup>]'),
                initial_diagonal_ylabel = plot_state.get('diagonal_ylabel', 'ΔOD'),
                parent = self.tab_widget,
                is_spline_corrected = is_spline_corrected_for_tab
            )
            self.tab_widget.addTab(new_instance, new_instance.objectName( ))

            if is_spline_corrected_for_tab:
                self.spline_corrected_diagonal_plot_widget_instance = new_instance
            else:
                self.original_diagonal_plot_widget_instance = new_instance

            new_instance.destroyed.connect(self._clear_diagonal_plot_instance_ref)
            target_instance = new_instance

        # Ensure the diagonal tab is visible and active
        idx = self.tab_widget.indexOf(target_instance)
        if idx != -1:
            self.tab_widget.setCurrentWidget(target_instance)

    def _save_project_data(self, file_path):
        """
        Helper method to encapsulate the actual saving logic.
        This method performs the saving to the given file_path.
        Returns True on success, False on failure.
        """
        project_state = {
            'data_loaded': self.data_loaded,
            'axis_labels': self.axis_labels,
            'peak_finder_params': self.peak_finder_params,
            'x_legend_font_size': self.x_legend_font_size,
            'y_legend_font_size': self.y_legend_font_size,
            'show_peak_labels': self.show_peak_labels,
            'current_slice_linewidth': self.current_slice_linewidth,
            'x_slice_legend_unit': self.x_unit_input.text( ),
            'y_slice_legend_unit': self.y_unit_input.text( ),
            'marker_font_size': self.marker_font_size,
            'is_spline_corrected': self.is_spline_corrected,
            'current_interp_method': self._current_interp_method, # Save interpolation settings
            'current_interp_multiplier': self._current_interp_multiplier, # Save interpolation settings
        }

        if self.data_loaded:
            project_state.update({
                'initial_raw_x_values': self._initial_raw_x_values.tolist( ),
                'initial_raw_y_values': self._initial_raw_y_values.tolist( ),
                'initial_raw_signal_data': self._initial_raw_signal_data.tolist( ),
                'x_slider_value': self.x_slider.value( ),
                'y_slider_value': self.y_slider.value( ),
                'min_level_input': self.min_level_input.text( ),
                'max_level_input': self.max_level_input.text( )
            })

        def save_marker(scatter_item, text_item):
            font_size = self.marker_font_size
            if hasattr(text_item, '_font') and text_item._font is not None:
                font_size = text_item._font.pointSize( )
            symbol = scatter_item.opts.get('symbol', 'o')
            size = scatter_item.opts.get('size', 10)
            pen_color_rgb = (0.0, 0.0, 0.0, 1.0)
            if scatter_item.opts.get('pen') is not None:
                pen_color_rgb = scatter_item.opts [ 'pen' ].color( ).getRgbF( )

            brush_color_rgb = (0.0, 0.0, 0.0, 0.0)
            if scatter_item.opts.get('brush') is not None:
                brush_color_rgb = scatter_item.opts [ 'brush' ].color( ).getRgbF( )

            return {
                'x': scatter_item.getData( ) [ 0 ] [ 0 ],
                'y': scatter_item.getData( ) [ 1 ] [ 0 ],
                'text': text_item.text( ),
                'color': text_item.get_rgb_f_color( ),
                'font_size': font_size,
                'anchor': (text_item.anchor [ 0 ], text_item.anchor [ 1 ]),
                'is_peak_label': getattr(text_item, 'is_peak_label', False),
                'symbol': symbol,
                'size': size,
                'pen_color': pen_color_rgb,
                'brush_color': brush_color_rgb
            }

        project_state [ 'placed_markers' ] = [ save_marker(*m) for m in self.placed_markers ]
        project_state [ 'detected_peaks_items' ] = [ save_marker(*m) for m in self.detected_peaks_items ]

        def save_plot_item(item):
            x, y = item.getData( )
            pen_color_rgb = (0.0, 0.0, 0.0, 1.0)
            pen_width = 1
            pen_style = str(Qt.SolidLine)

            if item.opts.get('pen') is not None:
                pen_obj = item.opts [ 'pen' ]
                pen_color_rgb = pen_obj.color( ).getRgbF( )
                pen_width = pen_obj.width( )
                pen_style = str(pen_obj.style( ))

            return {
                'x_data': x.tolist( ),
                'y_data': y.tolist( ),
                'pen_color_rgb': pen_color_rgb,
                'pen_width': pen_width,
                'pen_style': pen_style,
                'name': item.name( ),
                'z_value': item.zValue( )
            }

        project_state [ 'held_x_plots' ] = [
            save_plot_item(item) for item in self.x_slice_plot_widget.listDataItems( )
            if item != self.x_slice_curve
        ]

        project_state [ 'held_y_plots' ] = [
            save_plot_item(item) for item in self.y_slice_plot_widget.listDataItems( )
            if item != self.y_slice_curve
        ]

        if self.original_diagonal_plot_widget_instance:
            # Update the saved state with current values from the open window
            self.saved_original_diagonal_plot_state [ 'min_level' ] = float(
                self.original_diagonal_plot_widget_instance.min_level_input.text( ))
            self.saved_original_diagonal_plot_state [ 'max_level' ] = float(
                self.original_diagonal_plot_widget_instance.max_level_input.text( ))
            self.saved_original_diagonal_plot_state [ 'diagonal_slice_linewidth' ] = \
            self.original_diagonal_plot_widget_instance.diagonal_slice_curve.opts [ 'pen' ].width( )
            self.saved_original_diagonal_plot_state [ 'dashed_diagonal_line_linewidth' ] = \
            self.original_diagonal_plot_widget_instance.dashed_diagonal_line.opts [ 'pen' ].width( )
            self.saved_original_diagonal_plot_state [
                'dashed_diagonal_line_color' ] = self.original_diagonal_plot_widget_instance.current_dashed_line_color
            self.saved_original_diagonal_plot_state [
                'contour_xlabel' ] = self.original_diagonal_plot_widget_instance.contour_plot_widget.getPlotItem( ).getAxis(
                'bottom').labelText
            self.saved_original_diagonal_plot_state [
                'contour_ylabel' ] = self.original_diagonal_plot_widget_instance.contour_plot_widget.getPlotItem( ).getAxis(
                'left').labelText
            self.saved_original_diagonal_plot_state [
                'diagonal_xlabel' ] = self.original_diagonal_plot_widget_instance.diagonal_plot_widget.getPlotItem( ).getAxis(
                'bottom').labelText
            self.saved_original_diagonal_plot_state [
                'diagonal_ylabel' ] = self.original_diagonal_plot_widget_instance.diagonal_plot_widget.getPlotItem( ).getAxis(
                'left').labelText
            self.saved_original_diagonal_plot_state [
                'is_spline_corrected' ] = False  # This must be False for original data
        project_state [ 'saved_original_diagonal_plot_state' ] = self.saved_original_diagonal_plot_state

        if self.spline_corrected_diagonal_plot_widget_instance:
            # Update the saved state with current values from the open window
            self.saved_spline_corrected_diagonal_plot_state [ 'min_level' ] = float(
                self.spline_corrected_diagonal_plot_widget_instance.min_level_input.text( ))
            self.saved_spline_corrected_diagonal_plot_state [ 'max_level' ] = float(
                self.spline_corrected_diagonal_plot_widget_instance.max_level_input.text( ))
            self.saved_spline_corrected_diagonal_plot_state [ 'diagonal_slice_linewidth' ] = \
            self.spline_corrected_diagonal_plot_widget_instance.diagonal_slice_curve.opts [ 'pen' ].width( )
            self.saved_spline_corrected_diagonal_plot_state [ 'dashed_diagonal_line_linewidth' ] = \
            self.spline_corrected_diagonal_plot_widget_instance.dashed_diagonal_line.opts [ 'pen' ].width( )
            self.saved_spline_corrected_diagonal_plot_state [
                'dashed_diagonal_line_color' ] = self.spline_corrected_diagonal_plot_widget_instance.current_dashed_line_color
            self.saved_spline_corrected_diagonal_plot_state [
                'contour_xlabel' ] = self.spline_corrected_diagonal_plot_widget_instance.contour_plot_widget.getPlotItem( ).getAxis(
                'bottom').labelText
            self.saved_spline_corrected_diagonal_plot_state [
                'contour_ylabel' ] = self.spline_corrected_diagonal_plot_widget_instance.contour_plot_widget.getPlotItem( ).getAxis(
                'left').labelText
            self.saved_spline_corrected_diagonal_plot_state [
                'diagonal_xlabel' ] = self.spline_corrected_diagonal_plot_widget_instance.diagonal_plot_widget.getPlotItem( ).getAxis(
                'bottom').labelText
            self.saved_spline_corrected_diagonal_plot_state [
                'diagonal_ylabel' ] = self.spline_corrected_diagonal_plot_widget_instance.diagonal_plot_widget.getPlotItem( ).getAxis(
                'left').labelText
            self.saved_spline_corrected_diagonal_plot_state [
                'is_spline_corrected' ] = True  # This must be True for spline corrected data
        project_state [ 'saved_spline_corrected_diagonal_plot_state' ] = self.saved_spline_corrected_diagonal_plot_state

        # Save states of all active fitter tabs
        project_state [ 'active_fitter_tabs_states' ] = [ tab_info [ 'widget' ].get_fitter_state( ) for tab_info in
                                                          self.active_fitter_tabs ]

        try:
            with open(file_path, 'w') as f:
                json.dump(project_state, f, indent = 4)
            QMessageBox.information(self, "Success", f"Project saved to {file_path}")
            self._current_project_file = file_path
            self._data_modified = False
            self._update_window_title( )
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {str(e)}")
            return False

    def _save_project(self):
        """
        Saves the current project state to the current project file.
        If no project file is set, it calls _save_project_as.
        Returns True if save was successful, False otherwise.
        """
        if not self._current_project_file:
            return self._save_project_as( )
        else:
            return self._save_project_data(self._current_project_file)

    def _save_project_as(self):
        """
        Prompts the user for a new file path and saves the current project state to it.
        Returns True if save was successful, False otherwise (e.g., user cancelled).
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", "", "Project Files (*.specdat);;JSON Files (*.json);;All Files (*)"
        )

        if not file_path:
            QMessageBox.information(self, "Save Project As", "Project save cancelled.")
            return False

        return self._save_project_data(file_path)

    def _load_project_from_path(self, file_path):
        """
        Loads the application state from a given JSON file path.
        This is a helper method called by _load_project and the command-line argument handler.
        """
        try:
            with open(file_path, 'r') as f:
                project_state = json.load(f)

            self.clear_all_markers( )
            self.clear_detected_peaks( )
            self.x_unit_input.textChanged.disconnect(self._set_data_modified)
            self.y_unit_input.textChanged.disconnect(self._set_data_modified)
            self.min_level_input.textChanged.disconnect(self._set_data_modified)
            self.max_level_input.textChanged.disconnect(self._set_data_modified)
            self.marker_font_size_input.valueChanged.disconnect(self._update_marker_font_size)
            # self.s_value_spinbox.valueChanged.disconnect(self._on_s_value_changed) # Removed

            self.clear_x_slice_plots( )
            self.clear_y_slice_plots( )

            # Close all active fitter tabs before loading new ones
            for tab_info in list(self.active_fitter_tabs):
                self.tab_widget.removeTab(self.tab_widget.indexOf(tab_info [ 'widget' ]))
                tab_info [ 'widget' ].deleteLater( )  # Ensure the widget is properly deleted
            self.active_fitter_tabs.clear( )

            if project_state.get('data_loaded', False):
                self._initial_raw_x_values = np.array(project_state [ 'initial_raw_x_values' ])
                self._initial_raw_y_values = np.array(project_state [ 'initial_raw_y_values' ])
                self._initial_raw_signal_data = np.array(project_state [ 'initial_raw_signal_data' ])

                self.current_x_values = self._initial_raw_x_values.copy( )
                self.current_y_values = self._initial_raw_y_values.copy( )
                self.current_signal_data = self._initial_raw_signal_data.copy( )
                self.data_loaded = True

                self.is_spline_corrected = project_state.get('is_spline_corrected', False)
                # spline_s_value = project_state.get('spline_s_value', 0.0) # Removed
                # self.s_value_spinbox.setValue(spline_s_value) # Removed

                if self.is_spline_corrected:
                    # Call without s_value
                    corrected_data = self.spline_baseline_correction(
                        self.current_signal_data, self.current_x_values
                    )
                    self.current_signal_data = corrected_data
                    print("Spline correction re-applied during load.")

                self._refresh_all_plots( )

                self.x_slider.setValue(project_state.get('x_slider_value', self.x_dim // 2 if self.x_dim else 0))
                self.y_slider.setValue(project_state.get('y_slider_value', self.y_dim // 2 if self.y_dim else 0))

                self.min_level_input.setText(project_state.get('min_level_input', ''))
                self.max_level_input.setText(project_state.get('max_level_input', ''))
            else:
                self.data_loaded = False
                self.update_plots( )

            self.axis_labels.update(project_state.get('axis_labels', { }))
            self._apply_axis_labels( )

            self.peak_finder_params.update(project_state.get('peak_finder_params', { }))

            self.x_legend_font_size = project_state.get('x_legend_font_size', 14)
            self.y_legend_font_size = project_state.get('y_legend_font_size', 14)
            self._apply_legend_font_size(self.x_slice_legend, self.x_legend_font_size)
            self._apply_legend_font_size(self.y_slice_legend, self.y_legend_font_size)

            self.show_peak_labels = project_state.get('show_peak_labels', True)

            self.current_slice_linewidth = project_state.get('current_slice_linewidth', 2)
            self.x_slice_curve.setPen(pg.mkPen('b', width = self.current_slice_linewidth))
            self.y_slice_curve.setPen(pg.mkPen('r', width = self.current_slice_linewidth))

            self.x_slice_legend_unit = project_state.get('x_slice_legend_unit', "cm^-1")
            self.y_slice_legend_unit = project_state.get('y_slice_legend_unit', "cm^-1")
            self.x_unit_input.setText(self.x_slice_legend_unit)
            self.y_unit_input.setText(self.y_slice_legend_unit)

            self.marker_font_size = project_state.get('marker_font_size', 10)
            self.marker_font_size_input.setValue(
                self.marker_font_size)

            # Load interpolation settings for main window slices
            self._current_interp_method = project_state.get('current_interp_method', "None")
            self._current_interp_multiplier = project_state.get('current_interp_multiplier', 1)
            self.interp_method_combo.setCurrentText(self._current_interp_method)
            self.interp_multiplier_combo.setCurrentText(f"x{self._current_interp_multiplier}")


            for marker_data in project_state.get('placed_markers', [ ]):
                x, y = marker_data [ 'x' ], marker_data [ 'y' ]
                color_rgb = marker_data [ 'color' ]
                color = QColor.fromRgbF(*color_rgb)
                font_size = marker_data.get('font_size', self.marker_font_size)
                anchor = tuple(marker_data.get('anchor', (1, 1)))
                is_peak_label = marker_data.get('is_peak_label', False)
                symbol = marker_data.get('symbol', 'x')
                size = marker_data.get('size', 10)
                pen_color_rgb = marker_data.get('pen_color', color_rgb)
                brush_color_rgb = marker_data.get('brush_color', (0.0, 0.0, 0.0, 0.0))

                scatter_item = pg.PlotDataItem([ x ], [ y ], symbol = symbol, size = size,
                                               pen = pg.mkPen(QColor.fromRgbF(*pen_color_rgb), width = 2),
                                               brush = pg.mkBrush(QColor.fromRgbF(*brush_color_rgb)))
                self.signal_plot_widget.addItem(scatter_item)

                text_item = ClickableTextItem(text = marker_data [ 'text' ], color = color, anchor = anchor,
                                              parent_app = self, associated_scatter = scatter_item,
                                              is_peak_label = is_peak_label)
                text_item.setPos(x, y)
                text_item.setFont(QFont("Times New Roman", font_size))
                self.signal_plot_widget.addItem(text_item)
                self.placed_markers.append((scatter_item, text_item))

            for peak_data in project_state.get('detected_peaks_items', [ ]):
                x, y = peak_data [ 'x' ], peak_data [ 'y' ]
                color_rgb = peak_data [ 'color' ]
                color = QColor.fromRgbF(*color_rgb)
                font_size = peak_data.get('font_size', 10)
                anchor = tuple(peak_data.get('anchor', (0.5, 1)))
                is_peak_label = peak_data.get('is_peak_label', True)
                symbol = peak_data.get('symbol', 'o')
                size = peak_data.get('size', 10)
                pen_color_rgb = peak_data.get('pen_color', color_rgb)
                brush_color_rgb = peak_data.get('brush_color', (0.0, 0.0, 0.0, 0.0))

                scatter_item = pg.PlotDataItem([ x ], [ y ], symbol = symbol, size = size,
                                               pen = pg.mkPen(QColor.fromRgbF(*pen_color_rgb), width = 2),
                                               brush = pg.mkBrush(QColor.fromRgbF(*brush_color_rgb)))
                self.signal_plot_widget.addItem(scatter_item)

                text_item = ClickableTextItem(text = peak_data [ 'text' ], color = color, anchor = anchor,
                                              parent_app = self, associated_scatter = scatter_item,
                                              is_peak_label = is_peak_label)
                text_item.setPos(x, y)
                text_item.setFont(QFont("Times New Roman", font_size))
                text_item.setVisible(self.show_peak_labels)
                self.signal_plot_widget.addItem(text_item)
                self.detected_peaks_items.append((scatter_item, text_item))

            self.held_x_slices_count = 0
            for plot_data in project_state.get('held_x_plots', [ ]):
                x_data = np.array(plot_data [ 'x_data' ])
                y_data = np.array(plot_data [ 'y_data' ])
                pen_color = QColor.fromRgbF(*plot_data [ 'pen_color_rgb' ])
                name = plot_data [ 'name' ]
                pen_width = plot_data.get('pen_width', self.current_slice_linewidth)
                pen_style_str = plot_data.get('pen_style', str(Qt.SolidLine))
                pen_style = Qt.SolidLine
                if pen_style_str == str(Qt.DotLine):
                    pen_style = Qt.DotLine
                elif pen_style_str == str(Qt.DashLine):
                    pen_style = Qt.DashLine
                elif pen_style_str == str(Qt.DashDotLine):
                    pen_style = Qt.DashDotLine
                elif pen_style_str == str(Qt.SolidLine):
                    pen_style = Qt.SolidLine

                pen = pg.mkPen(pen_color, width = pen_width, style = pen_style)
                self.x_slice_plot_widget.plot(x_data, y_data, pen = pen, name = name)
                self.held_x_slices_count += 1

            self.held_y_slices_count = 0
            for plot_data in project_state.get('held_y_plots', [ ]):
                x_data = np.array(plot_data [ 'x_data' ])
                y_data = np.array(plot_data [ 'y_data' ])
                pen_color = QColor.fromRgbF(*plot_data [ 'pen_color_rgb' ])
                name = plot_data [ 'name' ]
                pen_width = plot_data.get('pen_width', self.current_slice_linewidth)
                pen_style_str = plot_data.get('pen_style', str(Qt.SolidLine))
                pen_style = Qt.SolidLine
                if pen_style_str == str(Qt.DotLine):
                    pen_style = Qt.DotLine
                elif pen_style_str == str(Qt.DashLine):
                    pen_style = Qt.DashLine
                elif pen_style_str == str(Qt.DashDotLine):
                    pen_style = Qt.DashDotLine
                elif pen_style_str == str(Qt.SolidLine):
                    pen_style = Qt.SolidLine

                pen = pg.mkPen(pen_color, width = pen_width, style = pen_style)
                self.y_slice_plot_widget.plot(x_data, y_data, pen = pen, name = name)
                self.held_y_slices_count += 1

            self.saved_original_diagonal_plot_state = project_state.get('saved_original_diagonal_plot_state')
            if self.saved_original_diagonal_plot_state:
                self._open_diagonal_plot_tab(self.saved_original_diagonal_plot_state, False)
            else:
                if self.original_diagonal_plot_widget_instance:
                    idx = self.tab_widget.indexOf(self.original_diagonal_plot_widget_instance)
                    if idx != -1:
                        self.tab_widget.removeTab(idx)
                    self.original_diagonal_plot_widget_instance.deleteLater( )
                    self.original_diagonal_plot_widget_instance = None

            self.saved_spline_corrected_diagonal_plot_state = project_state.get(
                'saved_spline_corrected_diagonal_plot_state')
            if self.saved_spline_corrected_diagonal_plot_state:
                self._open_diagonal_plot_tab(self.saved_spline_corrected_diagonal_plot_state, True)
            else:
                if self.spline_corrected_diagonal_plot_widget_instance:
                    idx = self.tab_widget.indexOf(self.spline_corrected_diagonal_plot_widget_instance)
                    if idx != -1:
                        self.tab_widget.removeTab(idx)
                    self.spline_corrected_diagonal_plot_widget_instance.deleteLater( )
                    self.spline_corrected_diagonal_plot_widget_instance = None

            # Load states of active fitter tabs
            fitter_states = project_state.get('active_fitter_tabs_states', [ ])
            for state in fitter_states:
                fitter_widget = GaussianFitterApp(self)
                fitter_widget.set_fitter_state(state)
                tab_index = self.tab_widget.addTab(fitter_widget, fitter_widget.objectName( ))
                self.active_fitter_tabs.append({ 'widget': fitter_widget, 'tab_index': tab_index })
                fitter_widget.destroyed.connect(
                    lambda: self._remove_fitter_tab(fitter_widget)
                )

            if self.show_peak_labels:
                self.toggle_peak_labels_action.setText("Hide Peak Labels")
            else:
                self.toggle_peak_labels_action.setText("Show Peak Labels")

            QMessageBox.information(self, "Load Project", f"Project loaded successfully from '{file_path}'")
            self._current_project_file = file_path
            self._data_modified = False
            self._update_window_title( )
            return True
        except json.JSONDecodeError as jde:
            QMessageBox.critical(self, "Load Error", f"Failed to parse JSON file. Invalid format: {jde}")
            self._data_modified = False
            self._update_window_title( )
            return False
        except KeyError as ke:
            QMessageBox.critical(self, "Load Error",
                                 f"Missing data in project file: {ke}. File might be corrupted or from an incompatible version.")
            self._data_modified = False
            self._update_window_title( )
            return False
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load project: {e}")
            self._data_modified = False
            self._update_window_title( )
            return False
        finally:
            self.x_unit_input.textChanged.connect(self._set_data_modified)
            self.y_unit_input.textChanged.connect(self._set_data_modified)
            self.min_level_input.textChanged.connect(self._set_data_modified)
            self.max_level_input.textChanged.connect(self._set_data_modified)
            self.marker_font_size_input.valueChanged.connect(self._update_marker_font_size)

    def _load_project(self):
        """
        Opens a file dialog to select a project file and then loads it.
        """
        if self._data_modified:
            reply = QMessageBox.question(self, 'Save Changes',
                                         "You have unsaved changes. Do you want to save them before loading a new project?",
                                         QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                                         QMessageBox.Save)
            if reply == QMessageBox.Save:
                save_successful = self._save_project( )
                if not save_successful:
                    return
            elif reply == QMessageBox.Cancel:
                return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Project",
            "",
            "Project Files (*.specdat);;JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self._load_project_from_path(file_path)
        else:
            QMessageBox.information(self, "Load Project", "Project load cancelled.")

    def _update_marker_font_size(self, new_size):
        """
        Updates the font size of all existing markers and peak labels based on the new_size value.
        This method is connected to the QSpinBox's valueChanged signal.
        """

        if self.marker_font_size == new_size:
            return

        self.marker_font_size = new_size
        for scatter_item, text_item in self.placed_markers:
            font = QFont("Times New Roman", self.marker_font_size)
            text_item.setFont(font)

        for scatter_item, text_item in self.detected_peaks_items:
            font = QFont("Times New Roman", self.marker_font_size)
            text_item.setFont(font)

        self._set_data_modified( )

    def closeEvent(self, event):
        """
        Overrides the default close event to prompt the user to save changes.
        """
        if self._data_modified:
            reply = QMessageBox.question(self, 'Save Changes',
                                         "You have unsaved changes. Do you want to save them before quitting?",
                                         QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                                         QMessageBox.Save)

            if reply == QMessageBox.Save:
                save_successful = self._save_project( )
                if save_successful:
                    # Close all active fitter tabs
                    for tab_info in list(self.active_fitter_tabs):
                        self.tab_widget.removeTab(self.tab_widget.indexOf(tab_info [ 'widget' ]))
                        tab_info [ 'widget' ].deleteLater( )
                    # Safely handle diagonal_plot_widget_instance before closing the app
                    if self.original_diagonal_plot_widget_instance is not None:
                        try:
                            _ = self.original_diagonal_plot_widget_instance.objectName( )
                            idx = self.tab_widget.indexOf(self.original_diagonal_plot_widget_instance)
                            if idx != -1:
                                self.tab_widget.removeTab(idx)
                            self.original_diagonal_plot_widget_instance.deleteLater( )
                            self.original_diagonal_plot_widget_instance = None
                        except RuntimeError:
                            self.original_diagonal_plot_widget_instance = None

                    if self.spline_corrected_diagonal_plot_widget_instance is not None:
                        try:
                            _ = self.spline_corrected_diagonal_plot_widget_instance.objectName( )
                            idx = self.tab_widget.indexOf(self.spline_corrected_diagonal_plot_widget_instance)
                            if idx != -1:
                                self.tab_widget.removeTab(idx)
                            self.spline_corrected_diagonal_plot_widget_instance.deleteLater( )
                            self.spline_corrected_diagonal_plot_widget_instance = None
                        except RuntimeError:
                            self.spline_corrected_diagonal_plot_widget_instance = None
                    event.accept( )
                else:
                    event.ignore( )
            elif reply == QMessageBox.Discard:
                # Close all active fitter tabs
                for tab_info in list(self.active_fitter_tabs):
                    self.tab_widget.removeTab(self.tab_widget.indexOf(tab_info [ 'widget' ]))
                    tab_info [ 'widget' ].deleteLater( )
                # Safely handle diagonal_plot_widget_instance
                if self.original_diagonal_plot_widget_instance is not None:
                    try:
                        _ = self.original_diagonal_plot_widget_instance.objectName( )
                        idx = self.tab_widget.indexOf(self.original_diagonal_plot_widget_instance)
                        if idx != -1:
                            self.tab_widget.removeTab(idx)
                        self.original_diagonal_plot_widget_instance.deleteLater( )
                        self.original_diagonal_plot_widget_instance = None
                    except RuntimeError:
                        self.original_diagonal_plot_widget_instance = None

                if self.spline_corrected_diagonal_plot_widget_instance is not None:
                    try:
                        _ = self.spline_corrected_diagonal_plot_widget_instance.objectName( )
                        idx = self.tab_widget.indexOf(self.spline_corrected_diagonal_plot_widget_instance)
                        if idx != -1:
                            self.tab_widget.removeTab(idx)
                        self.spline_corrected_diagonal_plot_widget_instance.deleteLater( )
                        self.spline_corrected_diagonal_plot_widget_instance = None
                    except RuntimeError:
                        self.spline_corrected_diagonal_plot_widget_instance = None
                event.accept( )
            else:
                event.ignore( )
        else:  # No data modified
            # Close all active fitter tabs
            for tab_info in list(self.active_fitter_tabs):
                self.tab_widget.removeTab(self.tab_widget.indexOf(tab_info [ 'widget' ]))
                tab_info [ 'widget' ].deleteLater( )
            # Safely handle diagonal_plot_widget_instance
            if self.original_diagonal_plot_widget_instance is not None:
                try:
                    _ = self.original_diagonal_plot_widget_instance.objectName( )
                    idx = self.tab_widget.indexOf(self.original_diagonal_plot_widget_instance)
                    if idx != -1:
                        self.tab_widget.removeTab(idx)
                    self.original_diagonal_plot_widget_instance.deleteLater( )
                    self.original_diagonal_plot_widget_instance = None
                except RuntimeError:
                    self.original_diagonal_plot_widget_instance = None

            if self.spline_corrected_diagonal_plot_widget_instance is not None:
                try:
                    _ = self.spline_corrected_diagonal_plot_widget_instance.objectName( )
                    idx = self.tab_widget.indexOf(self.spline_corrected_diagonal_plot_widget_instance)
                    if idx != -1:
                        self.tab_widget.removeTab(idx)
                    self.spline_corrected_diagonal_plot_widget_instance.deleteLater( )
                    self.spline_corrected_diagonal_plot_widget_instance = None
                except RuntimeError:
                    self.spline_corrected_diagonal_plot_widget_instance = None
            event.accept( )


if __name__ == '__main__':
    if sys.platform == 'win32':
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('dataviewer2D.MjölnIR_app.1.0')
        except AttributeError:
            pass
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(':/icons/icon.ico'))

    window = SignalPlotterApp( )
    window.show( )
    if len(sys.argv) > 1:
        file_to_open = sys.argv [ 1 ]
        if os.path.exists(file_to_open) and file_to_open.lower( ).endswith(('.specdat', '.json')):#
            try:
                window._load_project_from_path(file_to_open)
            except Exception as e:
                QMessageBox.critical(window, "Error Opening Project",
                                     f"Failed to load project from '{file_to_open}': {e}")
        else:
            QMessageBox.warning(window, "Unsupported File",
                                f"The file '{file_to_open}' is not a recognized project file (.specdat or .json).")

    sys.exit(app.exec_( ))
