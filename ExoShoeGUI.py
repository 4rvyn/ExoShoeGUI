# <<< START OF FILE >>>
import asyncio
from bleak import BleakScanner, BleakClient, BleakError
import logging
from typing import Optional, Callable, Dict, List, Tuple, Set, Any, Type
import time
from functools import partial
from collections import deque # Keep for potential future optimization if lists grow huge
import threading
import sys
import datetime
import csv # Added for CSV writing
import os
import pandas as pd # Re-added for easier CSV data handling (merging/resampling)
import struct
import bisect
import numpy as np # Still needed for pyqtgraph, potentially useful for PGF data prep

# --- Matplotlib Imports (used ONLY for PGF export) ---
import matplotlib # Use Agg backend to avoid GUI conflicts if possible
matplotlib.use('Agg') # Set backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import scienceplots # Make sure it's installed: pip install scienceplots for formatting

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QGridLayout, QCheckBox, QLineEdit,
    QScrollArea, QMessageBox, QSizePolicy, QTextEdit, QTabWidget,
    QComboBox # Added QComboBox
)
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QMetaObject, QThread

# --- PyQtGraph Import ---
import pyqtgraph as pg

# Apply PyQtGraph global options for background/foreground
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOptions(antialias=True) # Ensure anti-aliasing is enabled

# Configure logging (same as before)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Create a separate logger for data logs (same as before)
data_logger = logging.getLogger("data_logger")
data_logger.propagate = False # Don't send data logs to root logger's handlers by default
data_console_handler = logging.StreamHandler() # Specific handler for console data logs
data_console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
data_console_handler.setLevel(logging.INFO)
data_logger.addHandler(data_console_handler)
data_logger.setLevel(logging.DEBUG) # Logger level should be lowest level to capture

# --- Custom Log Handler for PyQt GUI ---
class QtLogHandler(logging.Handler, QObject):
    log_signal = pyqtSignal(str)
    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        try:
            msg = self.format(record)
            # Emit the signal to be connected to the text box in the main thread
            self.log_signal.emit(msg)
        except Exception:
            self.handleError(record) # Default handler error handling

# --- Global variables ---
disconnected_event = asyncio.Event()
last_received_time = 0
# data_buffers now holds ALL received data since connection/clear
data_buffers: Dict[str, List[Tuple[float, float]]] = {}
start_time: Optional[datetime.datetime] = None # Absolute start time of the current connection/session
stop_flag = False
state = "idle"
current_task: Optional[asyncio.Task] = None
loop: Optional[asyncio.AbstractEventLoop] = None
client: Optional[BleakClient] = None
plotting_paused = False
flowing_interval = 10.0  # Initial interval in seconds

# --- Configuration classes ---
class CharacteristicConfig:
    def __init__(self, uuid: str, handler: Callable[[bytearray], dict], produces_data_types: List[str]):
        self.uuid = uuid
        self.handler = handler
        self.produces_data_types = produces_data_types # List of data keys this characteristic's handler produces

class DeviceConfig:
    def __init__(self, name: str, service_uuid: str, characteristics: list[CharacteristicConfig],
                 find_timeout: float = 10.0, data_timeout: float = 1.0):
        self.name = name
        self.service_uuid = service_uuid
        self.characteristics = characteristics
        self.find_timeout = find_timeout
        self.data_timeout = data_timeout
        # Map data types back to their source UUIDs
        self.data_type_to_uuid_map: Dict[str, str] = self._build_data_type_map()

    def _build_data_type_map(self) -> Dict[str, str]:
        """Builds the mapping from data_type keys to their source UUID."""
        mapping = {}
        for char_config in self.characteristics:
            for data_type in char_config.produces_data_types:
                if data_type in mapping:
                    logger.warning(f"Data type '{data_type}' is produced by multiple UUIDs. Mapping to {char_config.uuid}.")
                mapping[data_type] = char_config.uuid
        # logger.debug(f"Built data_type -> UUID map: {mapping}")
        return mapping

    def update_name(self, name: str):
        self.name = name
        # Potentially update other device-specific settings here if needed in the future

    def get_uuid_for_data_type(self, data_type: str) -> Optional[str]:
        return self.data_type_to_uuid_map.get(data_type)


#####################################################################################################################
# Start of customizable section
#####################################################################################################################
# This section is where you can customize the device configuration, data handling, and plotting.

# 1. Data handlers for different characteristics
# 2. Device configuration (add UUIDs AND `produces_data_types`)
# 3. Define GUI Component Classes (e.g., plots, indicators)
# 4. Define Tab Layout Configuration using the components

# --- Data Handlers ---
def handle_orientation_data(data: bytearray) -> dict:
    try:
        if len(data) != 6: data_logger.error("Invalid data length for orientation"); return {}
        x_int, y_int, z_int = struct.unpack('<hhh', data)
        x, y, z = x_int / 100.0, y_int / 100.0, z_int / 100.0
        data_logger.info(f"Orientation Data: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        return {'orientation_x': x, 'orientation_y': y, 'orientation_z': z}
    except Exception as e: data_logger.error(f"Error parsing orientation data: {e}"); return {}

def handle_gyro_data(data: bytearray) -> dict:
    try:
        if len(data) != 6: data_logger.error("Invalid data length for gyro"); return {}
        x_int, y_int, z_int = struct.unpack('<hhh', data)
        x, y, z = x_int / 100.0, y_int / 100.0, z_int / 100.0
        data_logger.info(f"Gyro Data: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        return {'gyro_x': x, 'gyro_y': y, 'gyro_z': z}
    except Exception as e: data_logger.error(f"Error parsing gyro data: {e}"); return {}

def handle_lin_accel_data(data: bytearray) -> dict:
    try:
        if len(data) != 6: data_logger.error("Invalid data length for linear acceleration"); return {}
        x_int, y_int, z_int = struct.unpack('<hhh', data)
        x, y, z = x_int / 100.0, y_int / 100.0, z_int / 100.0
        data_logger.info(f"Linear Acceleration Data: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        return {'lin_accel_x': x, 'lin_accel_y': y, 'lin_accel_z': z}
    except Exception as e: data_logger.error(f"Error parsing linear acceleration data: {e}"); return {}

def handle_mag_data(data: bytearray) -> dict:
    try:
        if len(data) != 6: data_logger.error("Invalid data length for magnetometer"); return {}
        x_int, y_int, z_int = struct.unpack('<hhh', data)
        x, y, z = x_int / 100.0, y_int / 100.0, z_int / 100.0
        data_logger.info(f"Magnetometer Data: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        return {'mag_x': x, 'mag_y': y, 'mag_z': z}
    except Exception as e: data_logger.error(f"Error parsing magnetometer data: {e}"); return {}

def handle_accel_data(data: bytearray) -> dict:
    try:
        if len(data) != 6: data_logger.error("Invalid data length for accelerometer"); return {}
        x_int, y_int, z_int = struct.unpack('<hhh', data)
        x, y, z = x_int / 100.0, y_int / 100.0, z_int / 100.0
        data_logger.info(f"Accelerometer Data: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        return {'accel_x': x, 'accel_y': y, 'accel_z': z}
    except Exception as e: data_logger.error(f"Error parsing accelerometer data: {e}"); return {}

def handle_gravity_data(data: bytearray) -> dict:
    try:
        if len(data) != 6: data_logger.error("Invalid data length for gravity"); return {}
        x_int, y_int, z_int = struct.unpack('<hhh', data)
        x, y, z = x_int / 100.0, y_int / 100.0, z_int / 100.0
        data_logger.info(f"Gravity Data: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        return {'gravity_x': x, 'gravity_y': y, 'gravity_z': z}
    except Exception as e: data_logger.error(f"Error parsing gravity data: {e}"); return {}

# --- Device Configuration (Initial Device Name still set here) ---
# This object's 'name' attribute will be updated by the dropdown menu
device_config = DeviceConfig(
    name="Nano33IoT", # Initial default name
    service_uuid="19B10000-E8F2-537E-4F6C-D104768A1214", # Assuming same service UUID for now
    characteristics=[
        CharacteristicConfig(uuid="19B10001-E8F2-537E-4F6C-D104768A1214", handler=handle_orientation_data,
                             produces_data_types=['orientation_x', 'orientation_y', 'orientation_z']),
        CharacteristicConfig(uuid="19B10003-E8F2-537E-4F6C-D104768A1214", handler=handle_gyro_data,
                             produces_data_types=['gyro_x', 'gyro_y', 'gyro_z']),
        CharacteristicConfig(uuid="19B10004-E8F2-537E-4F6C-D104768A1214", handler=handle_lin_accel_data,
                             produces_data_types=['lin_accel_x', 'lin_accel_y', 'lin_accel_z']),
        CharacteristicConfig(uuid="19B10005-E8F2-537E-4F6C-D104768A1214", handler=handle_mag_data,
                             produces_data_types=['mag_x', 'mag_y', 'mag_z']),
        CharacteristicConfig(uuid="19B10006-E8F2-537E-4F6C-D104768A1214", handler=handle_accel_data,
                             produces_data_types=['accel_x', 'accel_y', 'accel_z']),
        CharacteristicConfig(uuid="19B10007-E8F2-537E-4F6C-D104768A1214", handler=handle_gravity_data,
                             produces_data_types=['gravity_x', 'gravity_y', 'gravity_z']),
    ],
    find_timeout=5.0,
    data_timeout=1.0
)

# List of available device names for the dropdown
AVAILABLE_DEVICE_NAMES = ["Nano33IoT", "NanoESP32"]


# --- Component Base Class ---
class BaseGuiComponent(QWidget):
    # Base class for modular GUI components.
    def __init__(self, config: Dict[str, Any], data_buffers_ref: Dict[str, List[Tuple[float, float]]], device_config_ref: DeviceConfig, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config = config
        self.data_buffers_ref = data_buffers_ref
        self.device_config_ref = device_config_ref
        # Basic size policy, can be overridden by subclasses or config
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def get_widget(self) -> QWidget:
        """Returns the primary widget this component manages."""
        return self # Default: the component itself is the widget

    def update_component(self, current_relative_time: float, is_flowing: bool):
        """Update the component's visual representation based on current data and time."""
        raise NotImplementedError("Subclasses must implement update_component")

    def clear_component(self):
        """Clear the component's display and internal state."""
        raise NotImplementedError("Subclasses must implement clear_component")

    def get_required_data_types(self) -> Set[str]:
        """Returns a set of data_type keys this component requires."""
        return set() # Default: no data required

    def handle_missing_uuids(self, missing_uuids_for_component: Set[str]):
        """
        Called by the GuiManager when relevant UUIDs are found to be missing.
        'missing_uuids_for_component' contains only the UUIDs relevant to this specific component.
        """
        pass # Default: do nothing


# --- Specific Component Implementations ---

class TimeSeriesPlotComponent(BaseGuiComponent):
    """A GUI component that displays time-series data using pyqtgraph."""
    def __init__(self, config: Dict[str, Any], data_buffers_ref: Dict[str, List[Tuple[float, float]]], device_config_ref: DeviceConfig, parent: Optional[QWidget] = None):
        super().__init__(config, data_buffers_ref, device_config_ref, parent)

        self.plot_widget = pg.PlotWidget()
        self.plot_item: pg.PlotItem = self.plot_widget.getPlotItem()
        self.lines: Dict[str, pg.PlotDataItem] = {} # data_type -> PlotDataItem
        self.uuid_not_found_text: Optional[pg.TextItem] = None
        self.required_data_types: Set[str] = set()
        self.missing_relevant_uuids: Set[str] = set() # UUIDs this plot needs that are missing

        self._setup_plot()

        # Main layout for this component (contains just the plot widget)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

    def get_widget(self) -> QWidget:
        return self # The component itself holds the plot

    def _setup_plot(self):
        """Configures the plot based on the self.config dictionary."""
        plot_height = self.config.get('plot_height')
        plot_width = self.config.get('plot_width')
        # Apply size constraints to the component itself, layout will handle the rest
        if plot_height is not None: self.setFixedHeight(plot_height)
        if plot_width is not None: self.setFixedWidth(plot_width)
        # If only one dimension is set, allow expansion in the other
        if plot_height is not None and plot_width is None:
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        elif plot_width is not None and plot_height is None:
            self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        elif plot_width is None and plot_height is None:
             self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        else: # Both fixed
             self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


        self.plot_item.setTitle(self.config.get('title', 'Plot'), size='10pt')
        self.plot_item.setLabel('bottom', self.config.get('xlabel', 'Time [s]'))
        self.plot_item.setLabel('left', self.config.get('ylabel', 'Value'))
        self.plot_item.showGrid(x=True, y=True, alpha=0.1)
        self.plot_item.addLegend(offset=(10, 5))
        self.plot_item.getViewBox().setDefaultPadding(0.01)
        # Connect view change signal to reposition text
        self.plot_item.getViewBox().sigRangeChanged.connect(self._position_text_item)

        for dataset in self.config.get('datasets', []):
            data_type = dataset['data_type']
            self.required_data_types.add(data_type)
            pen = pg.mkPen(color=dataset.get('color', 'k'), width=1.5)
            line = self.plot_item.plot(pen=pen, name=dataset.get('label', data_type))
            self.lines[data_type] = line

        self.clear_component() # Initialize axes

    def get_required_data_types(self) -> Set[str]:
        return self.required_data_types

    def handle_missing_uuids(self, missing_uuids_for_component: Set[str]):
        """Shows or hides the 'UUID not found' message."""
        self.missing_relevant_uuids = missing_uuids_for_component
        first_missing_uuid = next(iter(missing_uuids_for_component), None)

        if first_missing_uuid:
            text_content = f"UUID:\n{first_missing_uuid}\nnot found!"
            if self.uuid_not_found_text:
                # logger.debug(f"Updating existing text for plot '{self.config.get('title')}'")
                self.uuid_not_found_text.setText(text_content)
            else:
                # logger.debug(f"Creating new text for plot '{self.config.get('title')}'")
                self.uuid_not_found_text = pg.TextItem(text_content, color=(150, 150, 150), anchor=(0.5, 0.5))
                self.uuid_not_found_text.setZValue(100) # Ensure it's on top
                self.plot_item.addItem(self.uuid_not_found_text)

            # Position the text item (or reposition if it existed)
            self._position_text_item()

        else: # No missing UUIDs for this plot
            if self.uuid_not_found_text:
                # logger.debug(f"Removing text for plot '{self.config.get('title')}'")
                try:
                     self.plot_item.removeItem(self.uuid_not_found_text)
                except Exception as e:
                     logger.warning(f"Error removing text item from plot '{self.config.get('title')}': {e}")
                self.uuid_not_found_text = None

        # Trigger an update to clear lines if needed and adjust Y-axis ranging
        QTimer.singleShot(0, self._request_gui_update_for_yrange)

    def _request_gui_update_for_yrange(self):
         # Find the main window instance to emit the signal
         try:
             mw = next(widget for widget in QApplication.topLevelWidgets() if isinstance(widget, MainWindow))
             mw.request_plot_update_signal.emit() # Request general update which includes this plot
         except StopIteration: logger.error("Could not find MainWindow instance to request plot update.")
         except Exception as e: logger.error(f"Error requesting plot update: {e}")

    def _position_text_item(self):
        """Positions the text item in the center of the plot's current view."""
        if not self.uuid_not_found_text: return

        try:
            view_box = self.plot_item.getViewBox()
            if not view_box.autoRangeEnabled()[1]: # If Y is not auto-ranging, use default range
                 y_range = self.plot_item.getAxis('left').range
            else:
                 y_range = view_box.viewRange()[1] # Use view range if auto-ranging
            x_range = view_box.viewRange()[0]

            if None in x_range or None in y_range or x_range[1] <= x_range[0] or y_range[1] <= y_range[0]:
                center_x, center_y = 0.5, 0.5
            else:
                center_x = x_range[0] + (x_range[1] - x_range[0]) / 2
                center_y = y_range[0] + (y_range[1] - y_range[0]) / 2

            self.uuid_not_found_text.setPos(center_x, center_y)
        except Exception as e:
            logger.warning(f"Could not position text item for plot '{self.config.get('title')}': {e}")

    def update_component(self, current_relative_time: float, is_flowing: bool):
        """Updates the plot lines based on data in the shared buffer."""
        if plotting_paused: return

        # --- Determine Time Axis Range ---
        min_time_axis = 0
        max_time_axis = max(current_relative_time, flowing_interval)

        if is_flowing:
            min_time_axis = max(0, current_relative_time - flowing_interval)
            max_time_axis = current_relative_time
            self.plot_item.setXRange(min_time_axis, max_time_axis, padding=0.02)
        else:
            max_data_time = 0
            # Only consider data types from non-missing UUIDs for axis range
            for data_type in self.required_data_types:
                 uuid = self.device_config_ref.get_uuid_for_data_type(data_type)
                 if uuid and uuid not in self.missing_relevant_uuids and data_type in self.data_buffers_ref and self.data_buffers_ref[data_type]:
                     try: max_data_time = max(max_data_time, self.data_buffers_ref[data_type][-1][0])
                     except IndexError: pass

            max_time_axis = max(max_data_time, flowing_interval)
            self.plot_item.setXRange(0, max_time_axis, padding=0.02)

        # --- Update Data for Lines ---
        data_updated_in_plot = False
        plot_has_missing_uuid_text = (self.uuid_not_found_text is not None)

        for data_type, line in self.lines.items():
            uuid = self.device_config_ref.get_uuid_for_data_type(data_type)

            # Check if UUID is missing for *this specific plot*
            if uuid and uuid in self.missing_relevant_uuids:
                line.setData(x=[], y=[]) # Clear line if UUID is missing
                # logger.debug(f"Cleared line for {data_type} (UUID {uuid} missing)")
            elif data_type in self.data_buffers_ref:
                data = self.data_buffers_ref[data_type]
                plot_data_tuples = []
                if is_flowing:
                    # Find the starting index for the flowing window
                    start_idx = bisect.bisect_left(data, min_time_axis, key=lambda x: x[0])
                    plot_data_tuples = data[start_idx:]
                else:
                    plot_data_tuples = data # Use all data

                if plot_data_tuples:
                    try:
                        times = np.array([item[0] for item in plot_data_tuples])
                        values = np.array([item[1] for item in plot_data_tuples])
                        line.setData(x=times, y=values)
                        data_updated_in_plot = True
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not convert/set data for {data_type} in plot '{self.config.get('title')}': {e}")
                        line.setData(x=[], y=[])
                else:
                    line.setData(x=[], y=[]) # Clear line if no data in range

            else:
                line.setData(x=[], y=[]) # Clear line if data_type not found in buffers

        # --- Y Auto-Ranging ---
        if data_updated_in_plot and not plot_has_missing_uuid_text:
            self.plot_item.enableAutoRange(axis='y', enable=True)
        elif plot_has_missing_uuid_text:
             # Disable auto-ranging and set a default range if text is shown
             self.plot_item.enableAutoRange(axis='y', enable=False)
             # Keep existing range or set a default? Let's try setting a simple default.
             self.plot_item.setYRange(-1, 1, padding=0.1) # Example default range

    def clear_component(self):
        """Clears the plot lines and resets axes."""
        for line in self.lines.values():
            line.setData(x=[], y=[])

        # Remove "UUID not found" text if present
        if self.uuid_not_found_text:
             try: self.plot_item.removeItem(self.uuid_not_found_text)
             except Exception as e: logger.warning(f"Error removing text item during clear: {e}")
             self.uuid_not_found_text = None
        self.missing_relevant_uuids.clear()

        # Reset view ranges
        self.plot_item.setXRange(0, flowing_interval, padding=0.02)
        self.plot_item.setYRange(-1, 1, padding=0.1) # Reset Y range to default
        self.plot_item.enableAutoRange(axis='y', enable=True) # Re-enable Y auto-ranging


# --- Add more component classes here in the future ---
# Example:
# class StatusIndicatorComponent(BaseGuiComponent):
#     def __init__(self, config, data_buffers_ref, device_config_ref, parent=None):
#         super().__init__(config, data_buffers_ref, device_config_ref, parent)
#         self.label = QLabel("Status: --")
#         # ... setup layout ...
#         layout = QVBoxLayout(self)
#         layout.addWidget(self.label)
#         self.setLayout(layout)
#     def update_component(self, current_relative_time, is_flowing):
#         # Example: Update label based on latest value of a specific data_type
#         dtype = self.config.get("data_type_to_monitor")
#         if dtype and dtype in self.data_buffers_ref and self.data_buffers_ref[dtype]:
#             latest_val = self.data_buffers_ref[dtype][-1][1]
#             self.label.setText(f"{dtype}: {latest_val:.2f}")
#     def clear_component(self):
#         self.label.setText("Status: --")
#     def get_required_data_types(self) -> Set[str]:
#         dtype = self.config.get("data_type_to_monitor")
#         return {dtype} if dtype else set()


# --- Tab Layout Configuration ---
# List of dictionaries, each defining a tab.
# Each dictionary contains 'tab_title' and 'layout'.
# 'layout' is a list of component definitions for that tab's grid.
tab_configs = [
    {
        'tab_title': 'IMU Basic',
        'layout': [
            {   'component_class': TimeSeriesPlotComponent,
                'row': 0, 'col': 0, # 'colspan' and 'rowspan' can be used for larger components
                'config': { # Configuration specific to this component instance
                    'title': 'Orientation vs Time', 'xlabel': 'Time [s]', 'ylabel': 'Degrees',
                    'plot_height': 300, 'plot_width': 450, # Size constraints applied to the component
                    'datasets': [{'data_type': 'orientation_x', 'label': 'X (Roll)', 'color': 'r'},
                                 {'data_type': 'orientation_y', 'label': 'Y (Pitch)', 'color': 'g'},
                                 {'data_type': 'orientation_z', 'label': 'Z (Yaw)', 'color': 'b'}]
                }
            },
            {   'component_class': TimeSeriesPlotComponent,
                'row': 0, 'col': 1,
                'config': {
                    'title': 'Angular Velocity vs Time','xlabel': 'Time [s]','ylabel': 'Degrees/s',
                    'plot_height': 300, 'plot_width': 600,
                    'datasets': [{'data_type': 'gyro_x', 'label': 'X', 'color': 'r'},
                                 {'data_type': 'gyro_y', 'label': 'Y', 'color': 'g'},
                                 {'data_type': 'gyro_z', 'label': 'Z', 'color': 'b'}]
                }
            },
            # Add other components (plots, labels, etc.) here for this tab
            # Example: Add a placeholder label component if defined
            # { 'component_class': StatusIndicatorComponent, 'row': 1, 'col': 0,
            #   'config': {'data_type_to_monitor': 'gyro_x'} }
        ]
    },
    {
        'tab_title': 'IMU Acceleration',
        'layout': [
            {   'component_class': TimeSeriesPlotComponent,
                'row': 0, 'col': 0,
                'config': {
                    'title': 'Linear Acceleration vs Time','xlabel': 'Time [s]','ylabel': 'm/s²',
                    # No size constraints -> uses default Expanding policy
                    'datasets': [{'data_type': 'lin_accel_x', 'label': 'X', 'color': 'r'},
                                 {'data_type': 'lin_accel_y', 'label': 'Y', 'color': 'g'},
                                 {'data_type': 'lin_accel_z', 'label': 'Z', 'color': 'b'}]
                }
            },
            {   'component_class': TimeSeriesPlotComponent,
                'row': 1, 'col': 0,
                'config': {
                    'title': 'Raw Acceleration vs Time','xlabel': 'Time [s]','ylabel': 'm/s²',
                    'plot_height': 280, # Height constraint
                    'datasets': [{'data_type': 'accel_x', 'label': 'X', 'color': 'r'},
                                 {'data_type': 'accel_y', 'label': 'Y', 'color': 'g'},
                                 {'data_type': 'accel_z', 'label': 'Z', 'color': 'b'}]
                }
            },
            {   'component_class': TimeSeriesPlotComponent,
                'row': 1, 'col': 1,
                'config': {
                     'title': 'Gravity vs Time','xlabel': 'Time [s]','ylabel': 'm/s²',
                     'plot_width': 400, # Width constraint
                     'datasets': [{'data_type': 'gravity_x', 'label': 'X', 'color': 'r'},
                                  {'data_type': 'gravity_y', 'label': 'Y', 'color': 'g'},
                                  {'data_type': 'gravity_z', 'label': 'Z', 'color': 'b'}]
                }
            }
        ]
    },
    {
        'tab_title': 'Other Sensors',
        'layout': [
             {  'component_class': TimeSeriesPlotComponent,
                 'row': 0, 'col': 0,
                 'config': {
                     'title': 'Magnetic Field vs Time','xlabel': 'Time [s]','ylabel': 'µT',
                     'plot_height': 350, 'plot_width': 600, # Both constraints
                     'datasets': [{'data_type': 'mag_x', 'label': 'X', 'color': 'r'},
                                  {'data_type': 'mag_y', 'label': 'Y', 'color': 'g'},
                                  {'data_type': 'mag_z', 'label': 'Z', 'color': 'b'}]
                 }
             },
             # Example of a future component placement
             # {  'component_class': SomeOtherComponent, 'row': 1, 'col': 0, 'config': {...} }
        ]
    }
]

#####################################################################################################################
# End of customizable section
#####################################################################################################################


# --- GUI Manager ---
class GuiManager:
    #Manages the creation, layout, and updating of GUI components across tabs.
    def __init__(self, tab_widget: QTabWidget, tab_configs: List[Dict], data_buffers_ref: Dict[str, List[Tuple[float, float]]], device_config_ref: DeviceConfig):
        self.tab_widget = tab_widget
        self.tab_configs = tab_configs
        self.data_buffers_ref = data_buffers_ref
        self.device_config_ref = device_config_ref
        self.all_components: List[BaseGuiComponent] = [] # Flat list of all instantiated components
        self.active_missing_uuids: Set[str] = set() # Overall set of missing UUIDs

        self.create_gui_layout()

    def create_gui_layout(self):
        # Creates tabs and places components based on tab_configs.
        for tab_index, tab_config in enumerate(self.tab_configs):
            tab_title = tab_config.get('tab_title', f'Tab {tab_index + 1}')
            component_layout_defs = tab_config.get('layout', [])

            # Create the content widget and grid layout for the tab
            tab_content_widget = QWidget()
            grid_layout = QGridLayout(tab_content_widget)

            if not component_layout_defs:
                # Handle empty tab definition
                empty_label = QLabel(f"No components configured for '{tab_title}'")
                empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                grid_layout.addWidget(empty_label, 0, 0)
            else:
                # Instantiate and place components
                for comp_def in component_layout_defs:
                    comp_class: Type[BaseGuiComponent] = comp_def.get('component_class')
                    config = comp_def.get('config', {})
                    row = comp_def.get('row', 0)
                    col = comp_def.get('col', 0)
                    rowspan = comp_def.get('rowspan', 1)
                    colspan = comp_def.get('colspan', 1)

                    if not comp_class or not issubclass(comp_class, BaseGuiComponent):
                        logger.error(f"Invalid or missing 'component_class' in tab '{tab_title}', row {row}, col {col}. Skipping.")
                        # Optionally add a placeholder error widget
                        error_widget = QLabel(f"Error:\nInvalid Component\n(Row {row}, Col {col})")
                        error_widget.setStyleSheet("QLabel { color: red; border: 1px solid red; }")
                        grid_layout.addWidget(error_widget, row, col, rowspan, colspan)
                        continue

                    try:
                        # Instantiate the component
                        component_instance = comp_class(config, self.data_buffers_ref, self.device_config_ref)
                        self.all_components.append(component_instance)

                        # Add the component's widget to the grid
                        widget_to_add = component_instance.get_widget()
                        grid_layout.addWidget(widget_to_add, row, col, rowspan, colspan)
                        logger.debug(f"Added component {comp_class.__name__} to tab '{tab_title}' at ({row}, {col})")

                    except Exception as e:
                        logger.error(f"Failed to instantiate/add component {comp_class.__name__} in tab '{tab_title}': {e}", exc_info=True)
                        # Add a placeholder error widget
                        error_widget = QLabel(f"Error:\n{comp_class.__name__}\nFailed to load\n(Row {row}, Col {col})")
                        error_widget.setStyleSheet("QLabel { color: red; border: 1px solid red; }")
                        grid_layout.addWidget(error_widget, row, col, rowspan, colspan)


            # Make the grid layout the layout for the content widget
            tab_content_widget.setLayout(grid_layout)

            # Add the content widget (potentially within a scroll area) to the tab widget
            # Decide if scroll area is needed (e.g., based on total component size hint?) - simple approach: always use scroll area
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setWidget(tab_content_widget)
            self.tab_widget.addTab(scroll_area, tab_title)


    def update_all_components(self, current_relative_time: float, is_flowing: bool):
        # Calls the update method on all managed components.
        if plotting_paused or start_time is None: # Check global pause state here
            return
        for component in self.all_components:
            try:
                component.update_component(current_relative_time, is_flowing)
            except Exception as e:
                logger.error(f"Error updating component {type(component).__name__}: {e}", exc_info=True)
                # Optionally disable the component or show an error state visually

    def clear_all_components(self):
        # Calls the clear method on all managed components.
        logger.info("GuiManager clearing all components.")
        self.active_missing_uuids.clear() # Clear overall missing UUID state
        for component in self.all_components:
            try:
                component.clear_component()
                # Also reset any missing UUID state within the component itself
                component.handle_missing_uuids(set())
            except Exception as e:
                logger.error(f"Error clearing component {type(component).__name__}: {e}", exc_info=True)

    def notify_missing_uuids(self, missing_uuids_set: Set[str]):
        # Receives the set of all missing UUIDs and notifies relevant components.
        logger.info(f"GuiManager received missing UUIDs: {missing_uuids_set if missing_uuids_set else 'None'}")
        self.active_missing_uuids = missing_uuids_set

        for component in self.all_components:
            required_types = component.get_required_data_types()
            if not required_types:
                continue # Component doesn't need data, skip UUID check

            relevant_missing_uuids_for_comp = set()
            for data_type in required_types:
                uuid = self.device_config_ref.get_uuid_for_data_type(data_type)
                if uuid and uuid in self.active_missing_uuids:
                    relevant_missing_uuids_for_comp.add(uuid)

            try:
                # Pass only the UUIDs that are *both* required by the component *and* missing overall
                component.handle_missing_uuids(relevant_missing_uuids_for_comp)
            except Exception as e:
                 logger.error(f"Error notifying component {type(component).__name__} about missing UUIDs: {e}", exc_info=True)


# --- Bluetooth Protocol Handling ---
class GuiSignalEmitter(QObject):
    state_change_signal = pyqtSignal(str)
    scan_throbber_signal = pyqtSignal(str)
    connection_status_signal = pyqtSignal(str)
    show_error_signal = pyqtSignal(str, str) # title, message
    # Signal to emit the set of missing UUIDs
    missing_uuids_signal = pyqtSignal(set) # Emits set[str]

    def emit_state_change(self, new_state):
        self.state_change_signal.emit(new_state)

    def emit_scan_throbber(self, text):
        self.scan_throbber_signal.emit(text)

    def emit_connection_status(self, text):
        self.connection_status_signal.emit(text)

    def emit_show_error(self, title, message):
         self.show_error_signal.emit(title, message)

    def emit_missing_uuids(self, missing_set: set):
        self.missing_uuids_signal.emit(missing_set)


gui_emitter = GuiSignalEmitter()

def disconnected_callback(client_instance: BleakClient) -> None:
    logger.info(f"Device {client_instance.address} disconnected (detected by Bleak). Setting disconnected_event.")
    if loop and loop.is_running():
        loop.call_soon_threadsafe(disconnected_event.set)
    else:
        disconnected_event.set()

async def notification_handler(char_config: CharacteristicConfig, sender: int, data: bytearray) -> None:
    global last_received_time, start_time, data_buffers
    last_received_time = time.time()
    values = char_config.handler(data)
    if not values: return

    current_time_dt = datetime.datetime.now()
    if start_time is None:
        start_time = current_time_dt
        logger.info(f"First data received. Setting session start time: {start_time}")

    relative_time = (current_time_dt - start_time).total_seconds()

    for key, value in values.items():
        if key not in data_buffers:
            data_buffers[key] = [] # Use list, deque not strictly necessary here unless very high frequency/long runs
        data_buffers[key].append((relative_time, value))

async def find_device(device_config_current: DeviceConfig) -> Optional[BleakClient]: # Use current config
    """Finds the target device using BleakScanner."""
    found_event = asyncio.Event()
    target_device = None
    scan_cancelled = False

    def detection_callback(device, advertisement_data):
        nonlocal target_device, found_event
        if not found_event.is_set():
            # Use name and service UUID from the *current* device_config object
            target_service_lower = device_config_current.service_uuid.lower()
            advertised_uuids_lower = [u.lower() for u in advertisement_data.service_uuids]
            device_name = getattr(device, 'name', None)
            if device_name == device_config_current.name and target_service_lower in advertised_uuids_lower:
                target_device = device
                found_event.set()
                logger.info(f"Match found and event SET for: {device.name} ({device.address})")
            elif device_name and device_config_current.name and device_name.lower() == device_config_current.name.lower():
                 logger.debug(f"Found name match '{device_name}' but service UUID mismatch. Adv: {advertised_uuids_lower}, Target: {target_service_lower}")

    scanner = BleakScanner(
        detection_callback=detection_callback,
        service_uuids=[device_config_current.service_uuid] # Use current service UUID
    )
    logger.info(f"Starting scanner for {device_config_current.name} (Service: {device_config_current.service_uuid})...")
    gui_emitter.emit_scan_throbber("Scanning...")

    try:
        await scanner.start()
        try:
            await asyncio.wait_for(found_event.wait(), timeout=device_config_current.find_timeout) # Use current timeout
            if target_device:
                logger.info(f"Device found event confirmed for {target_device.name}")
            else:
                 logger.warning("Found event was set, but target_device is still None.")
        except asyncio.TimeoutError:
            logger.warning(f"Device '{device_config_current.name}' not found within {device_config_current.find_timeout} seconds (timeout).")
            target_device = None
        except asyncio.CancelledError:
             logger.info("Scan cancelled by user.")
             scan_cancelled = True
             target_device = None
    except BleakError as e:
         logger.error(f"Scanner start failed with BleakError: {e}")
         target_device = None
    except Exception as e:
         logger.error(f"Scanner start failed with unexpected error: {e}")
         target_device = None
    finally:
        logger.debug("Executing scanner stop block...")
        if 'scanner' in locals() and scanner is not None:
            try:
                logger.info(f"Attempting to stop scanner {scanner}...")
                await scanner.stop()
                logger.info(f"Scanner stop command issued for {scanner}.")
            except Exception as e:
                logger.warning(f"Error encountered while stopping scanner: {e}", exc_info=False)
        else:
            logger.debug("Scanner object not found or is None in finally block, skipping stop.")

    if scan_cancelled: raise asyncio.CancelledError
    return target_device


async def connection_task():
    global client, last_received_time, state, device_config # Access global device_config
    # Keep track of characteristics we successfully subscribe to
    found_char_configs: List[CharacteristicConfig] = []

    while state == "scanning":
        target_device = None
        found_char_configs = [] # Reset for each connection attempt cycle
        # --- Use the *current* global device_config state ---
        current_device_config = device_config
        # ----------------------------------------------------
        try:
            target_device = await find_device(current_device_config) # Pass the current config
        except asyncio.CancelledError:
            logger.info("connection_task: Scan was cancelled.")
            break
        except Exception as e:
            logger.error(f"Error during scanning phase: {e}")
            gui_emitter.emit_show_error("Scan Error", f"Scan failed: {e}")
            await asyncio.sleep(3)
            continue

        if not target_device:
            if state == "scanning":
                 logger.info(f"Device '{current_device_config.name}' not found, retrying scan in 3 seconds...")
                 gui_emitter.emit_scan_throbber(f"Device '{current_device_config.name}' not found. Retrying...")
                 await asyncio.sleep(3)
                 continue
            else:
                 logger.info("Scan stopped while waiting for device.")
                 break

        gui_emitter.emit_connection_status(f"Found {current_device_config.name}. Connecting...")
        client = None
        connection_successful = False
        for attempt in range(3):
             if state != "scanning": logger.info("Connection attempt aborted, state changed."); break
             try:
                  logger.info(f"Connecting (attempt {attempt + 1})...")
                  # Pass disconnected_callback here
                  client = BleakClient(target_device, disconnected_callback=disconnected_callback)
                  await client.connect(timeout=10.0)
                  logger.info("Connected successfully")
                  connection_successful = True
                  gui_emitter.emit_state_change("connected")
                  break
             except Exception as e:
                  logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                  if client:
                      try: await client.disconnect()
                      except Exception as disconnect_err: logger.warning(f"Error during disconnect after failed connection: {disconnect_err}")
                  client = None
                  if attempt < 2: await asyncio.sleep(2)

        if not connection_successful:
             logger.error("Max connection attempts reached or connection aborted.")
             gui_emitter.emit_missing_uuids(set()) # Ensure GUI resets if connection fails
             if state == "scanning":
                 logger.info("Retrying scan...")
                 gui_emitter.emit_scan_throbber("Connection failed. Retrying scan...")
                 await asyncio.sleep(1)
                 continue
             else:
                 logger.info("Exiting connection task as state is no longer 'scanning'.")
                 break

        # --- *** CHARACTERISTIC DISCOVERY AND NOTIFICATION START *** ---
        notification_errors = False
        missing_uuids = set()
        try:
            # Use the service UUID from the config used for *this* connection attempt
            logger.info(f"Checking characteristics for service {current_device_config.service_uuid}...")
            service = client.services.get_service(current_device_config.service_uuid)
            if not service:
                 logger.error(f"Service {current_device_config.service_uuid} not found on connected device.")
                 gui_emitter.emit_show_error("Connection Error", f"Service UUID\n{current_device_config.service_uuid}\nnot found on device.")
                 gui_emitter.emit_state_change("disconnecting") # Treat as fatal error for this connection
                 notification_errors = True # Skip notification attempts
            else:
                 logger.info("Service found. Checking configured characteristics...")
                 found_char_configs = [] # Reset list for this successful connection
                 # Use characteristics from the config used for *this* connection attempt
                 for char_config in current_device_config.characteristics:
                     bleak_char = service.get_characteristic(char_config.uuid)
                     if bleak_char:
                         logger.info(f"Characteristic found: {char_config.uuid}")
                         if "notify" in bleak_char.properties or "indicate" in bleak_char.properties:
                             found_char_configs.append(char_config)
                         else:
                             logger.warning(f"Characteristic {char_config.uuid} found but does not support notify/indicate.")
                             missing_uuids.add(char_config.uuid)
                     else:
                         logger.warning(f"Characteristic NOT FOUND: {char_config.uuid}")
                         missing_uuids.add(char_config.uuid)

                 # Emit the set of missing UUIDs to the GUI
                 gui_emitter.emit_missing_uuids(missing_uuids)

                 if not found_char_configs:
                      logger.error("No usable (found and notifiable) characteristics from config. Disconnecting.")
                      gui_emitter.emit_show_error("Connection Error", "None of the configured characteristics\nwere found or support notifications.")
                      gui_emitter.emit_state_change("disconnecting")
                      notification_errors = True
                 else:
                    logger.info(f"Starting notifications for {len(found_char_configs)} found characteristics...")
                    notify_tasks = []
                    for char_config in found_char_configs: # Iterate through FOUND characteristics
                        handler_with_char = partial(notification_handler, char_config)
                        notify_tasks.append(client.start_notify(char_config.uuid, handler_with_char))

                    results = await asyncio.gather(*notify_tasks, return_exceptions=True)
                    all_notifications_started = True
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            char_uuid = found_char_configs[i].uuid # Use the correct list here
                            logger.error(f"Failed to start notification for {char_uuid}: {result}")
                            all_notifications_started = False; notification_errors = True
                            missing_uuids.add(char_uuid) # Add to missing if start_notify failed

                    if not all_notifications_started:
                        logger.error("Could not start all required notifications. Disconnecting.")
                        gui_emitter.emit_missing_uuids(missing_uuids) # Update GUI with newly failed ones
                        gui_emitter.emit_state_change("disconnecting")
                    else:
                        logger.info("Notifications started successfully. Listening...")
                        last_received_time = time.time()
                        disconnected_event.clear()
                        while state == "connected":
                             try:
                                 # Use data timeout from the config used for *this* connection
                                 await asyncio.wait_for(disconnected_event.wait(), timeout=current_device_config.data_timeout + 1.0)
                                 logger.info("Disconnected event received while listening.")
                                 gui_emitter.emit_state_change("disconnecting")
                                 break
                             except asyncio.TimeoutError:
                                 current_time = time.time()
                                 # Use data timeout from the config used for *this* connection
                                 if current_time - last_received_time > current_device_config.data_timeout:
                                     logger.warning(f"No data received for {current_time - last_received_time:.1f}s (timeout: {current_device_config.data_timeout}s). Assuming disconnect.")
                                     gui_emitter.emit_state_change("disconnecting")
                                     break
                                 else: continue # Continue listening
                             except asyncio.CancelledError:
                                 logger.info("Listening loop cancelled.")
                                 gui_emitter.emit_state_change("disconnecting"); raise
                             except Exception as e:
                                 logger.error(f"Error during notification listening loop: {e}")
                                 gui_emitter.emit_state_change("disconnecting"); notification_errors = True; break


        except asyncio.CancelledError:
             logger.info("Notification setup or listening task was cancelled.")
             if state == "connected": gui_emitter.emit_state_change("disconnecting")
        except Exception as e: # Catch BleakError and others
             logger.error(f"Error during characteristic check or notification handling: {e}")
             gui_emitter.emit_state_change("disconnecting"); notification_errors = True
        finally:
            logger.info("Performing cleanup for connection task...")
            local_client = client; client = None # Use local var for safety
            stop_notify_errors = []
            if local_client:
                 is_conn = False
                 try: is_conn = local_client.is_connected
                 except Exception as check_err: logger.warning(f"Error checking client connection status during cleanup: {check_err}")

                 if is_conn:
                      logger.info("Attempting to stop notifications and disconnect client...")
                      try:
                          stop_tasks = []
                          # Iterate through FOUND characteristics only (using the list from this connection attempt)
                          for char_config in found_char_configs:
                              try:
                                  # Use service UUID from the config active during *this* connection
                                  service = local_client.services.get_service(current_device_config.service_uuid)
                                  if service and service.get_characteristic(char_config.uuid):
                                      logger.debug(f"Preparing stop_notify for {char_config.uuid}")
                                      stop_tasks.append(local_client.stop_notify(char_config.uuid))
                                  else: logger.debug(f"Characteristic {char_config.uuid} not found/unavailable during cleanup, skipping stop_notify.")
                              except Exception as notify_stop_err: logger.warning(f"Error preparing stop_notify for {char_config.uuid}: {notify_stop_err}")

                          if stop_tasks:
                               results = await asyncio.gather(*stop_tasks, return_exceptions=True)
                               logger.info(f"Notifications stop attempts completed for {len(stop_tasks)} characteristics.")
                               for i, result in enumerate(results):
                                   if isinstance(result, Exception): logger.warning(f"Error stopping notification {i} ({found_char_configs[i].uuid}): {result}"); stop_notify_errors.append(result)
                          else: logger.info("No notifications needed stopping or could be prepared.")
                      except Exception as e: logger.warning(f"General error during stop_notify phase: {e}"); stop_notify_errors.append(e)

                      try:
                          await asyncio.wait_for(local_client.disconnect(), timeout=5.0)
                          if not stop_notify_errors: logger.info("Client disconnected.")
                          else: logger.warning("Client disconnected, but errors occurred during notification cleanup.")
                      except Exception as e: logger.error(f"Error during client disconnect: {e}")
                 else: logger.info("Client object existed but was not connected during cleanup.")
            else: logger.info("No active client object to cleanup.")

            # Reset missing UUIDs in GUI when disconnected or connection failed
            if state in ["connected", "disconnecting"] or not connection_successful:
                 gui_emitter.emit_missing_uuids(set())

            if state in ["connected", "disconnecting"]:
                 logger.info("Signalling state change to idle after cleanup.")
                 gui_emitter.emit_state_change("idle")

            disconnected_event.clear()
            found_char_configs = [] # Clear the list specific to this attempt

        # Check global state before looping back to scan
        if state != "scanning":
            logger.info(f"Exiting connection_task as state is '{state}'.")
            break

    logger.info("Connection task loop finished.")


async def main_async():
    """The main entry point for the asyncio part."""
    global loop, current_task
    loop = asyncio.get_running_loop()
    logger.info("Asyncio loop running.")

    while not stop_flag:
        await asyncio.sleep(0.1) # Keep asyncio loop responsive

    logger.info("Asyncio loop stopping (stop_flag is True).")

    if current_task and not current_task.done():
        logger.info("main_async: Waiting for connection_task cancellation/cleanup to complete...")
        try:
            await current_task # Wait for the task to finish its cleanup after cancellation
            logger.info("main_async: connection_task completed its run after stop signal.")
        except asyncio.CancelledError:
            logger.info("main_async: connection_task finished handling cancellation.")
        except Exception as e:
            logger.error(f"main_async: Error occurred while awaiting connection_task finalization: {e}", exc_info=True)
    else:
        logger.info("main_async: No active connection_task or task already done upon exit.")

    logger.info("main_async function finished.")


# --- PyQt6 Main Application Window ---

class LEDWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(20, 20)
        self._color = QColor("red")
    def set_color(self, color_name: str):
        self._color = QColor(color_name); self.update()
    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(self._color)); painter.setPen(QPen(Qt.PenStyle.NoPen))
        rect = self.rect(); diameter = min(rect.width(), rect.height()) - 4
        painter.drawEllipse(rect.center(), diameter // 2, diameter // 2)


class MainWindow(QMainWindow):
    # Signal to request an update (can be triggered by timer or other events)
    request_plot_update_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modular BLE Data GUI") # Updated title
        self.setGeometry(100, 100, 1200, 850)

        # --- Capture State ---
        self.is_capturing = False
        self.capture_start_relative_time: Optional[float] = None
        self.capture_t0_absolute: Optional[datetime.datetime] = None
        self.capture_output_base_dir: Optional[str] = None
        self.capture_timestamp: Optional[str] = None

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # --- Top Button Bar---
        self.button_bar = QWidget()
        self.button_layout = QHBoxLayout(self.button_bar)
        self.button_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.led_indicator = LEDWidget(); self.button_layout.addWidget(self.led_indicator)
        self.device_label = QLabel("Target:")
        self.button_layout.addWidget(self.device_label)
        self.device_combo = QComboBox()
        self.device_combo.addItems(AVAILABLE_DEVICE_NAMES)
        self.device_combo.setFixedWidth(100)
        initial_device_index = self.device_combo.findText(device_config.name)
        if initial_device_index != -1: self.device_combo.setCurrentIndex(initial_device_index)
        self.device_combo.currentTextChanged.connect(self.update_target_device)
        self.button_layout.addWidget(self.device_combo)
        self.scan_button = QPushButton("Start Scan"); self.scan_button.clicked.connect(self.toggle_scan); self.button_layout.addWidget(self.scan_button)
        self.pause_resume_button = QPushButton("Pause Plotting"); self.pause_resume_button.setEnabled(False); self.pause_resume_button.clicked.connect(self.toggle_pause_resume); self.button_layout.addWidget(self.pause_resume_button)
        self.capture_button = QPushButton("Start Capture"); self.capture_button.setEnabled(False); self.capture_button.clicked.connect(self.toggle_capture); self.button_layout.addWidget(self.capture_button)
        self.clear_button = QPushButton("Clear GUI"); self.clear_button.clicked.connect(self.clear_gui_action); self.button_layout.addWidget(self.clear_button) # Renamed slightly
        self.status_label = QLabel("On Standby"); self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter); self.status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred); self.button_layout.addWidget(self.status_label)
        self.main_layout.addWidget(self.button_bar)

        # --- Tab Area (Managed by GuiManager) ---
        self.tab_widget = QTabWidget()
        self.tab_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Instantiate the new GuiManager
        self.gui_manager = GuiManager(self.tab_widget, tab_configs, data_buffers, device_config)
        self.main_layout.addWidget(self.tab_widget)

        # --- Bottom Control Bar ---
        self.bottom_bar = QWidget()
        self.bottom_layout = QHBoxLayout(self.bottom_bar)
        self.bottom_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.flowing_mode_check = QCheckBox("Flowing Mode"); self.flowing_mode_check.setChecked(False); self.bottom_layout.addWidget(self.flowing_mode_check)
        self.interval_label = QLabel("Interval (s):"); self.bottom_layout.addWidget(self.interval_label)
        self.interval_entry = QLineEdit(str(flowing_interval)); self.interval_entry.setFixedWidth(50); self.bottom_layout.addWidget(self.interval_entry)
        self.apply_interval_button = QPushButton("Apply Interval"); self.apply_interval_button.clicked.connect(self.apply_interval); self.bottom_layout.addWidget(self.apply_interval_button)
        self.data_log_check = QCheckBox("Log Raw Data to Console"); self.data_log_check.setChecked(True); self.data_log_check.stateChanged.connect(self.toggle_data_log); self.bottom_layout.addWidget(self.data_log_check)
        self.bottom_layout.addStretch(1)
        self.main_layout.addWidget(self.bottom_bar)

        # --- Log Text Box ---
        self.log_text_box = QTextEdit()
        self.log_text_box.setReadOnly(True)
        self.log_text_box.setMaximumHeight(150)
        self.log_text_box.setMinimumHeight(50)
        self.log_text_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.log_text_box.document().setMaximumBlockCount(1000)
        self.main_layout.addWidget(self.log_text_box)

        # --- Setup Logging to GUI ---
        self.log_handler = QtLogHandler()
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_format)
        self.log_handler.setFormatter(formatter)
        self.log_handler.setLevel(logging.INFO)
        self.log_handler.log_signal.connect(self.append_log_message)
        logging.getLogger().addHandler(self.log_handler)

        # --- Timers and Signals ---
        self.plot_update_timer = QTimer(self)
        self.plot_update_timer.setInterval(50) # ~20 FPS
        self.plot_update_timer.timeout.connect(self.trigger_gui_update) # Renamed target slot
        self.plot_update_timer.start()

        self.scan_throbber_timer = QTimer(self)
        self.scan_throbber_timer.setInterval(150)
        self.scan_throbber_timer.timeout.connect(self.animate_scan_throbber)
        self.throbber_chars = ["|", "/", "-", "\\"]
        self.throbber_index = 0

        # Connect signals from GuiEmitter
        gui_emitter.state_change_signal.connect(self.handle_state_change)
        gui_emitter.scan_throbber_signal.connect(self.update_scan_status)
        gui_emitter.connection_status_signal.connect(self.update_connection_status)
        gui_emitter.show_error_signal.connect(self.show_message_box)
        # Connect the missing UUIDs signal to the GuiManager's slot
        gui_emitter.missing_uuids_signal.connect(self.gui_manager.notify_missing_uuids)
        # Connect internal request signal to the actual update slot
        self.request_plot_update_signal.connect(self._update_gui_now)

        self.handle_state_change("idle") # Initialize state

    # --- Slot to Append Log Messages ---
    def append_log_message(self, message):
        self.log_text_box.append(message)

    # --- GUI Update Triggering ---
    def trigger_gui_update(self):
        """Slot called by the timer."""
        self.request_plot_update_signal.emit()

    def _update_gui_now(self):
        """Slot that performs the actual GUI component update."""
        if start_time is not None: # Check if connection established and data received
            current_relative = (datetime.datetime.now() - start_time).total_seconds()
            is_flowing = self.flowing_mode_check.isChecked()
            # Delegate update to the GuiManager
            self.gui_manager.update_all_components(current_relative, is_flowing)
        # No else needed: GuiManager.update_all_components checks plotting_paused internally

    # --- Scan Animation ---
    def animate_scan_throbber(self):
        if state == "scanning":
            text = "Scanning... " + self.throbber_chars[self.throbber_index]
            self.status_label.setText(text)
            self.throbber_index = (self.throbber_index + 1) % len(self.throbber_chars)
        else:
            self.scan_throbber_timer.stop()

    # --- GUI Action Slots ---

    # --- Handler for Device Dropdown ---
    def update_target_device(self, selected_name: str):
        global device_config # Need to modify the global object
        if device_config.name != selected_name:
            logger.info(f"Target device changed via GUI: {selected_name}")
            # Update the name in the global device_config object
            device_config.update_name(selected_name)
            logger.info("Device config name updated.")
            # If connected, changing device implies disconnect/rescan needed.
            # update the config; user needs to manually restart scan.

    # State Change Handling
    def handle_state_change(self, new_state: str):
        global state, plotting_paused, start_time # Added start_time reset
        logger.info(f"GUI received state change: {new_state}")
        previous_state = state
        state = new_state # Update global state

        if new_state != "scanning" and self.scan_throbber_timer.isActive():
            self.scan_throbber_timer.stop()

        is_idle = (new_state == "idle")
        self.device_combo.setEnabled(is_idle) # Can only change device when idle
        self.scan_button.setEnabled(True) # Scan/Disconnect always enabled except during disconnect transition

        if new_state == "idle":
            self.scan_button.setText("Start Scan")
            self.led_indicator.set_color("red"); self.status_label.setText("On Standby")
            self.pause_resume_button.setEnabled(False); self.pause_resume_button.setText("Pause Plotting")
            self.capture_button.setEnabled(False); self.capture_button.setText("Start Capture")
            plotting_paused = True # Set plots to paused state logically

            # Automatically clear GUI/state when becoming idle
            logger.info("State changed to idle. Automatically clearing GUI and state.")
            self.clear_gui_action(confirm=False) # Handles GUI components, buffers, start_time, missing_uuids

            if self.is_capturing: # Should be false if clear_gui_action ran correctly
                 logger.warning("Capture was active when state became idle (likely disconnect). Files were NOT generated automatically by clear.")

        elif new_state == "scanning":
            self.scan_button.setText("Stop Scan")
            self.led_indicator.set_color("orange"); self.throbber_index = 0
            if not self.scan_throbber_timer.isActive(): self.scan_throbber_timer.start()
            self.pause_resume_button.setEnabled(False)
            self.capture_button.setEnabled(False)
            # Clearing state now happens in toggle_scan before starting scan

        elif new_state == "connected":
            # Use the name from the *current* global device_config
            self.scan_button.setText("Disconnect")
            self.led_indicator.set_color("lightgreen"); self.status_label.setText(f"Connected to: {device_config.name}")
            # Enable pause/resume ONLY IF NOT currently capturing
            if not self.is_capturing:
                self.pause_resume_button.setEnabled(True)
            self.pause_resume_button.setText("Pause Plotting") # Set text regardless
            self.capture_button.setEnabled(True) # Capture can always be started if connected
            plotting_paused = False # Resume on connect

        elif new_state == "disconnecting":
            self.scan_button.setText("Disconnecting..."); self.scan_button.setEnabled(False)
            self.led_indicator.set_color("red"); self.status_label.setText("Status: Disconnecting...")
            self.pause_resume_button.setEnabled(False); self.pause_resume_button.setText("Pause Plotting")
            self.capture_button.setEnabled(False)
            plotting_paused = True

    # Scan/Connection Status Updates
    def update_scan_status(self, text: str):
         if state == "scanning": self.status_label.setText(text)
    def update_connection_status(self, text: str):
         if state != "connected" and state != "idle": self.status_label.setText(text)
    def show_message_box(self, title: str, message: str):
        QMessageBox.warning(self, title, message)

    # Scan/Connect/Disconnect Logic (clear GUI/state on start)
    def toggle_scan(self):
        global current_task, loop, state, data_buffers, start_time
        if state == "idle":
            if loop and loop.is_running():
                # Clear GUI, buffers, state BEFORE starting scan
                logger.info("Clearing state before starting scan...")
                self.clear_gui_action(confirm=False) # Ensures everything is reset

                self.handle_state_change("scanning")
                current_task = loop.create_task(connection_task()) # Uses global device_config implicitly
            else: logger.error("Asyncio loop not running!"); self.show_message_box("Error", "Asyncio loop is not running.")
        elif state == "scanning":
            # Request cancellation, state change handled via callback/finally block
            if current_task and not current_task.done():
                logger.info("Requesting scan cancellation...")
                # Ensure cancellation is requested in the loop's thread
                future = asyncio.run_coroutine_threadsafe(self.cancel_and_wait_task(current_task), loop)
                try:
                    future.result(timeout=0.1) # Brief wait for initiation
                except TimeoutError: pass # Ignore timeout, cancellation sent
                except Exception as e: logger.error(f"Error initiating task cancel: {e}")
                # GUI state will change to idle via the task's cleanup calling emit_state_change("idle")
            else:
                logger.warning("Stop scan requested, but no task was running/done.")
                self.handle_state_change("idle") # Force idle if no task
            current_task = None # Clear reference after requesting cancel
        elif state == "connected":
            # Request disconnect, state change handled via callback/finally block
            if loop and client and client.is_connected:
                logger.info("Requesting disconnection via disconnected_event...")
                loop.call_soon_threadsafe(disconnected_event.set)
                # GUI state change will happen via connection_task cleanup
            elif loop and current_task and not current_task.done():
                logger.info("Requesting disconnect via task cancellation...")
                future = asyncio.run_coroutine_threadsafe(self.cancel_and_wait_task(current_task), loop)
                try: future.result(timeout=0.1)
                except TimeoutError: pass
                except Exception as e: logger.error(f"Error initiating task cancel for disconnect: {e}")
                 # GUI state change will happen via connection_task cleanup
            else:
                logger.warning("Disconnect requested but no active connection/task found.")
                self.handle_state_change("idle") # Force idle

    # Pause/Resume Plotting (global flag logic)
    def toggle_pause_resume(self):
        global plotting_paused
        if not self.pause_resume_button.isEnabled():
            logger.warning("Pause/Resume toggled while button disabled. Ignoring.")
            return

        plotting_paused = not plotting_paused
        self.pause_resume_button.setText("Resume Plotting" if plotting_paused else "Pause Plotting")
        logger.info(f"Plotting {'paused' if plotting_paused else 'resumed'}")
        # If resuming, trigger an immediate GUI update
        if not plotting_paused:
            self.trigger_gui_update()

    # --- Capture Start/Stop Logic (uses global start_time) ---
    def toggle_capture(self):
        global start_time
        if not self.is_capturing:
            # Start Capture
            if state != "connected" or start_time is None:
                self.show_message_box("Capture Error", "Must be connected and receiving data.")
                return

            # --- Create directory FIRST ---
            self.capture_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_log_dir = "Logs"
            self.capture_output_base_dir = os.path.join(base_log_dir, self.capture_timestamp)
            try:
                os.makedirs(self.capture_output_base_dir, exist_ok=True)
                logger.info(f"Capture directory created: {self.capture_output_base_dir}")
            except Exception as e:
                logger.error(f"Failed to create capture dir: {e}")
                self.show_message_box("Capture Error", f"Failed to create directory:\n{e}")
                self.capture_output_base_dir = None; self.capture_timestamp = None
                return # Don't proceed

            # --- Set capture state variables ---
            self.is_capturing = True
            self.capture_button.setText("Stop Capture && Export")
            self.capture_t0_absolute = datetime.datetime.now()
            # Calculate start relative to the session's absolute start_time
            self.capture_start_relative_time = (self.capture_t0_absolute - start_time).total_seconds()

            # --- Disable Pause/Resume Button ---
            self.pause_resume_button.setEnabled(False)
            logger.info("Pause/Resume plotting disabled during capture.")

            logger.info(f"Capture started. Abs t0: {self.capture_t0_absolute}, Rel t0: {self.capture_start_relative_time:.3f}s.")

        else:
            # Stop Capture and Generate
            self.stop_and_generate_files()

    # --- Stop Capture & File Generation Helper ---
    def stop_and_generate_files(self):
        if not self.is_capturing or start_time is None: # Also check start_time exists
            logger.warning("stop_and_generate called but capture inactive or start_time missing.")
            if state == "connected": self.pause_resume_button.setEnabled(True)
            else: self.pause_resume_button.setEnabled(False)
            # Reset state just in case
            self.is_capturing = False
            self.capture_button.setText("Start Capture")
            self.capture_button.setEnabled(state == "connected")
            return

        logger.info("Stopping capture, generating PGF & CSV.")
        output_dir = self.capture_output_base_dir
        start_rel_time = self.capture_start_relative_time
        # Calculate end relative to the session's absolute start_time
        capture_end_relative_time = (datetime.datetime.now() - start_time).total_seconds()

        # --- Reset capture state FIRST ---
        self.is_capturing = False
        self.capture_button.setText("Start Capture")
        self.capture_button.setEnabled(state == "connected") # Enable only if still connected

        # --- Re-enable Pause/Resume Button ONLY if connected ---
        if state == "connected":
            self.pause_resume_button.setEnabled(True)
            logger.info("Pause/Resume plotting re-enabled after capture (still connected).")
        else:
            self.pause_resume_button.setEnabled(False)
            logger.info("Pause/Resume plotting remains disabled after capture (not connected).")


        # --- File Generation Logic ---
        if output_dir and start_rel_time is not None: # End time calculated above
            if not data_buffers:
                 logger.warning("No data captured during the active period. Skipping PGF/CSV generation.")
                 # Reset vars even if generation skipped
                 self.capture_output_base_dir = None; self.capture_start_relative_time = None
                 self.capture_t0_absolute = None; self.capture_timestamp = None
                 return # Skip generation

            pgf_subdir = os.path.join(output_dir, "pgf_plots")
            csv_subdir = os.path.join(output_dir, "csv_files")
            try: os.makedirs(pgf_subdir, exist_ok=True); os.makedirs(csv_subdir, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create PGF/CSV subdirs: {e}")
                self.show_message_box("File Gen Error", f"Could not create output subdirs:\n{e}")
                # Reset vars on failure
                self.capture_output_base_dir = None; self.capture_start_relative_time = None
                self.capture_t0_absolute = None; self.capture_timestamp = None
                return

            gen_errors = []
            try:
                # Pass necessary info: output dir, start rel time (for filtering/offset)
                self.generate_pgf_plots_from_buffer(pgf_subdir, start_rel_time)
            except Exception as e: logger.error(f"PGF generation failed: {e}", exc_info=True); gen_errors.append(f"PGF: {e}")
            try:
                # Pass necessary info: output dir, start/end rel time (for filtering), start rel time (for offset)
                self.generate_csv_files_from_buffer(csv_subdir, start_rel_time, capture_end_relative_time, start_rel_time)
            except Exception as e: logger.error(f"CSV generation failed: {e}", exc_info=True); gen_errors.append(f"CSV: {e}")

            if not gen_errors: self.show_message_box("Generation Complete", f"Files generated in:\n{output_dir}")
            else: self.show_message_box("Generation Errors", f"Completed with errors in:\n{output_dir}\n\n" + "\n".join(gen_errors))

        else:
             reason = ""
             if not output_dir: reason += " Output dir missing."
             if start_rel_time is None: reason += " Start time missing."
             logger.error(f"Cannot generate files:{reason}")
             self.show_message_box("File Gen Error", f"Internal error:{reason}")

        # --- Final reset of capture vars ---
        self.capture_output_base_dir = None
        self.capture_start_relative_time = None
        self.capture_t0_absolute = None
        self.capture_timestamp = None


    # --- PGF Generation (Uses tab_configs for structure, GuiManager for missing UUIDs) ---
    def generate_pgf_plots_from_buffer(self, pgf_dir: str, capture_start_relative_time: float):
        global data_buffers, tab_configs # Use new tab_configs

        logger.info(f"Generating PGF plots (t=0 at capture start, t_offset={capture_start_relative_time:.3f}s). Dir: {pgf_dir}")
        if not data_buffers: logger.warning("Data buffer empty, skipping PGF generation."); return

        try:
            plt.style.use('science')
            plt.rcParams.update({'text.usetex': False, 'figure.figsize': [5.5, 3.5], 'legend.fontsize': 9,
                                 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.titlesize': 11})
        except Exception as style_err:
            logger.warning(f"Could not apply 'science' style: {style_err}. Using default.")
            plt.rcParams.update({'figure.figsize': [6.0, 4.0]})

        gen_success = False
        # Iterate through the new tab_configs structure
        for tab_config in tab_configs:
            for comp_def in tab_config.get('layout', []):
                # Only process components that are TimeSeriesPlotComponent
                if comp_def.get('component_class') != TimeSeriesPlotComponent:
                    continue

                plot_config = comp_def.get('config', {}) # Get the config for this plot instance
                plot_title = plot_config.get('title', 'UntitledPlot')
                datasets = plot_config.get('datasets', [])
                if not datasets: continue

                # Check if any required UUIDs for *this plot* were missing during connection
                required_uuids_for_plot = set()
                for ds in datasets:
                    uuid = self.gui_manager.device_config_ref.get_uuid_for_data_type(ds['data_type'])
                    if uuid: required_uuids_for_plot.add(uuid)

                # Use the currently stored missing UUIDs from GuiManager
                missing_uuids_for_this_plot = required_uuids_for_plot.intersection(self.gui_manager.active_missing_uuids)

                if missing_uuids_for_this_plot:
                    logger.warning(f"Skipping PGF for '{plot_title}' as it depends on missing UUID(s): {missing_uuids_for_this_plot}")
                    continue # Skip this plot

                # --- Plotting logic (same as before) ---
                fig, ax = plt.subplots()
                ax.set_title(plot_config.get('title', 'Plot'));
                ax.set_xlabel(plot_config.get('xlabel', 'Time [s]'));
                ax.set_ylabel(plot_config.get('ylabel', 'Value'))
                plot_created = False
                for dataset in datasets:
                    data_type = dataset['data_type']
                    if data_type in data_buffers and data_buffers[data_type]:
                        full_data = data_buffers[data_type]
                        # Filter data based on capture start time *relative to session start*
                        plot_data = [(item[0] - capture_start_relative_time, item[1])
                                     for item in full_data if item[0] >= capture_start_relative_time]
                        if plot_data:
                            try:
                                times_rel_capture = [p[0] for p in plot_data]
                                values = [p[1] for p in plot_data]
                                ax.plot(times_rel_capture, values, label=dataset.get('label', data_type), color=dataset.get('color', 'k'), linewidth=1.2)
                                plot_created = True
                            except Exception as plot_err: logger.error(f"Error plotting {data_type} for PGF '{plot_title}': {plot_err}")

                if plot_created:
                    ax.legend(); ax.grid(True, linestyle='--', alpha=0.6); fig.tight_layout(pad=0.5)
                    safe_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in plot_title).rstrip().replace(' ', '_')
                    prefix = f"{self.capture_timestamp}_" if self.capture_timestamp else ""
                    pgf_filename = f"{prefix}{safe_title}.pgf"
                    pgf_filepath = os.path.join(pgf_dir, pgf_filename)
                    try: fig.savefig(pgf_filepath, bbox_inches='tight'); logger.info(f"Saved PGF: {pgf_filename}"); gen_success = True
                    except Exception as save_err: logger.error(f"Error saving PGF {pgf_filename}: {save_err}"); raise RuntimeError(f"Save PGF failed: {save_err}") from save_err
                else: logger.info(f"Skipping PGF '{plot_title}' (no data in capture window).")
                plt.close(fig) # Close the figure to release memory

        if gen_success: logger.info(f"PGF generation finished. Dir: {pgf_dir}")
        else: logger.warning("PGF done, but no plots saved (no data / missing UUIDs?).")


    # --- CSV Generation (Uses tab_configs, GuiManager for missing UUIDs) ---
    def generate_csv_files_from_buffer(self, csv_dir: str, filter_start_rel_time: float, filter_end_rel_time: float, time_offset: float):
        global data_buffers, tab_configs # Use new tab_configs

        logger.info(f"Generating CSVs (data {filter_start_rel_time:.3f}s-{filter_end_rel_time:.3f}s rel session, t=0 at capture start offset={time_offset:.3f}s). Dir: {csv_dir}")
        if not data_buffers: logger.warning("Data buffer empty, skipping CSV generation."); return

        def get_series(dt, start, end):
             # Exclude series if its source UUID was missing during the connection
             uuid = self.gui_manager.device_config_ref.get_uuid_for_data_type(dt)
             # Use the currently stored missing UUIDs from GuiManager
             if uuid and uuid in self.gui_manager.active_missing_uuids:
                 logger.debug(f"Excluding series '{dt}' from CSV (UUID {uuid} was missing).")
                 return None

             if dt in data_buffers and data_buffers[dt]:
                 # Filter data based on time relative to the *session* start
                 filt = [item for item in data_buffers[dt] if start <= item[0] <= end]
                 if filt:
                     try:
                         # Index is still relative to session start for alignment
                         return pd.Series([i[1] for i in filt], index=pd.Index([i[0] for i in filt], name='TimeRelSession'), name=dt)
                     except Exception as e:
                         logger.error(f"Error creating Pandas Series for {dt}: {e}")
             return None # Return None if no data in range or buffer empty/missing

        master_gen = False
        # Generate Master CSV per Tab
        for tab_index, tab_config in enumerate(tab_configs):
            tab_title = tab_config.get('tab_title', f"Tab_{tab_index}")
            logger.info(f"Processing Master CSV for tab: '{tab_title}'")

            # Collect all unique data types used by TimeSeriesPlotComponents in this tab
            tab_types = set()
            for comp_def in tab_config.get('layout', []):
                if comp_def.get('component_class') == TimeSeriesPlotComponent:
                    plot_config = comp_def.get('config', {})
                    for ds in plot_config.get('datasets', []):
                        tab_types.add(ds['data_type'])

            if not tab_types: logger.warning(f"Skipping Master CSV '{tab_title}': No plottable data types defined."); continue

            series_list = [s for dt in sorted(list(tab_types)) if (s := get_series(dt, filter_start_rel_time, filter_end_rel_time)) is not None]
            if not series_list: logger.warning(f"Skipping Master CSV '{tab_title}': No valid series data found in window."); continue

            try:
                # Concatenate using the TimeRelSession index for alignment
                master_df = pd.concat(series_list, axis=1, join='outer').sort_index()
                # Insert the adjusted time column (relative to capture start)
                master_df.insert(0, 'Master Time (s)', master_df.index - time_offset)
                safe_t_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in tab_title).rstrip().replace(' ', '_')
                prefix = f"{self.capture_timestamp}_" if self.capture_timestamp else ""
                csv_fname = f"{prefix}master_tab_{safe_t_title}.csv"
                csv_fpath = os.path.join(csv_dir, csv_fname)
                master_df.to_csv(csv_fpath, index=False, float_format='%.6f') # Don't write the TimeRelSession index
                logger.info(f"Saved Master CSV: {csv_fname}"); master_gen = True
            except Exception as e: logger.error(f"Error generating Master CSV '{tab_title}': {e}", exc_info=True); raise RuntimeError(f"Master CSV generation failed: {e}") from e

        indiv_gen = False
        # Generate Individual CSV per Plot Component
        for tab_index, tab_config in enumerate(tab_configs):
             for comp_index, comp_def in enumerate(tab_config.get('layout', [])):
                # Only process components that are TimeSeriesPlotComponent
                if comp_def.get('component_class') != TimeSeriesPlotComponent:
                    continue

                plot_config = comp_def.get('config', {})
                plot_title = plot_config.get('title', f"Tab{tab_index}_Plot{comp_index}")
                datasets = plot_config.get('datasets', [])
                if not datasets: logger.warning(f"Skipping Individual CSV '{plot_title}': No datasets defined."); continue

                logger.info(f"Processing Individual CSV for plot: '{plot_title}'")

                # Get only series relevant to this specific plot
                series_list = [s for ds in datasets if (s := get_series(ds['data_type'], filter_start_rel_time, filter_end_rel_time)) is not None]
                if not series_list: logger.warning(f"Skipping Individual CSV '{plot_title}': No valid series data found in window."); continue

                try:
                    plot_df = pd.concat(series_list, axis=1, join='outer').sort_index()
                    # Insert adjusted time column
                    plot_df.insert(0, 'Time (s)', plot_df.index - time_offset)
                    safe_p_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in plot_title).rstrip().replace(' ', '_')
                    prefix = f"{self.capture_timestamp}_" if self.capture_timestamp else ""
                    csv_fname = f"{prefix}plot_{safe_p_title}.csv"
                    csv_fpath = os.path.join(csv_dir, csv_fname)
                    plot_df.to_csv(csv_fpath, index=False, float_format='%.6f') # Don't write the TimeRelSession index
                    logger.info(f"Saved Individual CSV: {csv_fname}"); indiv_gen = True
                except Exception as e: logger.error(f"Error generating Individual CSV '{plot_title}': {e}", exc_info=True); raise RuntimeError(f"Individual CSV generation failed: {e}") from e

        if master_gen or indiv_gen: logger.info(f"CSV generation finished. Dir: {csv_dir}")
        else: logger.warning("CSV generation done, but no files were saved (no valid data in window?).")


    # --- Clear GUI Action ---
    def clear_gui_action(self, confirm=True):
        global data_buffers, start_time
        logger.info("Attempting to clear GUI components and data.")
        do_clear = False
        if confirm:
            question = "Clear all displayed data, reset UUID status, and clear internal buffers?"
            if self.is_capturing:
                question = "Capture is active. Clear all data (stopping capture WITHOUT exporting)?\nAlso resets UUID status and clears buffers."

            reply = QMessageBox.question(self, 'Clear GUI & Data', question,
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes: do_clear = True
        else: do_clear = True # Clear without confirm (used internally)

        if do_clear:
            logger.info("Confirmed clearing GUI, data buffers, resetting start time, and UUID status.")

            # --- Stop capture if active ---
            if self.is_capturing:
                 logger.warning("Capture active during clear. Stopping capture WITHOUT generating files.")
                 self.is_capturing = False
                 self.capture_button.setText("Start Capture")
                 self.capture_button.setEnabled(state == "connected") # Update button state
                 # Also reset pause/resume button state appropriately
                 if state == "connected": self.pause_resume_button.setEnabled(True)
                 else: self.pause_resume_button.setEnabled(False)
                 # Clear capture temp vars
                 self.capture_output_base_dir = None; self.capture_start_relative_time = None
                 self.capture_t0_absolute = None; self.capture_timestamp = None

            # --- Clear data and state ---
            data_buffers.clear()
            start_time = None
            # Delegate clearing visual components and their UUID state to GuiManager
            self.gui_manager.clear_all_components()


    # --- Apply Interval ---
    def apply_interval(self):
        global flowing_interval
        try:
            new_interval = float(self.interval_entry.text())
            if new_interval > 0:
                flowing_interval = new_interval
                logger.info(f"Flowing interval updated to {new_interval}s")
                # Trigger GUI update immediately if flowing mode is active
                if self.flowing_mode_check.isChecked():
                    self._update_gui_now()
            else: self.show_message_box("Invalid Input", "Interval must be positive.")
        except ValueError: self.show_message_box("Invalid Input", "Please enter a valid number for the interval.")

    # --- Toggle Data Logging ---
    def toggle_data_log(self, check_state_value):
        is_checked = (check_state_value == Qt.CheckState.Checked.value)
        if is_checked:
            data_console_handler.setLevel(logging.INFO)
            logger.info("Raw data logging (INFO level) to console enabled.")
        else:
            data_console_handler.setLevel(logging.WARNING) # Set level higher to effectively disable INFO logs
            logger.info("Raw data logging (INFO level) to console disabled.")

    # --- Close Event Handling ---
    def closeEvent(self, event):
        global stop_flag, current_task, loop, asyncio_thread

        logger.info("Close event triggered. Shutting down...")
        stop_flag = True # Signal asyncio loop FIRST

        # --- Cancel asyncio task and wait for thread ---
        task_cancelled = False
        if current_task and not current_task.done():
             if loop and loop.is_running():
                logger.info("Requesting cancellation of active asyncio task...")
                if not current_task.cancelled():
                    future = asyncio.run_coroutine_threadsafe(self.cancel_and_wait_task(current_task), loop)
                    try:
                        future.result(timeout=1.0) # Wait briefly for cancellation to be processed
                        task_cancelled = True
                        logger.info("Asyncio task cancellation initiated.")
                    except asyncio.TimeoutError: logger.warning("Timeout waiting for async task cancellation confirmation.")
                    except Exception as e: logger.error(f"Error during async task cancellation: {e}")
                else: logger.info("Asyncio task was already cancelled.")
             else: logger.warning("Asyncio loop not running, cannot cancel task.")
        else: logger.info("No active asyncio task or task already done.")

        # --- Wait for asyncio thread to finish cleanly ---
        if asyncio_thread and asyncio_thread.is_alive():
            logger.info("Waiting for asyncio thread to finish (max 5s)...")
            asyncio_thread.join(timeout=5.0)
            if asyncio_thread.is_alive(): logger.warning("Asyncio thread did not terminate cleanly within the timeout.")
            else: logger.info("Asyncio thread finished.")
        else: logger.info("Asyncio thread not running or already finished.")

        # --- Now perform GUI/Synchronous cleanup ---
        logger.info("Performing GUI cleanup...")

        # Stop GUI timers
        self.plot_update_timer.stop()
        self.scan_throbber_timer.stop()

        # Remove Log Handler
        if self.log_handler:
            logger.info("Removing GUI log handler...")
            try:
                if self.log_handler in logging.getLogger().handlers: logging.getLogger().removeHandler(self.log_handler)
                self.log_handler.close(); self.log_handler = None
                logger.info("GUI log handler removed and closed.")
            except Exception as e: logging.error(f"Error removing/closing GUI log handler: {e}", exc_info=True)

        # Clear GUI Components (Optional but good practice)
        logger.info("Clearing GUI components before closing window...")
        try:
            self.clear_gui_action(confirm=False) # Use the new clear method
        except Exception as e: logger.error(f"Error clearing GUI during closeEvent: {e}", exc_info=True)

        # Handle final capture state (Likely redundant if clear_gui_action worked)
        if self.is_capturing:
            logger.warning("Capture still marked active during final shutdown phase. Attempting generation.")
            try: self.stop_and_generate_files()
            except Exception as e: logger.error(f"Error generating files on close: {e}", exc_info=True)
            self.is_capturing = False # Ensure reset

        logger.info("Exiting application.")
        event.accept() # Allow the window and application to close

    # Helper coroutine for closeEvent
    async def cancel_and_wait_task(self, task):
        if task and not task.done():
            task.cancel()
            try:
                await task # Wait for the cancellation to be processed
            except asyncio.CancelledError:
                logger.info("Async task successfully cancelled and awaited.")
            except Exception as e:
                logger.error(f"Exception while awaiting cancelled task: {e}", exc_info=True)

# --- END OF MainWindow CLASS DEFINITION ---


# --- Asyncio Thread Setup ---
def run_asyncio_loop():
    global loop
    try: asyncio.run(main_async())
    except Exception as e: logging.critical(f"Fatal exception in asyncio loop: {e}", exc_info=True)
    finally: logging.info("Asyncio thread function finished.")


# --- Main Execution ---
if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        main_window = MainWindow()
        main_window.show()

        logger.info("Starting asyncio thread...")
        asyncio_thread = threading.Thread(target=run_asyncio_loop, daemon=True)
        asyncio_thread.start()

        logger.info("Starting PyQt application event loop...")
        exit_code = app.exec()
        logger.info(f"PyQt application finished with exit code {exit_code}.")

        stop_flag = True # Ensure flag is set even if GUI closed abnormally
        if asyncio_thread.is_alive():
            # Give the thread a chance to finish after loop stop signal
            asyncio_thread.join(timeout=2.0)
            if asyncio_thread.is_alive(): logger.warning("Asyncio thread still alive after final join.")

        sys.exit(exit_code)
    finally:
        try: plt.style.use('default'); logger.debug("Reset matplotlib style.")
        except Exception as e: logger.warning(f"Could not reset matplotlib style: {e}")

# <<< END OF FILE >>>