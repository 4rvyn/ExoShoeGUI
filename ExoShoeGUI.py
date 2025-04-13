# <<< START OF FILE >>>
import asyncio
from bleak import BleakScanner, BleakClient, BleakError
import logging
from typing import Optional, Callable, Dict, List, Tuple, Set
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

# --- Configuration classes (MODIFIED CharacteristicConfig) ---
class CharacteristicConfig:
    def __init__(self, uuid: str, handler: Callable[[bytearray], dict], produces_data_types: List[str]):
        self.uuid = uuid
        self.handler = handler
        # *** ADDED ***: List of data keys this characteristic's handler produces
        self.produces_data_types = produces_data_types

class DeviceConfig:
    def __init__(self, name: str, service_uuid: str, characteristics: list[CharacteristicConfig],
                 find_timeout: float = 10.0, data_timeout: float = 1.0):
        self.name = name
        self.service_uuid = service_uuid
        self.characteristics = characteristics
        self.find_timeout = find_timeout
        self.data_timeout = data_timeout


#####################################################################################################################
# Start of customizable section
#####################################################################################################################
# This section is where you can customize the device configuration, data handling, and plotting.

# 1. Data handlers for different characteristics
# 2. Device configuration (add UUIDs AND `produces_data_types`)
# 3. Plots configuration (tabs, titles, labels, datasets, plot sizes)

# --- Data Handlers (Unchanged) ---
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
# *** MODIFIED: Added produces_data_types to each CharacteristicConfig ***
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
        # Example of a potentially missing UUID for testing:
        # CharacteristicConfig(uuid="19B100FF-E8F2-537E-4F6C-D104768A1214", handler=lambda d: {'dummy': 0.0},
        #                      produces_data_types=['dummy']),
    ],
    find_timeout=5.0,
    data_timeout=1.0
)

# List of available device names for the dropdown
AVAILABLE_DEVICE_NAMES = ["Nano33IoT", "NanoESP32"]

# --- Plot Configuration (Unchanged) ---
# The top-level list now represents tabs. Each dictionary defines a tab.
# Each plot configuration dictionary can optionally include 'plot_height' and 'plot_width'.
plot_groups = [
    {
        'tab_title': 'IMU Basic', # Changed 'title' to 'tab_title' for clarity
        'plots': [
            {   'row': 0,'col': 0,'title': 'Orientation vs Time','xlabel': 'Time [s]','ylabel': 'Degrees',
                'plot_height': 300, 'plot_width': 450, # Configurable plot size
                'datasets': [{'data_type': 'orientation_x', 'label': 'X (Roll)', 'color': 'r'},
                             {'data_type': 'orientation_y', 'label': 'Y (Pitch)', 'color': 'g'},
                             {'data_type': 'orientation_z', 'label': 'Z (Yaw)', 'color': 'b'}]
            },
            {   'row': 0,'col': 1,'title': 'Angular Velocity vs Time','xlabel': 'Time [s]','ylabel': 'Degrees/s',
                'plot_height': 300, 'plot_width': 600, # Configurable plot size
                'datasets': [{'data_type': 'gyro_x', 'label': 'X', 'color': 'r'},
                             {'data_type': 'gyro_y', 'label': 'Y', 'color': 'g'},
                             {'data_type': 'gyro_z', 'label': 'Z', 'color': 'b'}]
            },
        ]
    },
    {
        'tab_title': 'IMU Acceleration',
        'plots': [
            {   'row': 0,'col': 0,'title': 'Linear Acceleration vs Time','xlabel': 'Time [s]','ylabel': 'm/s²',
                # Missing size config: will use default/layout behavior
                'datasets': [{'data_type': 'lin_accel_x', 'label': 'X', 'color': 'r'},
                             {'data_type': 'lin_accel_y', 'label': 'Y', 'color': 'g'},
                             {'data_type': 'lin_accel_z', 'label': 'Z', 'color': 'b'}]
            },
             {  'row': 1,'col': 0,'title': 'Raw Acceleration vs Time','xlabel': 'Time [s]','ylabel': 'm/s²',
                'plot_height': 280, # Only height specified
                'datasets': [{'data_type': 'accel_x', 'label': 'X', 'color': 'r'},
                             {'data_type': 'accel_y', 'label': 'Y', 'color': 'g'},
                             {'data_type': 'accel_z', 'label': 'Z', 'color': 'b'}]
            },
             {  'row': 1,'col': 1,'title': 'Gravity vs Time','xlabel': 'Time [s]','ylabel': 'm/s²',
                'plot_width': 400, # Only width specified
                'datasets': [{'data_type': 'gravity_x', 'label': 'X', 'color': 'r'},
                             {'data_type': 'gravity_y', 'label': 'Y', 'color': 'g'},
                             {'data_type': 'gravity_z', 'label': 'Z', 'color': 'b'}]
            }
        ]
    },
    {
        'tab_title': 'Other Sensors',
        'plots': [
             {  'row': 0,'col': 0,'title': 'Magnetic Field vs Time','xlabel': 'Time [s]','ylabel': 'µT',
                 'plot_height': 350, 'plot_width': 600, # Larger plot
                 'datasets': [{'data_type': 'mag_x', 'label': 'X', 'color': 'r'},
                             {'data_type': 'mag_y', 'label': 'Y', 'color': 'g'},
                             {'data_type': 'mag_z', 'label': 'Z', 'color': 'b'}]
             },
             # Example Plot using the dummy UUID if it were added above
             # {  'row': 1,'col': 0,'title': 'Dummy Data Plot','xlabel': 'Time [s]','ylabel': 'Value',
             #    'datasets': [{'data_type': 'dummy', 'label': 'Dummy', 'color': 'm'}]
             # },
        ]
    }
]
#####################################################################################################################
# End of customizable section
#####################################################################################################################


# --- PyQtGraph Plot Manager (MODIFIED) ---
class PlotManager:
    # Manages plots distributed across multiple tabs
    def __init__(self, tab_widget: QTabWidget, plot_groups_config: List[Dict], device_config_ref: DeviceConfig):
        self.tab_widget = tab_widget # Parent is now the QTabWidget
        self.plot_groups_config = plot_groups_config
        self.device_config_ref = device_config_ref # Reference to build map
        # Store references to plot widgets, plot items, and lines
        self.plot_items: Dict[Tuple[int, int, int], pg.PlotItem] = {} # (tab_idx, row, col) -> PlotItem
        self.lines: Dict[Tuple[int, int, int, str], pg.PlotDataItem] = {} # (tab_idx, row, col, data_type) -> PlotDataItem
        self.plot_configs: Dict[Tuple[int, int, int], Dict] = {} # (tab_idx, row, col) -> plot_config dict
        self.plot_widgets: Dict[Tuple[int, int, int], pg.PlotWidget] = {} # (tab_idx, row, col) -> PlotWidget

        # *** ADDED ***: State for missing UUIDs and their display
        self.missing_uuids: Set[str] = set()
        self.uuid_not_found_texts: Dict[Tuple[int, int, int], pg.TextItem] = {} # (tab_idx, row, col) -> TextItem
        self.data_type_to_uuid_map: Dict[str, str] = {} # data_type -> uuid
        self._build_data_type_map()

        self.create_plots()

    def _build_data_type_map(self):
        """Builds the mapping from data_type keys to their source UUID."""
        self.data_type_to_uuid_map = {}
        for char_config in self.device_config_ref.characteristics:
            for data_type in char_config.produces_data_types:
                if data_type in self.data_type_to_uuid_map:
                    logger.warning(f"Data type '{data_type}' is produced by multiple UUIDs. Mapping to {char_config.uuid}.")
                self.data_type_to_uuid_map[data_type] = char_config.uuid
        # logger.debug(f"Built data_type -> UUID map: {self.data_type_to_uuid_map}")

    def create_plots(self):
        # (Plot creation logic remains largely the same as before)
        for tab_idx, tab_group_config in enumerate(self.plot_groups_config):
            tab_title = tab_group_config.get('tab_title', f'Tab {tab_idx + 1}')
            plots_config = tab_group_config.get('plots', [])

            if not plots_config:
                empty_tab_content = QWidget()
                empty_tab_content.setLayout(QVBoxLayout())
                empty_tab_content.layout().addWidget(QLabel(f"No plots configured for '{tab_title}'"))
                self.tab_widget.addTab(empty_tab_content, tab_title)
                continue

            tab_content_widget = QWidget()
            grid_layout = QGridLayout(tab_content_widget)

            for plot_config in plots_config:
                row = plot_config['row']
                col = plot_config['col']
                key = (tab_idx, row, col)
                self.plot_configs[key] = plot_config

                plot_widget = pg.PlotWidget()
                self.plot_widgets[key] = plot_widget

                plot_height = plot_config.get('plot_height')
                plot_width = plot_config.get('plot_width')
                if plot_height is not None: plot_widget.setFixedHeight(plot_height)
                if plot_width is not None: plot_widget.setFixedWidth(plot_width)

                plot_item = plot_widget.getPlotItem()
                self.plot_items[key] = plot_item

                plot_item.setTitle(plot_config['title'], size='10pt')
                plot_item.setLabel('bottom', plot_config['xlabel'])
                plot_item.setLabel('left', plot_config['ylabel'])
                plot_item.showGrid(x=True, y=True, alpha=0.1)
                plot_item.addLegend(offset=(10, 5))
                plot_item.getViewBox().setDefaultPadding(0.01)
                # *** ADDED ***: Connect view change signal to reposition text
                plot_item.getViewBox().sigRangeChanged.connect(partial(self.position_text_item_for_plot, key))


                for dataset in plot_config['datasets']:
                    data_type = dataset['data_type']
                    pen = pg.mkPen(color=dataset['color'], width=1.5)
                    line = plot_item.plot(pen=pen, name=dataset['label'])
                    self.lines[(tab_idx, row, col, data_type)] = line

                grid_layout.addWidget(plot_widget, row, col)

            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setWidget(tab_content_widget)
            self.tab_widget.addTab(scroll_area, tab_title)

    def update_missing_uuids(self, missing_uuids_set: set):
        """Slot to receive the set of missing UUIDs from the BLE task."""
        logger.info(f"PlotManager received missing UUIDs: {missing_uuids_set if missing_uuids_set else 'None'}")
        self.missing_uuids = missing_uuids_set
        self.refresh_missing_uuid_texts()
        # Trigger a plot update to clear lines if needed
        QTimer.singleShot(0, self._request_gui_update) # Request update in next cycle

    def _request_gui_update(self):
        # Find the main window instance to emit the signal
        # This assumes the PlotManager is owned by MainWindow or can access it
        # A more robust way would be to pass the signal emitter or main window ref
        try:
            # Try to find the MainWindow instance
            mw = next(widget for widget in QApplication.topLevelWidgets() if isinstance(widget, MainWindow))
            mw.request_plot_update_signal.emit()
        except StopIteration:
            logger.error("Could not find MainWindow instance to request plot update.")
        except Exception as e:
             logger.error(f"Error requesting plot update: {e}")


    def refresh_missing_uuid_texts(self):
        """Adds or removes 'UUID not found' messages based on self.missing_uuids."""
        logger.debug("Refreshing 'UUID not found' text items in plots...")
        for key, plot_config in self.plot_configs.items(): # key = (tab_idx, row, col)
            plot_item = self.plot_items.get(key)
            if not plot_item: continue

            required_uuids_for_plot = set()
            for dataset in plot_config.get('datasets', []):
                data_type = dataset['data_type']
                uuid = self.data_type_to_uuid_map.get(data_type)
                if uuid:
                    required_uuids_for_plot.add(uuid)
                else:
                    logger.warning(f"Plot '{plot_config.get('title', key)}' requires data_type '{data_type}' which has no UUID mapping.")

            missing_uuids_for_this_plot = required_uuids_for_plot.intersection(self.missing_uuids)

            if missing_uuids_for_this_plot:
                first_missing_uuid = next(iter(missing_uuids_for_this_plot)) # Get one missing UUID
                text_content = f"UUID:\n{first_missing_uuid}\nnot found!"
                text_item = self.uuid_not_found_texts.get(key)

                if text_item:
                    logger.debug(f"Updating existing text for plot {key}")
                    text_item.setText(text_content)
                else:
                    logger.debug(f"Creating new text for plot {key}")
                    text_item = pg.TextItem(text_content, color=(150, 150, 150), anchor=(0.5, 0.5))
                    text_item.setZValue(100) # Ensure it's on top
                    plot_item.addItem(text_item)
                    self.uuid_not_found_texts[key] = text_item

                # Position the text item (or reposition if it existed)
                self.position_text_item(key, text_item)

            else: # No missing UUIDs for this plot
                if key in self.uuid_not_found_texts:
                    logger.debug(f"Removing text for plot {key}")
                    text_item = self.uuid_not_found_texts[key]
                    plot_item.removeItem(text_item)
                    del self.uuid_not_found_texts[key]

    def position_text_item(self, key, text_item):
        """Positions the text item in the center of the plot's current view."""
        plot_item = self.plot_items.get(key)
        if not plot_item or not text_item: return

        try:
            view_box = plot_item.getViewBox()
            if not view_box.autoRangeEnabled()[1]: # If Y is not auto-ranging, use default range
                 y_range = plot_item.getAxis('left').range
            else:
                 y_range = view_box.viewRange()[1] # Use view range if auto-ranging
            x_range = view_box.viewRange()[0]

            # Handle potential invalid ranges during initialization or reset
            if None in x_range or None in y_range or x_range[1] <= x_range[0] or y_range[1] <= y_range[0]:
                # Fallback to a default position if range is invalid
                center_x = 0.5
                center_y = 0.5
                logger.debug(f"Using fallback position for text in plot {key} due to invalid range ({x_range}, {y_range})")
            else:
                center_x = x_range[0] + (x_range[1] - x_range[0]) / 2
                center_y = y_range[0] + (y_range[1] - y_range[0]) / 2

            text_item.setPos(center_x, center_y)
            # logger.debug(f"Positioned text for {key} at ({center_x:.2f}, {center_y:.2f})")
        except Exception as e:
            logger.warning(f"Could not position text item for plot {key}: {e}")

    def position_text_item_for_plot(self, key):
        """Slot connected to sigRangeChanged to reposition the text item."""
        if key in self.uuid_not_found_texts:
            self.position_text_item(key, self.uuid_not_found_texts[key])


    def update_plots(self, is_flowing: bool, current_relative_time: float):
        if plotting_paused or start_time is None:
            return

        for key, plot_item in self.plot_items.items(): # key = (tab_idx, row, col)
            config = self.plot_configs.get(key)
            if not config: continue

            # --- Determine Time Axis Range ---
            min_time_axis = 0
            max_time_axis = max(current_relative_time, flowing_interval)

            if is_flowing:
                min_time_axis = max(0, current_relative_time - flowing_interval)
                max_time_axis = current_relative_time
                plot_item.setXRange(min_time_axis, max_time_axis, padding=0.02)
            else:
                max_data_time = 0
                for dataset in config['datasets']:
                    data_type = dataset['data_type']
                    # Only consider data types from non-missing UUIDs for axis range
                    uuid = self.data_type_to_uuid_map.get(data_type)
                    if uuid and uuid not in self.missing_uuids and data_type in data_buffers and data_buffers[data_type]:
                        try:
                            max_data_time = max(max_data_time, data_buffers[data_type][-1][0])
                        except IndexError: pass

                max_time_axis = max(max_data_time, flowing_interval)
                plot_item.setXRange(0, max_time_axis, padding=0.02)

            # --- Update Data for Lines in this Plot ---
            data_updated_in_plot = False
            plot_has_missing_uuid_text = (key in self.uuid_not_found_texts)

            for dataset in config['datasets']:
                data_type = dataset['data_type']
                line_key = key + (data_type,)
                line = self.lines.get(line_key)
                uuid = self.data_type_to_uuid_map.get(data_type)

                # Check if UUID is missing
                if line and uuid and uuid in self.missing_uuids:
                    line.setData(x=[], y=[]) # Clear line if UUID is missing
                    # logger.debug(f"Cleared line for {data_type} (UUID {uuid} missing)")
                elif line and data_type in data_buffers:
                    data = data_buffers[data_type]
                    plot_data_tuples = []
                    if is_flowing:
                        start_idx = bisect.bisect_left(data, min_time_axis, key=lambda x: x[0])
                        plot_data_tuples = data[start_idx:]
                    else:
                        plot_data_tuples = data

                    if plot_data_tuples:
                        try:
                            times = np.array([item[0] for item in plot_data_tuples])
                            values = np.array([item[1] for item in plot_data_tuples])
                            line.setData(x=times, y=values)
                            data_updated_in_plot = True
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Could not convert/set data for {data_type}: {e}")
                            line.setData(x=[], y=[])
                    else:
                        line.setData(x=[], y=[])

                elif line:
                    line.setData(x=[], y=[]) # Clear line if data_type not found in buffers

            # --- Y Auto-Ranging ---
            # Only auto-range Y if data was updated AND no "UUID not found" text is shown
            if data_updated_in_plot and not plot_has_missing_uuid_text:
                plot_item.enableAutoRange(axis='y', enable=True)
            elif plot_has_missing_uuid_text:
                 # Disable auto-ranging and set a default range if text is shown
                 plot_item.enableAutoRange(axis='y', enable=False)
                 # Keep existing range or set a default? Let's try setting a simple default.
                 plot_item.setYRange(0, 1, padding=0.1) # Example default range


    def clear_plots(self):
        # Clear data from all lines
        for line in self.lines.values():
            line.setData(x=[], y=[])

        # Clear "UUID not found" text items
        keys_to_remove = list(self.uuid_not_found_texts.keys())
        for key in keys_to_remove:
             plot_item = self.plot_items.get(key)
             text_item = self.uuid_not_found_texts.get(key)
             if plot_item and text_item:
                 try:
                     plot_item.removeItem(text_item)
                 except Exception as e:
                     logger.warning(f"Error removing text item during clear: {e}")
             if key in self.uuid_not_found_texts:
                del self.uuid_not_found_texts[key]
        self.missing_uuids.clear() # Assume clear means we forget missing status too


        # Reset view ranges for all plot items
        for plot_item in self.plot_items.values():
            plot_item.setXRange(0, flowing_interval, padding=0.02)
            plot_item.setYRange(0, 1, padding=0.1) # Reset Y range to default
            plot_item.enableAutoRange(axis='y', enable=True) # Re-enable Y auto-ranging


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
            data_buffers[key] = []
        data_buffers[key].append((relative_time, value))

async def find_device(device_config: DeviceConfig) -> Optional[BleakClient]:
    """Finds the target device using BleakScanner."""
    found_event = asyncio.Event()
    target_device = None
    scan_cancelled = False

    def detection_callback(device, advertisement_data):
        nonlocal target_device, found_event
        if not found_event.is_set():
            target_service_lower = device_config.service_uuid.lower()
            advertised_uuids_lower = [u.lower() for u in advertisement_data.service_uuids]
            device_name = getattr(device, 'name', None)
            if device_name == device_config.name and target_service_lower in advertised_uuids_lower:
                target_device = device
                found_event.set()
                logger.info(f"Match found and event SET for: {device.name} ({device.address})")
            elif device_name and device_config.name and device_name.lower() == device_config.name.lower():
                 logger.debug(f"Found name match '{device_name}' but service UUID mismatch. Adv: {advertised_uuids_lower}, Target: {target_service_lower}")

    scanner = BleakScanner(
        detection_callback=detection_callback,
        service_uuids=[device_config.service_uuid]
    )
    logger.info(f"Starting scanner for {device_config.name} (Service: {device_config.service_uuid})...")
    gui_emitter.emit_scan_throbber("Scanning...")

    try:
        await scanner.start()
        try:
            await asyncio.wait_for(found_event.wait(), timeout=device_config.find_timeout)
            if target_device:
                logger.info(f"Device found event confirmed for {target_device.name}")
            else:
                 logger.warning("Found event was set, but target_device is still None.")
        except asyncio.TimeoutError:
            logger.warning(f"Device '{device_config.name}' not found within {device_config.find_timeout} seconds (timeout).")
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
        # Ensure the 'scanner' variable exists in the local scope before trying to access it
        if 'scanner' in locals() and scanner is not None:
            try:
                # Attempt to stop the scanner regardless of its reported 'is_scanning' state.
                # Bleak's stop() should be idempotent (safe to call multiple times)
                # or handle the already-stopped state gracefully.
                logger.info(f"Attempting to stop scanner {scanner}...")
                await scanner.stop()
                logger.info(f"Scanner stop command issued for {scanner}.")
            except Exception as e:
                # Log the error but don't let it prevent further cleanup if possible.
                # Avoid full traceback here unless debugging, as errors might be expected if already stopped.
                logger.warning(f"Error encountered while stopping scanner: {e}", exc_info=False)
        else:
            # This case can happen if scanner creation failed very early.
            logger.debug("Scanner object not found or is None in finally block, skipping stop.")
            
    if scan_cancelled: raise asyncio.CancelledError
    return target_device


async def connection_task():
    global client, last_received_time, state
    # Keep track of characteristics we successfully subscribe to
    found_char_configs: List[CharacteristicConfig] = []

    while state == "scanning":
        target_device = None
        found_char_configs = [] # Reset for each connection attempt cycle
        try:
            target_device = await find_device(device_config)
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
                 logger.info(f"Device '{device_config.name}' not found, retrying scan in 3 seconds...")
                 gui_emitter.emit_scan_throbber(f"Device '{device_config.name}' not found. Retrying...")
                 await asyncio.sleep(3)
                 continue
            else:
                 logger.info("Scan stopped while waiting for device.")
                 break

        gui_emitter.emit_connection_status(f"Found {device_config.name}. Connecting...")
        client = None
        connection_successful = False
        for attempt in range(3):
             if state != "scanning": logger.info("Connection attempt aborted, state changed."); break
             try:
                  logger.info(f"Connecting (attempt {attempt + 1})...")
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
            logger.info(f"Checking characteristics for service {device_config.service_uuid}...")
            service = client.services.get_service(device_config.service_uuid)
            if not service:
                 logger.error(f"Service {device_config.service_uuid} not found on connected device.")
                 gui_emitter.emit_show_error("Connection Error", f"Service UUID\n{device_config.service_uuid}\nnot found on device.")
                 gui_emitter.emit_state_change("disconnecting") # Treat as fatal error for this connection
                 notification_errors = True # Skip notification attempts
            else:
                 logger.info("Service found. Checking configured characteristics...")
                 found_char_configs = [] # Reset list for this successful connection
                 for char_config in device_config.characteristics:
                     bleak_char = service.get_characteristic(char_config.uuid)
                     if bleak_char:
                         logger.info(f"Characteristic found: {char_config.uuid}")
                         # Check if characteristic supports notify or indicate
                         if "notify" in bleak_char.properties or "indicate" in bleak_char.properties:
                             found_char_configs.append(char_config)
                         else:
                             logger.warning(f"Characteristic {char_config.uuid} found but does not support notify/indicate.")
                             # log and add to missing.
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
                            # If start_notify fails, maybe add to missing_uuids and re-emit?
                            missing_uuids.add(char_uuid)

                    if not all_notifications_started:
                        logger.error("Could not start all required notifications. Disconnecting.")
                        gui_emitter.emit_missing_uuids(missing_uuids) # Update GUI with newly failed ones
                        gui_emitter.emit_state_change("disconnecting")
                    else:
                        logger.info("Notifications started successfully. Listening...")
                        # (Rest of the listening loop remains the same)
                        last_received_time = time.time()
                        disconnected_event.clear()
                        while state == "connected":
                             try:
                                 await asyncio.wait_for(disconnected_event.wait(), timeout=device_config.data_timeout + 1.0)
                                 logger.info("Disconnected event received while listening.")
                                 gui_emitter.emit_state_change("disconnecting")
                                 break
                             except asyncio.TimeoutError:
                                 current_time = time.time()
                                 if current_time - last_received_time > device_config.data_timeout:
                                     logger.warning(f"No data received for {current_time - last_received_time:.1f}s (timeout: {device_config.data_timeout}s). Assuming disconnect.")
                                     gui_emitter.emit_state_change("disconnecting")
                                     break
                                 else: continue
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
                          # Iterate through FOUND characteristics only
                          for char_config in found_char_configs:
                              try:
                                  # Basic check if characteristic exists before stop_notify
                                  service = local_client.services.get_service(device_config.service_uuid)
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
            found_char_configs = [] # Clear the list

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
            await current_task
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
    request_plot_update_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live BLE Data Plotter (UUID Discovery)") # Updated title
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

        # --- Top Button Bar ---
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
        self.clear_button = QPushButton("Clear Plots"); self.clear_button.clicked.connect(self.clear_plots_action); self.button_layout.addWidget(self.clear_button)
        self.status_label = QLabel("On Standby"); self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter); self.status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred); self.button_layout.addWidget(self.status_label)
        self.main_layout.addWidget(self.button_bar)

        # --- Plot Area (TabWidget with PlotManager) ---
        self.tab_widget = QTabWidget()
        self.tab_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Pass device_config reference to PlotManager
        self.plot_manager = PlotManager(self.tab_widget, plot_groups, device_config)
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
        self.plot_update_timer.timeout.connect(self.trigger_plot_update)
        self.plot_update_timer.start()

        self.scan_throbber_timer = QTimer(self)
        self.scan_throbber_timer.setInterval(150)
        self.scan_throbber_timer.timeout.connect(self.animate_scan_throbber)
        self.throbber_chars = ["|", "/", "-", "\\"]
        self.throbber_index = 0

        gui_emitter.state_change_signal.connect(self.handle_state_change)
        gui_emitter.scan_throbber_signal.connect(self.update_scan_status)
        gui_emitter.connection_status_signal.connect(self.update_connection_status)
        gui_emitter.show_error_signal.connect(self.show_message_box)
        # Connect the missing UUIDs signal to the PlotManager's slot
        gui_emitter.missing_uuids_signal.connect(self.plot_manager.update_missing_uuids)
        self.request_plot_update_signal.connect(self._update_plots_now)

        self.handle_state_change("idle") # Initialize state

    # --- Slot to Append Log Messages ---
    def append_log_message(self, message):
        self.log_text_box.append(message)

    # --- Plot Update Triggering ---
    def trigger_plot_update(self):
        self.request_plot_update_signal.emit()

    def _update_plots_now(self):
         if start_time is not None: # Check start_time, plotting_paused is handled internally by PlotManager now indirectly
             current_relative = (datetime.datetime.now() - start_time).total_seconds()
             is_flowing = self.flowing_mode_check.isChecked()
             self.plot_manager.update_plots(is_flowing, current_relative)
         elif not start_time and state == "connected":
             # If connected but no data received yet (start_time is None),
             # ensure plots showing "UUID not found" are displayed correctly.
             self.plot_manager.refresh_missing_uuid_texts() # Re-check positioning

    # --- Scan Animation ---
    def animate_scan_throbber(self):
        if state == "scanning":
            text = "Scanning... " + self.throbber_chars[self.throbber_index]
            self.status_label.setText(text)
            self.throbber_index = (self.throbber_index + 1) % len(self.throbber_chars)
        else:
            self.scan_throbber_timer.stop()

    # --- GUI Action Slots ---

    # --- Handler for Device Dropdown Change ---
    def update_target_device(self, selected_name: str):
        global device_config
        if device_config.name != selected_name:
            logger.info(f"Target device changed via GUI: {selected_name}")
            device_config.name = selected_name
            # Also update the reference in plot manager if needed for future use?
            self.plot_manager.device_config_ref = device_config
            self.plot_manager._build_data_type_map() # Rebuild map if device changes
            logger.info("Device config and plot manager map updated.")

    # State Change Handling
    def handle_state_change(self, new_state: str):
        global state, plotting_paused, start_time # Added start_time reset
        logger.info(f"GUI received state change: {new_state}")
        previous_state = state
        state = new_state # Update global state

        if new_state != "scanning" and self.scan_throbber_timer.isActive():
            self.scan_throbber_timer.stop()

        is_idle = (new_state == "idle")
        self.device_combo.setEnabled(is_idle)
        self.scan_button.setEnabled(True) # Scan/Disconnect always enabled except during disconnect

        if new_state == "idle":
            self.scan_button.setText("Start Scan")
            self.led_indicator.set_color("red"); self.status_label.setText("On Standby")
            self.pause_resume_button.setEnabled(False); self.pause_resume_button.setText("Pause Plotting") # Keep text as Pause Plotting
            self.capture_button.setEnabled(False); self.capture_button.setText("Start Capture")
            plotting_paused = True # Set plots to paused state logically

            # *** Automatically clear plots/state when becoming idle ***
            logger.info("State changed to idle. Automatically clearing plots and state.")
            self.clear_plots_action(confirm=False) # This handles plots, buffers, start_time, missing_uuids

            # Handle capture stopped due to disconnect (clear_plots_action warns but doesn't generate)
            # If you WANT files generated automatically on disconnect, move this block *before* clear_plots_action
            if self.is_capturing: # This flag should be false now if clear_plots_action ran correctly
                 logger.warning("Capture was active when state became idle (likely disconnect). Files were NOT generated automatically by clear.")
                 # If generation on disconnect is desired, uncomment the next line AND move this block before clear_plots_action
                 # QTimer.singleShot(0, self.stop_and_generate_files)

        elif new_state == "scanning":
            self.scan_button.setText("Stop Scan")
            self.led_indicator.set_color("orange"); self.throbber_index = 0
            if not self.scan_throbber_timer.isActive(): self.scan_throbber_timer.start()
            self.pause_resume_button.setEnabled(False)
            self.capture_button.setEnabled(False)
            # Clearing state now happens in toggle_scan before starting scan

        elif new_state == "connected":
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

    # Scan/Connection Status Updates (Unchanged)
    def update_scan_status(self, text: str):
         if state == "scanning": self.status_label.setText(text)
    def update_connection_status(self, text: str):
         if state != "connected" and state != "idle": self.status_label.setText(text)
    def show_message_box(self, title: str, message: str):
        QMessageBox.warning(self, title, message)

    # Scan/Connect/Disconnect Logic (MODIFIED to clear plots/state on start)
    def toggle_scan(self):
        global current_task, loop, state, data_buffers, start_time
        if state == "idle":
            if loop and loop.is_running():
                # Clear plots, buffers, state BEFORE starting scan
                logger.info("Clearing state before starting scan...")
                self.clear_plots_action(confirm=False) # Ensures everything is reset

                self.handle_state_change("scanning")
                current_task = loop.create_task(connection_task())
            else: logger.error("Asyncio loop not running!"); self.show_message_box("Error", "Asyncio loop is not running.")
        elif state == "scanning":
            # Request cancellation, state change handled via callback/finally block
            if current_task and not current_task.done():
                logger.info("Requesting scan cancellation...")
                loop.call_soon_threadsafe(current_task.cancel)
                # GUI state will change to idle via the task's cleanup calling emit_state_change("idle")
            else:
                logger.warning("Stop scan requested, but no task was running/done.")
                self.handle_state_change("idle") # Force idle if no task
            current_task = None
        elif state == "connected":
            # Request disconnect, state change handled via callback/finally block
            if loop and client and client.is_connected:
                logger.info("Requesting disconnection via disconnected_event...")
                loop.call_soon_threadsafe(disconnected_event.set)
                # GUI state change will happen via connection_task cleanup
            elif loop and current_task and not current_task.done():
                logger.info("Requesting disconnect via task cancellation...")
                loop.call_soon_threadsafe(current_task.cancel)
                 # GUI state change will happen via connection_task cleanup
            else:
                logger.warning("Disconnect requested but no active connection/task found.")
                self.handle_state_change("idle") # Force idle

    # Pause/Resume Plotting (Unchanged)
    def toggle_pause_resume(self):
        global plotting_paused
        # Should only be possible if button is enabled (i.e., connected and not capturing)
        if not self.pause_resume_button.isEnabled():
            logger.warning("Pause/Resume toggled while button disabled. Ignoring.")
            return

        plotting_paused = not plotting_paused
        self.pause_resume_button.setText("Resume Plotting" if plotting_paused else "Pause Plotting")
        logger.info(f"Plotting {'paused' if plotting_paused else 'resumed'}")
        # If resuming, trigger an immediate plot update
        if not plotting_paused:
            self.trigger_plot_update()

    # --- Capture Start/Stop Logic (MODIFIED to disable pause/resume) ---
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
                # Reset potential partial state
                self.capture_output_base_dir = None; self.capture_timestamp = None
                return # Don't proceed to set capture state

            # --- Set capture state variables ---
            self.is_capturing = True
            self.capture_button.setText("Stop Capture && Export")
            self.capture_t0_absolute = datetime.datetime.now()
            self.capture_start_relative_time = (self.capture_t0_absolute - start_time).total_seconds()

            # --- Disable Pause/Resume Button ---
            self.pause_resume_button.setEnabled(False)
            logger.info("Pause/Resume plotting disabled during capture.")

            logger.info(f"Capture started. Rel t0: {self.capture_start_relative_time:.3f}s.")

        else:
            # Stop Capture and Generate
            # Call the helper function which will also handle re-enabling the button
            self.stop_and_generate_files()

    # --- Stop Capture & File Generation Helper (MODIFIED to re-enable pause/resume) ---
    def stop_and_generate_files(self):
        if not self.is_capturing:
            logger.warning("stop_and_generate called but capture inactive.")
            # Ensure pause button state is correct even if called spuriously
            if state == "connected":
                 self.pause_resume_button.setEnabled(True)
            else:
                 self.pause_resume_button.setEnabled(False)
            return

        logger.info("Stopping capture, generating PGF & CSV.")
        output_dir = self.capture_output_base_dir
        start_rel_time = self.capture_start_relative_time
        capture_end_relative_time = (datetime.datetime.now() - start_time).total_seconds() if start_time else None

        # --- Reset capture state FIRST ---
        self.is_capturing = False
        self.capture_button.setText("Start Capture")
        # Ensure capture button is enabled only if connected (check current state)
        self.capture_button.setEnabled(state == "connected")

        # --- Re-enable Pause/Resume Button ONLY if connected ---
        if state == "connected":
            self.pause_resume_button.setEnabled(True)
            logger.info("Pause/Resume plotting re-enabled after capture (still connected).")
        else:
            # If not connected, ensure pause/resume remains disabled
            self.pause_resume_button.setEnabled(False)
            logger.info("Pause/Resume plotting remains disabled after capture (not connected).")


        # --- File Generation Logic ---
        if output_dir and start_rel_time is not None and capture_end_relative_time is not None:
            if not data_buffers:
                 logger.warning("No data captured during the active period. Skipping PGF/CSV generation.")
                 # Optionally show message box here too
                 # self.show_message_box("Generation Skipped", "No captured data found in the active period.")
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
            try: self.generate_pgf_plots_from_buffer(pgf_subdir, start_rel_time)
            except Exception as e: logger.error(f"PGF generation failed: {e}", exc_info=True); gen_errors.append(f"PGF: {e}")
            try: self.generate_csv_files_from_buffer(csv_subdir, start_rel_time, capture_end_relative_time, start_rel_time)
            except Exception as e: logger.error(f"CSV generation failed: {e}", exc_info=True); gen_errors.append(f"CSV: {e}")

            if not gen_errors: self.show_message_box("Generation Complete", f"Files generated in:\n{output_dir}")
            else: self.show_message_box("Generation Errors", f"Completed with errors in:\n{output_dir}\n\n" + "\n".join(gen_errors))

        else:
             reason = ""
             if not output_dir: reason += " Output dir missing."
             if start_rel_time is None: reason += " Start time missing."
             if capture_end_relative_time is None: reason += " End time missing (start_time lost?)."
             logger.error(f"Cannot generate files:{reason}")
             self.show_message_box("File Gen Error", f"Internal error:{reason}")

        # --- Final reset of capture vars ---
        self.capture_output_base_dir = None
        self.capture_start_relative_time = None
        self.capture_t0_absolute = None
        self.capture_timestamp = None

    # --- PGF Generation (Unchanged logic, uses plot_groups config) ---
    def generate_pgf_plots_from_buffer(self, pgf_dir: str, capture_start_relative_time: float):
        global data_buffers, plot_groups

        logger.info(f"Generating PGF plots (t=0 at capture start, t_offset={capture_start_relative_time:.3f}s). Dir: {pgf_dir}")
        # Check buffer again, although checked in caller
        if not data_buffers: logger.warning("Data buffer empty, skipping PGF generation."); return

        try:
            plt.style.use('science')
            plt.rcParams.update({'text.usetex': False, 'figure.figsize': [5.5, 3.5], 'legend.fontsize': 9,
                                 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.titlesize': 11})
        except Exception as style_err:
            logger.warning(f"Could not apply 'science' style: {style_err}. Using default.")
            plt.rcParams.update({'figure.figsize': [6.0, 4.0]})

        gen_success = False
        for group_config in plot_groups:
            group_title = group_config.get('tab_title', 'UnknownGroup')
            for config in group_config.get('plots', []):
                required_uuids_for_plot = set(self.plot_manager.data_type_to_uuid_map.get(ds['data_type'])
                                              for ds in config.get('datasets', [])
                                              if self.plot_manager.data_type_to_uuid_map.get(ds['data_type']))
                missing_uuids_for_this_plot = required_uuids_for_plot.intersection(self.plot_manager.missing_uuids)

                if missing_uuids_for_this_plot:
                    logger.warning(f"Skipping PGF for '{config['title']}' as it depends on missing UUID(s): {missing_uuids_for_this_plot}")
                    continue # Skip this plot

                fig, ax = plt.subplots()
                ax.set_title(config['title']); ax.set_xlabel(config['xlabel']); ax.set_ylabel(config['ylabel'])
                plot_created = False
                for dataset in config['datasets']:
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
                                ax.plot(times_rel_capture, values, label=dataset['label'], color=dataset['color'], linewidth=1.2)
                                plot_created = True
                            except Exception as plot_err: logger.error(f"Error plotting {data_type}: {plot_err}")

                if plot_created:
                    ax.legend(); ax.grid(True, linestyle='--', alpha=0.6); fig.tight_layout(pad=0.5)
                    safe_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in config['title']).rstrip().replace(' ', '_')
                    prefix = f"{self.capture_timestamp}_" if self.capture_timestamp else ""
                    pgf_filename = f"{prefix}{safe_title}.pgf"
                    pgf_filepath = os.path.join(pgf_dir, pgf_filename)
                    try: fig.savefig(pgf_filepath, bbox_inches='tight'); logger.info(f"Saved PGF: {pgf_filename}"); gen_success = True
                    except Exception as save_err: logger.error(f"Error saving PGF {pgf_filename}: {save_err}"); raise RuntimeError(f"Save PGF failed: {save_err}") from save_err
                else: logger.info(f"Skipping PGF '{config['title']}' (no data in capture window).")
                plt.close(fig)

        if gen_success: logger.info(f"PGF generation finished. Dir: {pgf_dir}")
        else: logger.warning("PGF done, but no plots saved (no data / missing UUIDs?).")

    # --- CSV Generation (MODIFIED get_series to exclude missing UUID data) ---
    def generate_csv_files_from_buffer(self, csv_dir: str, filter_start_rel_time: float, filter_end_rel_time: float, time_offset: float):
        global data_buffers, plot_groups

        logger.info(f"Generating CSVs (data {filter_start_rel_time:.3f}s-{filter_end_rel_time:.3f}s rel session, t=0 at capture start offset={time_offset:.3f}s). Dir: {csv_dir}")
        if not data_buffers: logger.warning("Data buffer empty, skipping CSV generation."); return

        def get_series(dt, start, end):
             # Exclude series if its source UUID was missing during the connection
             uuid = self.plot_manager.data_type_to_uuid_map.get(dt)
             if uuid and uuid in self.plot_manager.missing_uuids:
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
        for group_config in plot_groups:
            group_title = group_config.get('tab_title', f"Group_{id(group_config)}")
            logger.info(f"Processing Master CSV for group: '{group_title}'")
            group_types = set(ds['data_type'] for plot_cfg in group_config.get('plots', []) for ds in plot_cfg.get('datasets', []))
            if not group_types: logger.warning(f"Skipping Master CSV '{group_title}': No data types defined."); continue

            series_list = [s for dt in sorted(list(group_types)) if (s := get_series(dt, filter_start_rel_time, filter_end_rel_time)) is not None]
            if not series_list: logger.warning(f"Skipping Master CSV '{group_title}': No valid series data found in window."); continue

            try:
                # Concatenate using the TimeRelSession index for alignment
                master_df = pd.concat(series_list, axis=1, join='outer').sort_index()
                # Insert the adjusted time column (relative to capture start)
                master_df.insert(0, 'Master Time (s)', master_df.index - time_offset)
                safe_g_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in group_title).rstrip().replace(' ', '_')
                prefix = f"{self.capture_timestamp}_" if self.capture_timestamp else ""
                csv_fname = f"{prefix}master_{safe_g_title}.csv"
                csv_fpath = os.path.join(csv_dir, csv_fname)
                master_df.to_csv(csv_fpath, index=False, float_format='%.6f') # Don't write the TimeRelSession index
                logger.info(f"Saved Master CSV: {csv_fname}"); master_gen = True
            except Exception as e: logger.error(f"Error generating Master CSV '{group_title}': {e}", exc_info=True); raise RuntimeError(f"Master CSV generation failed: {e}") from e

        indiv_gen = False
        for group_config in plot_groups:
            for plot_config in group_config.get('plots', []):
                plot_title = plot_config.get('title', f"Plot_{id(plot_config)}")
                logger.info(f"Processing Individual CSV for plot: '{plot_title}'")
                datasets = plot_config.get('datasets', [])
                if not datasets: logger.warning(f"Skipping Individual CSV '{plot_title}': No datasets defined."); continue

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


    # --- Clear Plots Action (MODIFIED to update PlotManager state) ---
    def clear_plots_action(self, confirm=True):
        global data_buffers, start_time
        logger.info("Attempting to clear plot data.")
        do_clear = False
        if confirm:
            # Ask slightly different question if capture is active
            question = "Clear all plot data and reset UUID status?"
            if self.is_capturing:
                question = "Capture is active. Clear all plot data (stopping capture WITHOUT exporting)?\nAlso resets UUID status."

            reply = QMessageBox.question(self, 'Clear Plots', question,
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes: do_clear = True
        else: do_clear = True # Clear without confirm (used internally)

        if do_clear:
            logger.info("Confirmed clearing data buffers, resetting start time, clearing plots and UUID status.")

            # --- Stop capture if active ---
            if self.is_capturing:
                 logger.warning("Capture active during clear plots. Stopping capture WITHOUT generating files.")
                 self.is_capturing = False
                 self.capture_button.setText("Start Capture")
                 # Ensure capture button state reflects connection status after clearing
                 self.capture_button.setEnabled(state == "connected")
                 # Also reset pause/resume button state appropriately
                 if state == "connected":
                    self.pause_resume_button.setEnabled(True)
                 else:
                    self.pause_resume_button.setEnabled(False)
                 # Clear capture temp vars
                 self.capture_output_base_dir = None; self.capture_start_relative_time = None
                 self.capture_t0_absolute = None; self.capture_timestamp = None

            # --- Clear data and state ---
            data_buffers.clear()
            start_time = None
            # Reset the missing UUID state in PlotManager
            self.plot_manager.update_missing_uuids(set())
            # Clear visuals and internal text items in PlotManager
            self.plot_manager.clear_plots()


    # --- Apply Interval (Unchanged) ---
    def apply_interval(self):
        global flowing_interval
        try:
            new_interval = float(self.interval_entry.text())
            if new_interval > 0:
                flowing_interval = new_interval
                logger.info(f"Flowing interval updated to {new_interval}s")
                # Trigger plot update immediately if flowing mode is active
                if self.flowing_mode_check.isChecked():
                    self._update_plots_now()
            else: self.show_message_box("Invalid Input", "Interval must be positive.")
        except ValueError: self.show_message_box("Invalid Input", "Please enter a valid number for the interval.")

    # --- Toggle Data Logging (Unchanged) ---
    def toggle_data_log(self, check_state_value):
        # The check_state_value argument passed by Qt is the raw enum value (e.g., 2 for Checked, 0 for Unchecked)
        # We compare against the enum value directly.
        is_checked = (check_state_value == Qt.CheckState.Checked.value)
        if is_checked:
            data_console_handler.setLevel(logging.INFO)
            logger.info("Raw data logging (INFO level) to console enabled.")
        else:
            data_console_handler.setLevel(logging.WARNING) # Set level higher to effectively disable INFO logs
            logger.info("Raw data logging (INFO level) to console disabled.")

    # --- Close Event Handling (MODIFIED to call clear_plots_action) ---
    # In class MainWindow:
    # In class MainWindow:
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
                    # Schedule cancellation within the loop's thread
                    future = asyncio.run_coroutine_threadsafe(self.cancel_and_wait_task(current_task), loop)
                    try:
                        # Wait briefly for cancellation to be processed by the loop
                        future.result(timeout=1.0)
                        task_cancelled = True
                        logger.info("Asyncio task cancellation initiated.")
                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for async task cancellation confirmation.")
                    except Exception as e:
                        logger.error(f"Error during async task cancellation: {e}")

                else:
                    logger.info("Asyncio task was already cancelled.")
             else:
                 logger.warning("Asyncio loop not running, cannot cancel task.")
        else:
             logger.info("No active asyncio task or task already done.")

        # --- Wait for asyncio thread to finish cleanly ---
        # This join waits for main_async to finish, which in turn awaits the cancelled task's cleanup
        if asyncio_thread and asyncio_thread.is_alive():
            logger.info("Waiting for asyncio thread to finish (max 5s)...")
            asyncio_thread.join(timeout=5.0) # Adjust timeout if needed
            if asyncio_thread.is_alive():
                 logger.warning("Asyncio thread did not terminate cleanly within the timeout.")
                 # Potentially force loop stop if stuck? Be careful here.
                 # if loop and loop.is_running():
                 #     logger.warning("Forcing loop stop...")
                 #     loop.call_soon_threadsafe(loop.stop)
                 #     # Give it another short moment
                 #     asyncio_thread.join(timeout=1.0)
            else:
                 logger.info("Asyncio thread finished.")
        else:
             logger.info("Asyncio thread not running or already finished.")


        # --- Now perform GUI/Synchronous cleanup ---
        logger.info("Performing GUI cleanup...")

        # Stop GUI timers
        self.plot_update_timer.stop()
        self.scan_throbber_timer.stop()

        # Remove Log Handler
        if self.log_handler:
            logger.info("Removing GUI log handler...")
            try:
                # Check if handler is still present before removing
                if self.log_handler in logging.getLogger().handlers:
                    logging.getLogger().removeHandler(self.log_handler)
                self.log_handler.close()
                self.log_handler = None
                logger.info("GUI log handler removed and closed.")
            except Exception as e:
                logging.error(f"Error removing/closing GUI log handler: {e}", exc_info=True)

        # Clear Plots (Optional but good practice)
        logger.info("Clearing plots before closing window...")
        try:
            self.clear_plots_action(confirm=False)
        except Exception as e:
            logger.error(f"Error clearing plots during closeEvent: {e}", exc_info=True)

        # Handle final capture state (Likely redundant if clear_plots worked)
        if self.is_capturing:
            logger.warning("Capture still marked active during final shutdown phase. Attempting generation.")
            try:
                self.stop_and_generate_files()
            except Exception as e:
                logger.error(f"Error generating files on close: {e}", exc_info=True)
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

        stop_flag = True
        if asyncio_thread.is_alive():
            asyncio_thread.join(timeout=2.0)
            if asyncio_thread.is_alive(): logger.warning("Asyncio thread still alive after final join.")

        sys.exit(exit_code)
    finally:
        try: plt.style.use('default'); logger.debug("Reset matplotlib style.")
        except Exception as e: logger.warning(f"Could not reset matplotlib style: {e}")

# <<< END OF FILE >>>