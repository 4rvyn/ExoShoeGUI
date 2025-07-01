# <<< START OF FILE >>>
import asyncio
import qasync
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
import pandas as pd # Easier CSV data handling (merging/resampling)
import struct
import bisect
import numpy as np # needed for pyqtgraph AND heatmap
import re # For cleaning filenames
import math # Needed for heatmap CoP

# --- Matplotlib Imports (used ONLY for PDF export AND heatmap colormaps) ---
import matplotlib # Use Agg backend to avoid GUI conflicts if possible
matplotlib.use('Agg') # Set backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import scienceplots # Make sure it's installed: pip install scienceplots for formatting
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QGridLayout, QCheckBox, QLineEdit,
    QScrollArea, QMessageBox, QSizePolicy, QTextEdit, QTabWidget,
    QComboBox, QSlider # Added QComboBox, QSlider
)
from PyQt6.QtGui import (QColor, QPainter, QBrush, QPen, QPixmap, QImage, QPolygonF, QIntValidator, QDoubleValidator) # Added heatmap graphics items
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QMetaObject, QThread, QPointF # Added QPointF for heatmap


# --- PyQtGraph Import ---
import pyqtgraph as pg

# --- SuperQT Import for RangeSlider ---
from superqt.sliders import QRangeSlider # Needed for heatmap pressure range

# Apply PyQtGraph global options for background/foreground
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOptions(antialias=True) # Ensure anti-aliasing is enabled

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Create a separate logger for data logs
data_logger = logging.getLogger("data_logger")
data_logger.propagate = False # Don't send data logs to root logger's handlers by default
data_console_handler = logging.StreamHandler() # Specific handler for console data logs
data_console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
data_console_handler.setLevel(logging.WARNING) # Set level higher to disable INFO logs by default
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
# data_buffers hold ALL received data since connection/clear
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
        # Builds the mapping from data_type keys to their source UUID.
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
        # update other device config-specific settings here if needed

    def get_uuid_for_data_type(self, data_type: str) -> Optional[str]:
        return self.data_type_to_uuid_map.get(data_type)


#####################################################################################################################
# Start of customizable section
#####################################################################################################################
# Section for customizing device configuration, data handling, and plotting.

# 1. Data handlers for different characteristics
# 2. Device configuration (add UUIDs AND `produces_data_types`)
# 3. Define GUI Component Classes (e.g., plots, indicators)
# 4. Define Tab Layout Configuration using the components

# --- Constants needed for Insole Data Handling ---
ADC_MAX_VOLTAGE = 3.3           # Maximal voltage expected from ADC
# --- Define the keys specifically for the heatmap sensors ---
HEATMAP_KEYS = ["A0C0", "A1C0", "A2C0", "A0C1", "A1C1", "A2C1", "A0C2", "A1C2", "A2C2", "A1C3", "A2C3"]
NUM_HEATMAP_SENSORS = len(HEATMAP_KEYS)
# --- Define the key for the flex sensor ---
FLEX_SENSOR_KEY = "A0C3"

# --- This gain list remains ONLY for the FSR sensors used in the heatmap ---
DEFAULT_SENSOR_GAINS = np.array([ # Order corresponds to HEATMAP_KEYS.
    2.0, # Gain for Sensor 0 (A0C0)
    3.0, # Gain for Sensor 1 (A1C0)
    1.0, # Gain for Sensor 2 (A2C0)
    2.0, # Gain for Sensor 3 (A0C1)
    2.0, # Gain for Sensor 4 (A1C1)
    1.0, # Gain for Sensor 5 (A2C1)
    2.0, # Gain for Sensor 6 (A0C2)
    2.0, # Gain for Sensor 7 (A1C2)
    1.0, # Gain for Sensor 8 (A2C2)
    2.0, # Gain for Sensor 9 (A1C3)
    1.0  # Gain for Sensor 10 (A2C3)
], dtype=np.float32)

# --- START: Flex Sensor Angle Conversion Parameters ---
FLEX_REF_VOLTAGE = 2.415  # Voltage corresponding to 0 degrees
FLEX_REF_ANGLE = 0.0       # Angle at the reference voltage
FLEX_VOLTS_PER_90_DEG = -0.1 # Voltage change for a 90 degree increase
# Calculate slope and intercept for y = mx + c (Angle = slope * Voltage + intercept)
if abs(FLEX_VOLTS_PER_90_DEG) < 1e-9:
    logger.warning("FLEX_VOLTS_PER_90_DEG is near zero. Angle conversion disabled (slope set to 0).")
    FLEX_SLOPE_DEG_PER_VOLT = 0.0
else:
    FLEX_SLOPE_DEG_PER_VOLT = 90.0 / FLEX_VOLTS_PER_90_DEG # Degrees per Volt
FLEX_INTERCEPT_DEG = FLEX_REF_ANGLE - (FLEX_SLOPE_DEG_PER_VOLT * FLEX_REF_VOLTAGE)
logger.info(f"Flex sensor angle conversion: Slope={FLEX_SLOPE_DEG_PER_VOLT:.2f} deg/V, Intercept={FLEX_INTERCEPT_DEG:.2f} deg")
# --- END: Flex Sensor Angle Conversion Parameters ---

# --- Weight Estimation Factor ---
VOLTAGE_TO_WEIGHT_FACTOR = 25.0  # Convert total summed voltage to weight units

INITIAL_PRESSURE_SENSITIVITY = 300.0 # Initial Sensitivity range (GLOBAL, HANDLED BY HEATMAP COMPONENT)

# --- Data Handlers ---
def handle_orientation_data(data: bytearray) -> dict:
    try:
        text = data.decode("utf-8").strip()
        parts = text.split(",")
        if len(parts) != 3:
            data_logger.error("Invalid orientation payload")
            return {}
        x, y, z = map(float, parts)
        data_logger.info(f"Orientation Data: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        return {"orientation_x": x, "orientation_y": y, "orientation_z": z}
    except Exception as e:
        data_logger.error(f"Error parsing orientation data: {e}")
        return {}


def handle_gyro_data(data: bytearray) -> dict:
    try:
        text = data.decode("utf-8").strip()
        parts = text.split(",")
        if len(parts) != 3:
            data_logger.error("Invalid gyro payload")
            return {}
        x, y, z = map(float, parts)
        data_logger.info(f"Gyro Data: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        return {"gyro_x": x, "gyro_y": y, "gyro_z": z}
    except Exception as e:
        data_logger.error(f"Error parsing gyro data: {e}")
        return {}


def handle_lin_accel_data(data: bytearray) -> dict:
    try:
        text = data.decode("utf-8").strip()
        parts = text.split(",")
        if len(parts) != 3:
            data_logger.error("Invalid linear acceleration payload")
            return {}
        x, y, z = map(float, parts)
        data_logger.info(f"Linear Acceleration Data: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        return {"lin_accel_x": x, "lin_accel_y": y, "lin_accel_z": z}
    except Exception as e:
        data_logger.error(f"Error parsing linear acceleration data: {e}")
        return {}


def handle_mag_data(data: bytearray) -> dict:
    try:
        text = data.decode("utf-8").strip()
        parts = text.split(",")
        if len(parts) != 3:
            data_logger.error("Invalid magnetometer payload")
            return {}
        x, y, z = map(float, parts)
        data_logger.info(f"Magnetometer Data: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        return {"mag_x": x, "mag_y": y, "mag_z": z}
    except Exception as e:
        data_logger.error(f"Error parsing magnetometer data: {e}")
        return {}


def handle_accel_data(data: bytearray) -> dict:
    try:
        text = data.decode("utf-8").strip()
        parts = text.split(",")
        if len(parts) != 3:
            data_logger.error("Invalid accelerometer payload")
            return {}
        x, y, z = map(float, parts)
        data_logger.info(f"Accelerometer Data: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        return {"accel_x": x, "accel_y": y, "accel_z": z}
    except Exception as e:
        data_logger.error(f"Error parsing accelerometer data: {e}")
        return {}


def handle_gravity_data(data: bytearray) -> dict:
    try:
        text = data.decode("utf-8").strip()
        parts = text.split(",")
        if len(parts) != 3:
            data_logger.error("Invalid gravity payload")
            return {}
        x, y, z = map(float, parts)
        data_logger.info(f"Gravity Data: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        return {"gravity_x": x, "gravity_y": y, "gravity_z": z}
    except Exception as e:
        data_logger.error(f"Error parsing gravity data: {e}")
        return {}

# ────────── optical_flow_handler.py ──────────

# module‐level accumulators
opt_cum_x = 0
opt_cum_y = 0

def handle_optical_xy_data(data: bytearray) -> dict:
    global opt_cum_x, opt_cum_y
    try:
        text = data.decode("utf-8").strip()
        dx_str, dy_str = text.split(",")
        dx = int(dx_str)
        dy = int(dy_str)

        # update cumulative position
        opt_cum_x += dx
        opt_cum_y += dy

        data_logger.info(
            f"Optical flow: dx={dx}, dy={dy}, cum_x={opt_cum_x}, cum_y={opt_cum_y}"
        )
        return {
            "opt_dx": dx,
            "opt_dy": dy,
            "opt_cum_x": opt_cum_x,
            "opt_cum_y": opt_cum_y
        }
    except Exception as e:
        data_logger.error(f"Error parsing optical-flow data: {e}")
        return {}

# tof_handler.py

def handle_tof_data(data: bytearray) -> dict:
    try:
        text = data.decode("utf-8").strip()
        dist_str, bright_str = text.split(",")
        distance_mm = int(dist_str)
        brightness_kcps = int(bright_str)

        data_logger.info(
            f"ToF: distance={distance_mm} mm, brightness={brightness_kcps} kcps/spad"
        )
        return {
            "tof_distance_mm": distance_mm,
            "tof_brightness_kcps": brightness_kcps
        }
    except Exception as e:
        data_logger.error(f"Error parsing ToF data: {e}")
        return {}

# joystick_angle_handler.py

def handle_ankle_angle_data(data: bytearray) -> dict:
    try:
        text = data.decode("utf-8").strip()
        ax_str, ay_str = text.split(",")
        ankle_xz = float(ax_str)
        ankle_yz = float(ay_str)
        data_logger.info(f"Ankle angles: XZ={ankle_xz:.2f}°, YZ={ankle_yz:.2f}°")
        return {
            "ankle_xz": ankle_xz,
            "ankle_yz": ankle_yz
        }
    except Exception as e:
        data_logger.error(f"Error parsing ankle‑angle data: {e}")
        return {}


def handle_insole_data(data: bytearray) -> dict:
    """
    Parses the incoming insole data string (bytearray) from BLE.
    Extracts voltage values, applies gains, calculates individual pressures (FSR)
    using INITIAL_PRESSURE_SENSITIVITY, calculates summed gained voltage (FSRs),
    estimated weight (from sum), and flex angle (from flex sensor voltage).

    Args:
        data: The raw bytearray received via BLE.

    Returns:
        A dictionary containing keys for:
        - Each FSR sensor ('A0C0'...'A2C3') with its pressure value (relative to initial sensitivity).
        - 'estimated_weight' with the calculated weight value.
        - 'flex_angle' with the calculated angle value in degrees.
        Returns an empty dictionary if parsing or critical calculation fails.
    """
    try:
        data_string = data.decode('utf-8')
    except UnicodeDecodeError:
        data_logger.warning(f"Received non-UTF8 raw bytes for insole: {data}")
        return {}

    output_dict: Dict[str, float] = {}
    summed_gained_voltage = 0.0             # Accumulator for gained voltages (heatmap only)
    flex_voltage: Optional[float] = None    # Storage for the flex sensor RAW voltage
    # *** Uses INITIAL_PRESSURE_SENSITIVITY for the data stored in the buffer ***
    pressure_sensitivity = INITIAL_PRESSURE_SENSITIVITY

    part = "" # For error reporting
    try:
        parts = data_string.strip().rstrip(',').split(',')

        for part in parts:
            if ':' not in part:
                # data_logger.debug(f"Skipping malformed part (no ':'): '{part}' in '{data_string}'")
                continue

            key, value_str = part.split(':', 1)
            key = key.strip()
            value_str = value_str.strip()

            try:
                voltage = float(value_str)
            except ValueError:
                # data_logger.warning(f"Could not parse voltage value for key '{key}': '{value_str}'. Skipping.")
                continue

            # --- Check if it's the Flex Sensor ---
            if key == FLEX_SENSOR_KEY:
                flex_voltage = voltage # Store the raw voltage

            # --- Check if it's one of the Heatmap Sensors ---
            elif key in HEATMAP_KEYS:
                try:
                    sensor_index = HEATMAP_KEYS.index(key)
                    gain = DEFAULT_SENSOR_GAINS[sensor_index]
                except (ValueError, IndexError) as e:
                     # data_logger.warning(f"Error getting gain for heatmap key '{key}': {e}. Using gain=1.0.")
                     gain = 1.0 # Fallback gain

                gained_voltage = voltage * gain
                summed_gained_voltage += gained_voltage # Add to sum

                # Calculate pressure based on gained voltage and INITIAL sensitivity
                clamped_gained_voltage = max(0.0, min(ADC_MAX_VOLTAGE, gained_voltage))
                # *** This pressure is relative to the INITIAL sensitivity ***
                pressure = (clamped_gained_voltage / ADC_MAX_VOLTAGE) * pressure_sensitivity
                output_dict[key] = max(0.0, pressure) # Ensure pressure is non-negative, store FSR pressure

            # --- Else: Key is unrecognized ---
            # else: data_logger.debug(f"Skipping unrecognized key: '{key}'")

    except ValueError as e:
        data_logger.warning(f"ValueError parsing part '{part}': {e} in string: {data_string}")
        return {} # Indicate failure
    except Exception as e:
        data_logger.error(f"Error parsing data string '{data_string}' (part: '{part}'): {e}")
        return {} # Indicate failure

    # --- Perform final calculations ---

    # Weight Estimation
    output_dict['estimated_weight'] = summed_gained_voltage * VOLTAGE_TO_WEIGHT_FACTOR

    # Flex Angle Calculation
    if flex_voltage is None:
        # data_logger.debug(f"Flex sensor key '{FLEX_SENSOR_KEY}' not found in current packet. Calculating angle from default voltage 0.0.")
        flex_voltage = 0.0 # Use default voltage if not found
    flex_angle = FLEX_SLOPE_DEG_PER_VOLT * flex_voltage + FLEX_INTERCEPT_DEG
    output_dict['flex_angle'] = flex_angle

    # Check if all heatmap keys were populated (can be zero if voltage was zero/unparseable)
    for key in HEATMAP_KEYS:
        if key not in output_dict:
            output_dict[key] = 0.0 # Ensure all heatmap keys exist

    data_logger.info(f"Insole Parsed: { {k: f'{v:.1f}' for k, v in output_dict.items()} }")
    return output_dict

def handle_adc_data(data: bytearray) -> Dict[str, Any]:
    try:
        text = data.decode("utf-8", errors="ignore").strip()
        # parse phase diff (rad) and raw magnitude
        match = re.search(r"dphi:\s*(-?[\d.]+),\s*Mag:\s*(-?[\d.]+)", text)
        if not match:
            data_logger.warning(f"Could not parse ADC data format: {text}")
            return {}

        phi_diff_rad = float(match.group(1))
        raw_mag = float(match.group(2))

        # conversion constants
        A = 1.15
        COEFF = 87449.71
        PHASE_OFFSET_DEG = 66.58
        PHASE_OFFSET_RAD = PHASE_OFFSET_DEG * math.pi / 180.0

        # compute impedance magnitude (Ω)
        mag_Z_ohm = (COEFF * A**2) / (2.0 * raw_mag)
        # correct phase
        phase_Z = phi_diff_rad - PHASE_OFFSET_RAD

        # real & imag parts in ohms
        real_ohm = mag_Z_ohm * math.cos(phase_Z)
        imag_ohm = mag_Z_ohm * math.sin(phase_Z)

        # convert to kΩ
        real_kohm = real_ohm / 1000.0
        imag_kohm = imag_ohm / 1000.0

        data_logger.info(
            f"Impedance: |Z|={mag_Z_ohm:.2f} Ω, θ={phase_Z:.3f} rad → "
            f"Re={real_kohm:.3f} kΩ, Im={imag_kohm:.3f} kΩ"
        )

        return {
            "impedance_magnitude_ohm": mag_Z_ohm,
            "impedance_phase_rad": phase_Z * 180.0 / math.pi,  # Convert to degrees
            "real_part_kohm": real_kohm,
            "imag_part_kohm": imag_kohm,
        }

    except (ValueError, ZeroDivisionError) as e:
        data_logger.error(f"Error processing ADC data '{text}': {e}")
        return {}

# --- Device Configuration (Initial Device Name still set here) ---
# This object's 'name' attribute will be updated by the dropdown menu
device_config = DeviceConfig(
    name="Nano33IoT", # Initial default name
    service_uuid="19B10000-E8F2-537E-4F6C-D104768A1214",
    characteristics=[
        # BLE Characteristics
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
        # Insole Characteristic
        CharacteristicConfig(uuid="19B10002-E8F2-537E-4F6C-D104768A1214", handler=handle_insole_data,
                             produces_data_types=HEATMAP_KEYS + ['estimated_weight', 'flex_angle']), # Produces FSR pressures + weight + angle
        CharacteristicConfig(
            uuid="19B10009-E8F2-537E-4F6C-D104768A1214",
            handler=handle_adc_data,
            produces_data_types=['impedance_magnitude_ohm', 'impedance_phase_rad', 'real_part_kohm', 'imag_part_kohm']
        ),
        CharacteristicConfig(
            uuid="19B10012-E8F2-537E-4F6C-D104768A1214",
            handler=handle_optical_xy_data,
            produces_data_types=['opt_dx', 'opt_dy', 'opt_cum_x', 'opt_cum_y']
        ),
        CharacteristicConfig(
            uuid="19B10014-E8F2-537E-4F6C-D104768A1214",
            handler=handle_tof_data,
            produces_data_types=['tof_distance_mm', 'tof_brightness_kcps']
        ),
        CharacteristicConfig(
            uuid="19B10016-E8F2-537E-4F6C-D104768A1214",
            handler=handle_ankle_angle_data,
            produces_data_types=['ankle_xz', 'ankle_yz']
        ),
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
        self.tab_index: Optional[int] = None # Will be set by GuiManager during creation
        self.is_loggable: bool = config.get('enable_logging', False) # Check if logging is enabled via config
        # Basic size policy, can be overridden by subclasses or config
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.uuid_missing_overlay: Optional[QLabel] = None # Optional overlay for missing UUID message

    def get_widget(self) -> QWidget:
        #Returns the primary widget this component manages.
        return self # Default: the component itself is the widget

    def update_component(self, current_relative_time: float, is_flowing: bool):
        #Update the component's visual representation based on current data and time.
        raise NotImplementedError("Subclasses must implement update_component")

    def clear_component(self):
        #Clear the component's display and internal state.
        raise NotImplementedError("Subclasses must implement clear_component")

    def get_required_data_types(self) -> Set[str]:
        #Returns a set of data_type keys this component requires for display.
        # By default, assume a component doesn't directly require data types.
        return set()

    def get_loggable_data_types(self) -> Set[str]:
        """
        Returns a set of data_type keys this component wants to log,
        IF self.is_loggable is True.
        By default, it logs the same data types it requires for display.
        Subclasses can override this for more specific logging needs.
        """
        if self.is_loggable:
            return self.get_required_data_types()
        else:
            return set()

    def get_log_filename_suffix(self) -> str:
        """
        Returns a string suffix used to create the unique CSV filename for this component,
        IF self.is_loggable is True. The main window adds prefixes (like timestamp).
        This should be file-system safe.
        """
        if self.is_loggable:
            # Default implementation: Use class name and object ID (or position if available)
            class_name = self.__class__.__name__
            # Try to get a title from config as a better default
            title = self.config.get('title', f'Component_{id(self)}')
            # Clean the title for use in filename
            safe_suffix = re.sub(r'[^\w\-]+', '_', title).strip('_')
            return f"log_{safe_suffix}" if safe_suffix else f"log_{class_name}_{id(self)}"
        return "" # Return empty if not loggable

    def handle_missing_uuids(self, missing_uuids_for_component: Set[str]):
        """
        Called by the GuiManager when relevant UUIDs are found to be missing.
        'missing_uuids_for_component' contains only the UUIDs relevant to this specific component.
        Handles showing/hiding a generic overlay message.
        """
        if missing_uuids_for_component:
            first_missing_uuid = next(iter(missing_uuids_for_component), None)
            text_content = f"Required UUID:\n{first_missing_uuid}\nnot found!"

            if not self.uuid_missing_overlay:
                # Create overlay label centered within the component
                self.uuid_missing_overlay = QLabel(text_content, self)
                self.uuid_missing_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.uuid_missing_overlay.setStyleSheet("background-color: rgba(100, 100, 100, 200); color: white; font-weight: bold; border-radius: 5px; padding: 10px;")
                self.uuid_missing_overlay.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum) # Let it determine its size
                # Use layout to center (requires component to have a layout set)
                if self.layout():
                    # Position it manually - QLayout might be complex if overlaying
                    pass # See resizeEvent
                else:
                    logger.warning(f"Cannot auto-position missing UUID overlay for {self.__class__.__name__} as it has no layout.")
                self.uuid_missing_overlay.adjustSize() # Adjust size to content
                self.uuid_missing_overlay.raise_() # Bring to front
                self.uuid_missing_overlay.setVisible(True)
                self._position_overlay() # Initial positioning
            else:
                self.uuid_missing_overlay.setText(text_content)
                self.uuid_missing_overlay.adjustSize()
                self._position_overlay()
                self.uuid_missing_overlay.setVisible(True)
                self.uuid_missing_overlay.raise_()

        else: # No missing UUIDs for this component
            if self.uuid_missing_overlay:
                self.uuid_missing_overlay.setVisible(False)


    def resizeEvent(self, event):
        """Reposition overlay on resize."""
        super().resizeEvent(event)
        self._position_overlay()

    def _position_overlay(self):
        """Helper to center the overlay label."""
        if self.uuid_missing_overlay and self.uuid_missing_overlay.isVisible():
            overlay_size = self.uuid_missing_overlay.sizeHint()
            self_size = self.size()
            x = (self_size.width() - overlay_size.width()) // 2
            y = (self_size.height() - overlay_size.height()) // 2
            self.uuid_missing_overlay.setGeometry(x, y, overlay_size.width(), overlay_size.height())

    def showEvent(self, event):
        """ Ensure overlay is positioned correctly when widget becomes visible """
        super().showEvent(event)
        QTimer.singleShot(0, self._position_overlay) # Delay position until layout settles

# --- Specific Component Implementations ---

class TimeSeriesPlotComponent(BaseGuiComponent):
    """A GUI component that displays time-series data using pyqtgraph."""
    def __init__(self, config: Dict[str, Any], data_buffers_ref: Dict[str, List[Tuple[float, float]]], device_config_ref: DeviceConfig, parent: Optional[QWidget] = None):
        super().__init__(config, data_buffers_ref, device_config_ref, parent)

        self.plot_widget = pg.PlotWidget()
        self.plot_item: pg.PlotItem = self.plot_widget.getPlotItem()
        self.lines: Dict[str, pg.PlotDataItem] = {} # data_type -> PlotDataItem
        # self.uuid_not_found_text: Optional[pg.TextItem] = None # Use base class overlay instead
        self._required_data_types: Set[str] = set() # Internal store
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

        for dataset in self.config.get('datasets', []):
            data_type = dataset['data_type']
            self._required_data_types.add(data_type) # Use internal set
            pen = pg.mkPen(color=dataset.get('color', 'k'), width=1.5)
            line = self.plot_item.plot(pen=pen, name=dataset.get('label', data_type))
            self.lines[data_type] = line

        self.clear_component() # Initialize axes

    def get_required_data_types(self) -> Set[str]:
        # Returns the data types needed for plotting.
        return self._required_data_types

    # get_loggable_data_types is inherited from BaseGuiComponent and will use get_required_data_types

    def get_log_filename_suffix(self) -> str:
        """Overrides base method to provide a filename suffix based on the plot title."""
        if self.is_loggable:
            title = self.config.get('title', f'Plot_{id(self)}')
            safe_suffix = re.sub(r'[^\w\-]+', '_', title).strip('_')
            return f"plot_{safe_suffix}" if safe_suffix else f"plot_{id(self)}"
        return ""

    def handle_missing_uuids(self, missing_uuids_for_component: Set[str]):
        """Shows or hides the 'UUID not found' message using the base class overlay."""
        self.missing_relevant_uuids = missing_uuids_for_component
        super().handle_missing_uuids(missing_uuids_for_component) # Call base implementation for overlay

        # Request GUI update to potentially clear lines / adjust y-range if overlay is shown/hidden
        QTimer.singleShot(0, self._request_gui_update_for_yrange)

    def _request_gui_update_for_yrange(self):
         # Find the main window instance to emit the signal
         try:
             mw = next(widget for widget in QApplication.topLevelWidgets() if isinstance(widget, MainWindow))
             mw.request_plot_update_signal.emit() # Request general update which includes this plot
         except StopIteration: logger.error("Could not find MainWindow instance to request plot update.")
         except Exception as e: logger.error(f"Error requesting plot update: {e}")


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
            for data_type in self.get_required_data_types(): # Use the method here
                 uuid = self.device_config_ref.get_uuid_for_data_type(data_type)
                 if uuid and uuid not in self.missing_relevant_uuids and data_type in self.data_buffers_ref and self.data_buffers_ref[data_type]:
                     try: max_data_time = max(max_data_time, self.data_buffers_ref[data_type][-1][0])
                     except IndexError: pass

            max_time_axis = max(max_data_time, flowing_interval)
            self.plot_item.setXRange(0, max_time_axis, padding=0.02)

        # --- Update Data for Lines ---
        data_updated_in_plot = False
        # plot_has_missing_uuid_overlay = (self.uuid_missing_overlay is not None and self.uuid_missing_overlay.isVisible())
        plot_has_missing_uuid_overlay = bool(self.missing_relevant_uuids) # Simpler check


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
        if data_updated_in_plot and not plot_has_missing_uuid_overlay:
            self.plot_item.enableAutoRange(axis='y', enable=True)
        elif plot_has_missing_uuid_overlay:
             # Disable auto-ranging and set a default range if overlay is shown
             self.plot_item.enableAutoRange(axis='y', enable=False)
             # Keep existing range or set a default? Let's try setting a simple default.
             self.plot_item.setYRange(-1, 1, padding=0.1) # Example default range


    def clear_component(self):
        """Clears the plot lines and resets axes."""
        for line in self.lines.values():
            line.setData(x=[], y=[])

        # Hide overlay via base class method (handles None check)
        self.handle_missing_uuids(set())
        self.missing_relevant_uuids.clear() # Also clear internal set

        # Reset view ranges
        self.plot_item.setXRange(0, flowing_interval, padding=0.02)
        self.plot_item.setYRange(-1, 1, padding=0.1) # Reset Y range to default
        self.plot_item.enableAutoRange(axis='y', enable=True) # Re-enable Y auto-ranging


# --- HEATMAP COMPONENT ---
class PressureHeatmapComponent(BaseGuiComponent):
    """ Displays a pressure heatmap based on sensor data. """

    # --- Constants specific to this component ---
    DEFAULT_INSOLE_IMAGE_PATH = 'Sohle_rechts.png'
    DEFAULT_OUTLINE_COORDS = np.array([(186, 0), (146, 9), (108, 34), (79, 66), (59, 101), (43, 138), (30, 176), (19, 215),
                                       (11, 255), (6, 303), (2, 358), (0, 418), (3, 463), (8, 508), (14, 550), (23, 590),
                                       (34, 630), (47, 668), (60, 706), (71, 745), (82, 786), (91, 825), (95, 865),
                                       (97, 945), (94, 990), (89, 1035), (83, 1077), (76, 1121), (69, 1161), (64, 1204),
                                       (59, 1252), (56, 1293), (54, 1387), (57, 1430), (63, 1470), (70, 1512), (81, 1551),
                                       (97, 1588), (123, 1626), (152, 1654), (187, 1674), (226, 1688), (273, 1696),
                                       (314, 1696), (353, 1686), (390, 1668), (441, 1625), (466, 1591), (485, 1555),
                                       (500, 1515), (509, 1476), (515, 1431), (517, 1308), (516, 1264), (514, 1199),
                                       (512, 1141), (511, 1052), (514, 1011), (519, 969), (524, 929), (529, 887),
                                       (534, 845), (540, 801), (547, 759), (554, 719), (562, 674), (568, 634),
                                       (573, 584), (575, 536), (572, 491), (566, 451), (556, 409), (543, 369), (528, 331),
                                       (512, 294), (495, 257), (476, 220), (456, 185), (432, 150), (405, 116), (364, 73),
                                       (329, 45), (294, 23), (254, 7), (206, 0)])
    # This coordinate list remains ONLY for the FSR sensors used in the heatmap
    DEFAULT_SENSOR_COORDS = np.array([ # Order corresponds to HEATMAP_KEYS above.
        [439, 307], [129, 136], [440, 1087], [415, 567], [260, 451], [424, 1273],
        [273, 140], [116, 452], [205, 1580], [435, 925], [372, 1580]
    ])
    NUM_SENSORS = len(DEFAULT_SENSOR_COORDS) # Should match NUM_HEATMAP_SENSORS

    # --- Configuration Values (set in __init__ based on `config` arg) ---
    # These have defaults here but can be overridden via the `config` dict passed at instantiation
    DEFAULT_GRID_RESOLUTION = 10
    DEFAULT_ALPHA = 180
    DEFAULT_WINDOW_MIN = 0.0
    DEFAULT_WINDOW_MAX = 100.0
    DEFAULT_GAUSSIAN_SIGMA = 120.0
    DEFAULT_SENSITIVITY = 300.0 # Matches INITIAL_PRESSURE_SENSITIVITY
    DEFAULT_CMAP_NAME = 'jet'
    AVAILABLE_CMAPS = ['jet', 'nipy_spectral', 'gist_ncar', 'gist_rainbow', 'turbo', 'cubehelix', 'tab20b', 'tab20c']
    # https://matplotlib.org/stable/users/explain/colors/colormaps.html for more colormaps

    # --- Center of Pressure & Trail Configuration ---
    COP_MAIN_POINT_RADIUS = 8.0
    COP_MAIN_POINT_COLOR = QColor(255, 255, 255, 255) #white
    COP_TRAIL_MAX_LEN = 30
    COP_TRAIL_POINT_RADIUS = 5.0
    COP_TRAIL_COLOR = QColor(243 , 100 , 248) # bright pink
    COP_TRAIL_MAX_ALPHA = 255
    COP_TRAIL_MIN_ALPHA = 0
    COP_TRAIL_LINE_WIDTH = 7
    SPLINE_SAMPLES_PER_SEGMENT = 10
    TEXTBOX_WIDTH = 50

    def __init__(self, config: Dict[str, Any], data_buffers_ref: Dict[str, List[Tuple[float, float]]], device_config_ref: DeviceConfig, parent: Optional[QWidget] = None):
        super().__init__(config, data_buffers_ref, device_config_ref, parent)

        self._required_data_types = set(HEATMAP_KEYS) # Internal store

        # --- START: Add Size Configuration Reading ---
        # Get desired width/height from config, defaulting to None if not specified
        component_width = self.config.get('component_width', None)
        component_height = self.config.get('component_height', None)

        # Apply size constraints to the component itself, layout will handle the rest
        if component_height is not None: self.setFixedHeight(component_height)
        if component_width is not None: self.setFixedWidth(component_width)
        # If only one dimension is set, allow expansion in the other
        if component_height is not None and component_width is None:
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        elif component_width is not None and component_height is None:
            self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        elif component_width is None and component_height is None:
             self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        else: # Both fixed
             self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        # --- END: Add Size Configuration Reading ---

        # --- Load Configurable Parameters ---
        self.image_path = self.config.get('image_path', self.DEFAULT_INSOLE_IMAGE_PATH)
        self.grid_resolution = max(1, self.config.get('grid_resolution', self.DEFAULT_GRID_RESOLUTION))
        self.alpha_int = np.clip(self.config.get('alpha', self.DEFAULT_ALPHA), 0, 255)
        self.alpha_float = self.alpha_int / 255.0
        # Initial operational values (can be changed by GUI controls)
        # **** Uses the GLOBAL INITIAL_PRESSURE_SENSITIVITY as its starting point ****
        self.current_pressure_sensitivity = float(self.config.get('initial_sensitivity', INITIAL_PRESSURE_SENSITIVITY))
        self.current_gaussian_sigma = float(self.config.get('initial_gaussian_sigma', self.DEFAULT_GAUSSIAN_SIGMA))
        self.current_pressure_min = float(self.config.get('initial_window_min', self.DEFAULT_WINDOW_MIN))
        # Ensure initial max window isn't higher than initial sensitivity
        self.current_pressure_max = min(float(self.config.get('initial_window_max', self.DEFAULT_WINDOW_MAX)), self.current_pressure_sensitivity)
        self.current_cmap_name = self.config.get('initial_colormap', self.DEFAULT_CMAP_NAME)
        self.save_directory = self.config.get('snapshot_dir', "captured_pressure_maps")
        os.makedirs(self.save_directory, exist_ok=True)

        # Basic parameters for heatmap
        self.sensor_coords = self.DEFAULT_SENSOR_COORDS.astype(np.float32)
        self.pressure_values = np.zeros(self.NUM_SENSORS, dtype=np.float32) # Stores latest *rescaled* pressure values for heatmap sensors

        # Load background image
        self.original_pixmap = QPixmap(self.image_path)
        if self.original_pixmap.isNull():
            logger.error(f"Heatmap Error: loading image '{self.image_path}'. Using placeholder.")
            # Create a placeholder pixmap
            self.original_pixmap = QPixmap(300, 500) # Example size
            self.original_pixmap.fill(Qt.GlobalColor.lightGray)
            painter = QPainter(self.original_pixmap)
            painter.setPen(Qt.GlobalColor.red)
            painter.drawText(self.original_pixmap.rect(), Qt.AlignmentFlag.AlignCenter, f"Error:\nCould not load\n{os.path.basename(self.image_path)}")
            painter.end()
            self.img_width = 300
            self.img_height = 500
            # Use default coords even if image failed, mask will just cover the placeholder
        else:
            self.img_width = self.original_pixmap.width()
            self.img_height = self.original_pixmap.height()

        # Pre-computation for heatmap
        logger.info("Starting heatmap pre-computation...")
        try:
            self.mask_image = self._generate_outline_mask()
            self.mask_array = self._qimage_to_bool_mask(self.mask_image)
            self.valid_grid_pixels_y, self.valid_grid_pixels_x = self._precompute_valid_grid_pixels()
            self.num_valid_grid_points = len(self.valid_grid_pixels_x)
            if self.num_valid_grid_points == 0: logger.warning("Heatmap: No grid points inside mask!")
            self.precomputed_gaussian_factors = self._precompute_gaussian_factors() # Uses self.current_gaussian_sigma
            # Setup colormap and normalization
            try:
                self.cmap = matplotlib.colormaps[self.current_cmap_name]
            except KeyError:
                logger.warning(f"Heatmap: Colormap '{self.current_cmap_name}' not found. Using 'jet'.")
                self.current_cmap_name = 'jet' # Fallback default
                self.cmap = matplotlib.colormaps[self.current_cmap_name]
            self.norm = mcolors.Normalize(vmin=self.current_pressure_min, vmax=self.current_pressure_max, clip=True)
            # Buffers
            self.heatmap_buffer = np.zeros((self.img_height, self.img_width), dtype=np.uint32)
            self.heatmap_qimage = QImage(self.heatmap_buffer.data, self.img_width, self.img_height,
                                        self.img_width * 4, QImage.Format.Format_ARGB32_Premultiplied)
            logger.info("Heatmap pre-computation finished.")
        except Exception as e:
            logger.error(f"Heatmap pre-computation failed: {e}", exc_info=True)
            # Mark precomputation as failed to prevent errors later
            self.mask_image = None
            self.mask_array = None
            self.num_valid_grid_points = 0
            self.precomputed_gaussian_factors = None
            self.heatmap_buffer = None
            self.heatmap_qimage = None
            # Set up a dummy image label to show the error
            self.image_label = QLabel("Heatmap Precomputation Failed.\nCheck Logs.", self)
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_label.setStyleSheet("background-color: gray; color: white; font-weight: bold;")
            main_layout = QVBoxLayout(self)
            main_layout.addWidget(self.image_label)
            self.setLayout(main_layout)
            return # Stop __init__ here if precomputation failed

        # Center of Pressure & Trail state
        self.center_of_pressure: Optional[QPointF] = None
        self.cop_trail: deque[QPointF] = deque(maxlen=self.COP_TRAIL_MAX_LEN)

        # --- GUI Layout Setup ---
        self.main_layout = QVBoxLayout(self) # Main layout VERTICAL
        self.main_layout.setContentsMargins(5, 5, 5, 5) # Add some margins

        # --- Image Label (Top) ---
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(150, 250) # Min size for heatmap display
        # Let the image expand, it will be constrained by component size or window size
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_layout.addWidget(self.image_label) # Add image label first

        # --- Controls Widget (Bottom) ---
        self.controls_widget = QWidget()
        # Use a QVBoxLayout for the controls within the control widget
        self.controls_layout = QVBoxLayout(self.controls_widget)
        self.controls_layout.setContentsMargins(0, 5, 0, 0) # Add some top margin for separation

        # --- Sensitivity Control ---
        sensitivity_layout = QHBoxLayout()
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(100, 10000)
        self.sensitivity_slider.setValue(int(self.current_pressure_sensitivity))
        self.sensitivity_slider.valueChanged.connect(self._update_sensitivity_from_slider)
        self.sensitivity_textbox = QLineEdit()
        self.sensitivity_textbox.setValidator(QIntValidator(100, 10000, self))
        self.sensitivity_textbox.setText(str(int(self.current_pressure_sensitivity)))
        self.sensitivity_textbox.setFixedWidth(self.TEXTBOX_WIDTH)
        self.sensitivity_textbox.editingFinished.connect(self._update_sensitivity_from_textbox)
        sensitivity_layout.addWidget(QLabel("Sensitivity:"))
        sensitivity_layout.addWidget(self.sensitivity_slider)
        sensitivity_layout.addWidget(self.sensitivity_textbox)
        self.controls_layout.addLayout(sensitivity_layout)

        # --- Gaussian Sigma (Blur) Control ---
        sigma_layout = QHBoxLayout()
        self.sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.sigma_slider.setRange(10, 500)
        self.sigma_slider.setValue(int(self.current_gaussian_sigma))
        self.sigma_slider.valueChanged.connect(self._update_gaussian_sigma_from_slider)
        self.sigma_textbox = QLineEdit()
        self.sigma_textbox.setValidator(QIntValidator(10, 500, self))
        self.sigma_textbox.setText(str(int(self.current_gaussian_sigma)))
        self.sigma_textbox.setFixedWidth(self.TEXTBOX_WIDTH)
        self.sigma_textbox.editingFinished.connect(self._update_sigma_from_textbox)
        sigma_layout.addWidget(QLabel("Gaussian Sigma:"))
        sigma_layout.addWidget(self.sigma_slider)
        sigma_layout.addWidget(self.sigma_textbox)
        self.controls_layout.addLayout(sigma_layout)

        # --- Pressure Range Control (QRangeSlider + Textboxes) ---
        range_v_layout = QVBoxLayout()
        range_label_layout = QHBoxLayout()
        range_label_layout.addWidget(QLabel("Windowing:"))
        range_label_layout.addStretch(1)
        min_layout = QHBoxLayout(); min_layout.addWidget(QLabel("Min:"));
        self.min_pressure_textbox = QLineEdit()
        self.min_pressure_validator = QDoubleValidator(0.0, self.current_pressure_max, 1, self)
        self.min_pressure_textbox.setValidator(self.min_pressure_validator)
        self.min_pressure_textbox.setText(f"{self.current_pressure_min:.1f}")
        self.min_pressure_textbox.setFixedWidth(self.TEXTBOX_WIDTH)
        self.min_pressure_textbox.editingFinished.connect(self._update_range_from_textboxes)
        min_layout.addWidget(self.min_pressure_textbox); range_label_layout.addLayout(min_layout)
        range_label_layout.addStretch(2)
        max_layout = QHBoxLayout(); max_layout.addWidget(QLabel("Max:"))
        self.max_pressure_textbox = QLineEdit()
        self.max_pressure_validator = QDoubleValidator(0.0, self.current_pressure_sensitivity, 1, self)
        self.max_pressure_textbox.setValidator(self.max_pressure_validator)
        self.max_pressure_textbox.setText(f"{self.current_pressure_max:.1f}")
        self.max_pressure_textbox.setFixedWidth(self.TEXTBOX_WIDTH)
        self.max_pressure_textbox.editingFinished.connect(self._update_range_from_textboxes)
        max_layout.addWidget(self.max_pressure_textbox); range_label_layout.addLayout(max_layout)
        range_v_layout.addLayout(range_label_layout)
        self.pressure_range_slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.pressure_range_slider.setRange(0, int(self.current_pressure_sensitivity))
        self.pressure_range_slider.setValue((int(self.current_pressure_min), int(self.current_pressure_max)))
        self.pressure_range_slider.valueChanged.connect(self._update_pressure_range_from_slider)
        range_v_layout.addWidget(self.pressure_range_slider)
        self.controls_layout.addLayout(range_v_layout)

        # --- Colormap Selection ---
        cmap_layout = QHBoxLayout()
        self.cmap_combobox = QComboBox()
        available_cmaps = sorted(self.AVAILABLE_CMAPS)
        self.cmap_combobox.addItems(available_cmaps)
        if self.current_cmap_name in available_cmaps:
            self.cmap_combobox.setCurrentText(self.current_cmap_name)
        else:
             logger.warning(f"Heatmap: Configured cmap '{self.current_cmap_name}' not in available list. Using '{available_cmaps[0]}'.")
             self.cmap_combobox.setCurrentIndex(0)
             self._update_colormap(self.cmap_combobox.currentText()) # Update internal state
        self.cmap_combobox.currentTextChanged.connect(self._update_colormap)
        cmap_layout.addWidget(QLabel("Colormap:"))
        cmap_layout.addWidget(self.cmap_combobox)
        self.controls_layout.addLayout(cmap_layout)

        # --- Save/Snapshot Button ---
        self.save_button = QPushButton("Take Snapshot")
        self.save_button.clicked.connect(self.save_current_view)
        self.controls_layout.addWidget(self.save_button)

        # Set size policy for the controls widget (expand horizontally, take minimum vertical space)
        self.controls_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.main_layout.addWidget(self.controls_widget) # Add controls widget below the image

        self.setLayout(self.main_layout) # Set the main layout for the component

        # Initial display update if precomputation succeeded
        if self.heatmap_qimage:
             self._update_display_pixmap()

    def get_widget(self) -> QWidget:
        return self

    def get_required_data_types(self) -> Set[str]:
        return self._required_data_types

    # get_loggable_data_types is inherited from BaseGuiComponent
    # It will return HEATMAP_KEYS if logging is enabled for this component

    def get_log_filename_suffix(self) -> str:
        if self.is_loggable:
            title = self.config.get('title', 'PressureHeatmap')
            safe_suffix = re.sub(r'[^\w\-]+', '_', title).strip('_')
            return f"heatmap_{safe_suffix}" if safe_suffix else f"heatmap_{id(self)}"
        return ""

    def handle_missing_uuids(self, missing_uuids_for_component: Set[str]):
        """ Shows overlay and disables controls if the required UUID is missing. """
        is_missing = bool(missing_uuids_for_component)
        # Use base class overlay for the message
        super().handle_missing_uuids(missing_uuids_for_component)
        # Disable/Enable controls
        if hasattr(self, 'controls_widget'): # Check if controls widget exists (init might fail)
             self.controls_widget.setEnabled(not is_missing) # Disable entire control panel
        if is_missing:
             self.clear_component() # Also clear heatmap visually

    # --- Update and Clear ---
    def update_component(self, current_relative_time: float, is_flowing: bool):
        if plotting_paused: return
        if not self.heatmap_qimage: return # Skip update if precomputation failed

        # --- Get Latest Pressure Data AND RESCALE IT ---
        data_found_count = 0
        # Create a temporary array to store the pressures scaled by the *current* sensitivity
        rescaled_pressures = np.zeros(self.NUM_SENSORS, dtype=np.float32)

        for i, key in enumerate(HEATMAP_KEYS):
             if key in self.data_buffers_ref and self.data_buffers_ref[key]:
                 # Get the pressure value calculated with INITIAL_PRESSURE_SENSITIVITY
                 initial_pressure = self.data_buffers_ref[key][-1][1]

                 # --- FIX: Rescale the pressure using the current sensitivity ---
                 if INITIAL_PRESSURE_SENSITIVITY > 1e-6: # Avoid division by zero
                     current_pressure = initial_pressure * (self.current_pressure_sensitivity / INITIAL_PRESSURE_SENSITIVITY)
                 else:
                     current_pressure = 0.0 # Or handle error appropriately

                 rescaled_pressures[i] = max(0.0, current_pressure) # Apply non-negative constraint
                 # --- END FIX ---
                 data_found_count += 1
             # else: logger.debug(f"No data found for heatmap sensor {key}")

        # Only update if we actually found data for at least one sensor
        if data_found_count > 0:
            # --- Store the RESCALED pressures for rendering ---
            self.pressure_values = rescaled_pressures
            # logger.debug(f"Updating heatmap with RESCALED pressures: {self.pressure_values.round(1)}")

            # --- Heatmap Calculation and Rendering ---
            current_cop_qpoint = self._calculate_center_of_pressure()
            self.center_of_pressure = current_cop_qpoint
            if current_cop_qpoint is not None:
                is_different = True
                if self.cop_trail:
                    last_point = self.cop_trail[-1]
                    if abs(current_cop_qpoint.x() - last_point.x()) < 0.1 and abs(current_cop_qpoint.y() - last_point.y()) < 0.1: is_different = False
                if is_different: self.cop_trail.append(current_cop_qpoint)

            # Calculate pressure distribution using the rescaled pressure values and current gaussian factors
            calculated_pressures = self._calculate_pressure_fast()
            # Render the heatmap using the current windowing (norm) and colormap
            self._render_heatmap_to_buffer(calculated_pressures)
            # Update the display widget
            self._update_display_pixmap()


    def clear_component(self):
        if not self.heatmap_qimage: return # Skip if precomputation failed
        logger.info("Clearing PressureHeatmapComponent.")
        self.pressure_values.fill(0.0)
        self.center_of_pressure = None
        self.cop_trail.clear()
        # Clear the visual display
        calculated_pressures = self._calculate_pressure_fast() # Will be zeros
        self._render_heatmap_to_buffer(calculated_pressures)
        self._update_display_pixmap()
        # Hide overlay via base class method
        super().handle_missing_uuids(set())


    # --- Precomputation and Masking Methods (Internal) ---
    def _generate_outline_mask(self) -> QImage:
        mask_img = QImage(self.img_width, self.img_height, QImage.Format.Format_Grayscale8)
        mask_img.fill(Qt.GlobalColor.black)
        painter = QPainter(mask_img)
        polygon_points = [QPointF(p[0], p[1]) for p in self.DEFAULT_OUTLINE_COORDS]
        outline_polygon = QPolygonF(polygon_points)
        painter.setBrush(QBrush(Qt.GlobalColor.white))
        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.drawPolygon(outline_polygon)
        painter.end()
        return mask_img

    def _qimage_to_bool_mask(self, q_image: QImage) -> np.ndarray:
        ptr = q_image.constBits()
        byte_count = q_image.sizeInBytes()
        ptr.setsize(byte_count)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((self.img_height, self.img_width))
        bool_mask = arr > 128
        return bool_mask.copy()

    def _precompute_valid_grid_pixels(self) -> Tuple[np.ndarray, np.ndarray]:
        step = self.grid_resolution
        y_coords = np.arange(0, self.img_height, step, dtype=int)
        x_coords = np.arange(0, self.img_width, step, dtype=int)
        grid_y, grid_x = np.meshgrid(y_coords, x_coords, indexing='ij')
        grid_y_flat = grid_y.ravel()
        grid_x_flat = grid_x.ravel()
        # Ensure mask_array is valid before indexing
        if self.mask_array is None or self.mask_array.size == 0:
             logger.error("Heatmap: Mask array not available during valid grid pixel precomputation.")
             return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        # Clip coordinates to be within mask bounds before checking
        valid_indices = (grid_y_flat >= 0) & (grid_y_flat < self.img_height) & \
                        (grid_x_flat >= 0) & (grid_x_flat < self.img_width)
        grid_y_flat = grid_y_flat[valid_indices]
        grid_x_flat = grid_x_flat[valid_indices]

        # Check if any valid indices remain after bounding
        if grid_y_flat.size == 0:
             return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        is_inside = self.mask_array[grid_y_flat, grid_x_flat]
        valid_pixels_y = grid_y_flat[is_inside]
        valid_pixels_x = grid_x_flat[is_inside]
        return valid_pixels_y.astype(np.float32), valid_pixels_x.astype(np.float32)

    def _precompute_gaussian_factors(self) -> np.ndarray:
        if self.num_valid_grid_points == 0: return np.empty((self.NUM_SENSORS, 0), dtype=np.float32)
        grid_y_valid = self.valid_grid_pixels_y
        grid_x_valid = self.valid_grid_pixels_x
        factors = np.zeros((self.NUM_SENSORS, self.num_valid_grid_points), dtype=np.float32)
        sigma = self.current_gaussian_sigma # Use the current sigma
        two_sigma_sq = 2.0 * (sigma ** 2)
        if two_sigma_sq <= 1e-9:
             logger.warning(f"Heatmap: Gaussian sigma ({sigma}) resulted in near-zero denominator. Clamping.")
             two_sigma_sq = 1e-9
        dy_sq = (self.sensor_coords[:, 1, np.newaxis] - grid_y_valid)**2
        dx_sq = (self.sensor_coords[:, 0, np.newaxis] - grid_x_valid)**2
        dist_sq = dy_sq + dx_sq
        factors = np.exp(-dist_sq / two_sigma_sq)
        return factors

    # --- Pressure Calculation Methods (Internal) ---
    def _calculate_pressure_fast(self) -> np.ndarray:
        # This function uses self.pressure_values which are *already rescaled* in update_component
        # It also uses self.precomputed_gaussian_factors which are updated when sigma changes
        if self.num_valid_grid_points == 0 or self.precomputed_gaussian_factors is None or self.precomputed_gaussian_factors.size == 0:
             # logger.debug("Skipping calculate_pressure_fast: No grid points or factors.")
             return np.array([], dtype=np.float32)
        # Ensure pressure_values and factors have compatible shapes
        if self.pressure_values.shape[0] != self.precomputed_gaussian_factors.shape[0]:
             logger.error(f"Heatmap: Mismatch between pressure values ({self.pressure_values.shape[0]}) and gaussian factors ({self.precomputed_gaussian_factors.shape[0]}). Skipping calculation.")
             return np.array([], dtype=np.float32)

        current_pressures = self.pressure_values[:, np.newaxis] # Use the RESCALED pressures
        weighted_factors = current_pressures * self.precomputed_gaussian_factors # Use the CURRENT factors
        calculated_pressures_masked = np.sum(weighted_factors, axis=0)
        # logger.debug(f"Calculated pressures min/max: {np.min(calculated_pressures_masked):.1f}/{np.max(calculated_pressures_masked):.1f}")
        return calculated_pressures_masked

    def _calculate_center_of_pressure(self) -> Optional[QPointF]:
        # Use the current internal *rescaled* pressure values
        pressures = np.maximum(self.pressure_values, 0.0)
        total_pressure = np.sum(pressures)
        if total_pressure < 1e-6: return None
        sensor_x = self.sensor_coords[:, 0]
        sensor_y = self.sensor_coords[:, 1]
        cop_x = np.sum(pressures * sensor_x) / total_pressure
        cop_y = np.sum(pressures * sensor_y) / total_pressure
        return QPointF(cop_x, cop_y)

    # --- Heatmap Rendering Method (Internal) ---
    def _render_heatmap_to_buffer(self, calculated_pressures_masked: np.ndarray):
        if self.heatmap_buffer is None or self.heatmap_qimage is None: return # Skip if precomputation failed
        self.heatmap_buffer.fill(0) # Clear buffer (transparent)
        if calculated_pressures_masked is None or calculated_pressures_masked.size == 0: return
        # Determine which grid points have pressure above the minimum threshold
        draw_mask = calculated_pressures_masked >= self.current_pressure_min # Use >= to include min
        if not np.any(draw_mask): return

        valid_pressures = calculated_pressures_masked[draw_mask]
        valid_y_coords = self.valid_grid_pixels_y[draw_mask].astype(int)
        valid_x_coords = self.valid_grid_pixels_x[draw_mask].astype(int)

        # Normalize and colorize only the points that need drawing
        norm_pressures = self.norm(valid_pressures) # Apply current normalization
        rgba_colors_float = self.cmap(norm_pressures) # Apply current colormap

        # Apply alpha and pre-multiply
        alpha_f = self.alpha_float
        # Avoid in-place modification if rgba_colors_float is needed elsewhere unmodified
        rgba_premult_float = rgba_colors_float.copy()
        rgba_premult_float[:, 0] *= alpha_f
        rgba_premult_float[:, 1] *= alpha_f
        rgba_premult_float[:, 2] *= alpha_f
        # Convert to uint8 for QImage buffer
        rgba_premult_uint8 = (rgba_premult_float * 255).astype(np.uint8)

        # Combine into ARGB uint32 format (faster for buffer assignment)
        A = np.uint32(self.alpha_int) # Use the integer alpha directly
        R = rgba_premult_uint8[:, 0].astype(np.uint32)
        G = rgba_premult_uint8[:, 1].astype(np.uint32)
        B = rgba_premult_uint8[:, 2].astype(np.uint32)
        argb_values = (A << 24) | (R << 16) | (G << 8) | B

        # Draw squares onto the buffer efficiently
        draw_size = self.grid_resolution
        half_draw_size = draw_size // 2
        if draw_size == 1:
             # Direct assignment if grid resolution is 1 pixel
             # Ensure coordinates are within buffer bounds before assigning
             valid_coords_mask = (valid_y_coords >= 0) & (valid_y_coords < self.img_height) & \
                                (valid_x_coords >= 0) & (valid_x_coords < self.img_width)
             self.heatmap_buffer[valid_y_coords[valid_coords_mask], valid_x_coords[valid_coords_mask]] = argb_values[valid_coords_mask]
        else:
            # Iterate and draw squares for resolutions > 1
            h, w = self.img_height, self.img_width
            target_buffer = self.heatmap_buffer # Local reference for potential speedup
            for i in range(len(valid_pressures)):
                y, x = valid_y_coords[i], valid_x_coords[i]
                color = argb_values[i]
                # Calculate square bounds, clipping to image dimensions
                y_start = max(0, y - half_draw_size)
                y_end = min(h, y + draw_size - half_draw_size)
                x_start = max(0, x - half_draw_size)
                x_end = min(w, x + draw_size - half_draw_size)
                # Assign color block if valid bounds
                if y_start < y_end and x_start < x_end:
                    target_buffer[y_start:y_end, x_start:x_end] = color

    # --- CoP Trail Calculation Helpers (Internal) ---
    def _calculate_catmull_rom_point(self, t: float, p0: QPointF, p1: QPointF, p2: QPointF, p3: QPointF) -> QPointF:
        t2 = t * t; t3 = t2 * t
        x = 0.5 * ( (2 * p1.x()) + (-p0.x() + p2.x()) * t + (2 * p0.x() - 5 * p1.x() + 4 * p2.x() - p3.x()) * t2 + (-p0.x() + 3 * p1.x() - 3 * p2.x() + p3.x()) * t3 )
        y = 0.5 * ( (2 * p1.y()) + (-p0.y() + p2.y()) * t + (2 * p0.y() - 5 * p1.y() + 4 * p2.y() - p3.y()) * t2 + (-p0.y() + 3 * p1.y() - 3 * p2.y() + p3.y()) * t3 )
        return QPointF(x, y)

    def _get_spline_segment_points(self, p0: QPointF, p1: QPointF, p2: QPointF, p3: QPointF) -> List[QPointF]:
        points = []
        for i in range(self.SPLINE_SAMPLES_PER_SEGMENT + 1):
            t = i / self.SPLINE_SAMPLES_PER_SEGMENT
            points.append(self._calculate_catmull_rom_point(t, p0, p1, p2, p3))
        return points

    # --- Heatmap/CoP Drawing Method (Internal) ---
    def _update_display_pixmap(self):
        if not self.heatmap_qimage or self.original_pixmap.isNull():
             # logger.debug("Skipping display pixmap update: heatmap image or background missing.")
             return # Don't update if essential components missing

        # Get the target rectangle of the QLabel for scaling
        target_rect = self.image_label.contentsRect()
        if target_rect.isEmpty():
             # logger.debug("Skipping display pixmap update: target label rect is empty.")
             return # Don't try to scale into nothing

        # Create the final pixmap with a transparent background
        final_pixmap = QPixmap(target_rect.size())
        final_pixmap.fill(Qt.GlobalColor.transparent)
        painter_final = QPainter(final_pixmap)

        # Scale the background pixmap to fit the label
        scaled_background = self.original_pixmap.scaled(target_rect.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        # Calculate position to center the scaled background
        x_bg = (target_rect.width() - scaled_background.width()) / 2
        y_bg = (target_rect.height() - scaled_background.height()) / 2

        # Draw the scaled background onto the final pixmap
        painter_final.drawPixmap(int(x_bg), int(y_bg), scaled_background)

        # Now, prepare to overlay the heatmap and CoP onto the *original sized* combined image first
        # This requires drawing onto a pixmap matching the original image dimensions
        combined_orig_size_pixmap = self.original_pixmap.copy()
        painter_orig = QPainter(combined_orig_size_pixmap)
        painter_orig.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Draw the heatmap (using the pre-rendered QImage buffer)
        painter_orig.drawImage(0, 0, self.heatmap_qimage)

        # 2. Draw CoP Trail (Spline + Points)
        num_trail_points = len(self.cop_trail)
        trail_list = list(self.cop_trail)
        if num_trail_points >= 2:
            spline_pen = QPen(self.COP_TRAIL_COLOR)
            spline_pen.setWidthF(self.COP_TRAIL_LINE_WIDTH)
            spline_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            spline_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter_orig.setBrush(Qt.BrushStyle.NoBrush) # Ensure no fill for the polyline

            for i in range(num_trail_points - 1):
                p1 = trail_list[i]; p2 = trail_list[i+1]
                p0 = trail_list[i-1] if i > 0 else p1 # Handle start endpoint
                p3 = trail_list[i+2] if i < num_trail_points - 2 else p2 # Handle end endpoint

                # Calculate alpha based on position in the trail (fades out)
                alpha_fraction = (i + 1) / (num_trail_points - 1) if num_trail_points > 1 else 1.0
                current_alpha = int(self.COP_TRAIL_MIN_ALPHA + alpha_fraction * (self.COP_TRAIL_MAX_ALPHA - self.COP_TRAIL_MIN_ALPHA))
                current_alpha = max(0, min(255, current_alpha)) # Clamp alpha

                faded_line_color = QColor(self.COP_TRAIL_COLOR)
                faded_line_color.setAlpha(current_alpha)
                spline_pen.setColor(faded_line_color)
                painter_orig.setPen(spline_pen)

                # Calculate and draw the Catmull-Rom spline segment
                segment_points = self._get_spline_segment_points(p0, p1, p2, p3)
                if segment_points:
                    poly = QPolygonF(segment_points)
                    painter_orig.drawPolyline(poly)

        # 3. Draw CoP Trail Points (Circles)
        if num_trail_points > 0:
            point_pen = QPen(Qt.PenStyle.NoPen) # No outline for points
            point_brush = QBrush(self.COP_TRAIL_COLOR) # Base color
            painter_orig.setPen(point_pen)

            for i in range(num_trail_points):
                point_pos = trail_list[i]
                # Calculate alpha based on position (same logic as spline)
                alpha_fraction = i / (num_trail_points - 1) if num_trail_points > 1 else 1.0
                current_alpha = int(self.COP_TRAIL_MIN_ALPHA + alpha_fraction * (self.COP_TRAIL_MAX_ALPHA - self.COP_TRAIL_MIN_ALPHA))
                current_alpha = max(0, min(255, current_alpha))

                faded_point_color = QColor(self.COP_TRAIL_COLOR)
                faded_point_color.setAlpha(current_alpha)
                point_brush.setColor(faded_point_color)
                painter_orig.setBrush(point_brush)

                painter_orig.drawEllipse(point_pos, self.COP_TRAIL_POINT_RADIUS, self.COP_TRAIL_POINT_RADIUS)

        # 4. Draw Current CoP Point (Main White Dot)
        if self.center_of_pressure is not None:
             main_pen = QPen(Qt.PenStyle.NoPen) # No outline
             main_brush = QBrush(self.COP_MAIN_POINT_COLOR) # Solid white
             painter_orig.setPen(main_pen)
             painter_orig.setBrush(main_brush)
             painter_orig.drawEllipse(self.center_of_pressure, self.COP_MAIN_POINT_RADIUS, self.COP_MAIN_POINT_RADIUS)

        painter_orig.end() # Finish drawing on the original size pixmap

        # Scale the combined original size pixmap (with heatmap+CoP) down to fit the label
        scaled_combined = combined_orig_size_pixmap.scaled(target_rect.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        # Calculate position to center the scaled combined image on the final pixmap
        x_comb = (target_rect.width() - scaled_combined.width()) / 2
        y_comb = (target_rect.height() - scaled_combined.height()) / 2

        # Draw the scaled combined image OVER the scaled background
        painter_final.drawPixmap(int(x_comb), int(y_comb), scaled_combined)
        painter_final.end() # Finish drawing on the final pixmap

        # Set the final pixmap on the label
        self.image_label.setPixmap(final_pixmap)


    # --- Slot for Save Button ---
    def save_current_view(self):
        """ Saves the current heatmap view including CoP to a file. """
        if self.original_pixmap.isNull() or not self.heatmap_qimage:
             logger.warning("Cannot save snapshot: Background or heatmap data missing.")
             # Optionally show a message box to the user
             # QMessageBox.warning(self, "Snapshot Error", "Cannot save snapshot.\nBackground image or heatmap data missing.")
             return

        # Create a pixmap matching the *original* dimensions to draw onto
        save_pixmap = self.original_pixmap.copy()
        save_painter = QPainter(save_pixmap)
        save_painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw heatmap QImage onto the background copy
        save_painter.drawImage(0, 0, self.heatmap_qimage)

        # Draw CoP Trail (Spline) - Reuse logic from _update_display_pixmap
        num_trail_points = len(self.cop_trail); trail_list = list(self.cop_trail)
        if num_trail_points >= 2:
            spline_pen = QPen(self.COP_TRAIL_COLOR); spline_pen.setWidthF(self.COP_TRAIL_LINE_WIDTH); spline_pen.setCapStyle(Qt.PenCapStyle.RoundCap); spline_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            save_painter.setBrush(Qt.BrushStyle.NoBrush)
            for i in range(num_trail_points - 1):
                p1 = trail_list[i]; p2 = trail_list[i+1]; p0 = trail_list[i-1] if i > 0 else p1; p3 = trail_list[i+2] if i < num_trail_points - 2 else p2
                alpha_fraction = (i + 1) / (num_trail_points - 1) if num_trail_points > 1 else 1.0; current_alpha = int(self.COP_TRAIL_MIN_ALPHA + alpha_fraction * (self.COP_TRAIL_MAX_ALPHA - self.COP_TRAIL_MIN_ALPHA)); current_alpha = max(0, min(255, current_alpha))
                faded_line_color = QColor(self.COP_TRAIL_COLOR); faded_line_color.setAlpha(current_alpha); spline_pen.setColor(faded_line_color); save_painter.setPen(spline_pen)
                segment_points = self._get_spline_segment_points(p0, p1, p2, p3)
                if segment_points: poly = QPolygonF(segment_points); save_painter.drawPolyline(poly)

        # Draw CoP Trail Points
        if num_trail_points > 0:
            point_pen = QPen(Qt.PenStyle.NoPen); point_brush = QBrush(self.COP_TRAIL_COLOR)
            save_painter.setPen(point_pen)
            for i in range(num_trail_points):
                point_pos = trail_list[i]; alpha_fraction = i / (num_trail_points - 1) if num_trail_points > 1 else 1.0; current_alpha = int(self.COP_TRAIL_MIN_ALPHA + alpha_fraction * (self.COP_TRAIL_MAX_ALPHA - self.COP_TRAIL_MIN_ALPHA)); current_alpha = max(0, min(255, current_alpha))
                faded_point_color = QColor(self.COP_TRAIL_COLOR); faded_point_color.setAlpha(current_alpha); point_brush.setColor(faded_point_color); save_painter.setBrush(point_brush)
                save_painter.drawEllipse(point_pos, self.COP_TRAIL_POINT_RADIUS, self.COP_TRAIL_POINT_RADIUS)

        # Draw Current CoP Point
        if self.center_of_pressure is not None:
             main_pen = QPen(Qt.PenStyle.NoPen); main_brush = QBrush(self.COP_MAIN_POINT_COLOR); save_painter.setPen(main_pen); save_painter.setBrush(main_brush)
             save_painter.drawEllipse(self.center_of_pressure, self.COP_MAIN_POINT_RADIUS, self.COP_MAIN_POINT_RADIUS)

        save_painter.end()

        # Generate filename and save
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"pressure_map_{timestamp}.png"
        filepath = os.path.join(self.save_directory, filename)
        success = save_pixmap.save(filepath, "PNG")
        if success:
             logger.info(f"Successfully saved snapshot to: {filepath}")
             # Optional: Show confirmation to user
             # QMessageBox.information(self, "Snapshot Saved", f"Snapshot saved to:\n{filepath}")
        else:
             logger.error(f"Failed to save snapshot to: {filepath}")
             # Optional: Show error to user
             # QMessageBox.critical(self, "Snapshot Error", f"Failed to save snapshot to:\n{filepath}")

    # --- Sensitivity Control Logic (Internal Slots) ---
    def _update_sensitivity(self, new_value: float):
        """Core logic to update sensitivity state and dependent controls."""
        min_sens, max_sens = self.sensitivity_slider.minimum(), self.sensitivity_slider.maximum()
        new_value = max(float(min_sens), min(float(max_sens), new_value)) # Clamp to slider range
        value_int = int(new_value)
        new_sensitivity_float = float(value_int)

        if abs(self.current_pressure_sensitivity - new_sensitivity_float) < 1e-6: return # No change

        self.current_pressure_sensitivity = new_sensitivity_float
        logger.info(f"Heatmap sensitivity updated to: {self.current_pressure_sensitivity}")

        # Update slider and textbox (preventing signal loops)
        self.sensitivity_slider.blockSignals(True); self.sensitivity_slider.setValue(value_int); self.sensitivity_slider.blockSignals(False)
        self.sensitivity_textbox.blockSignals(True); self.sensitivity_textbox.setText(str(value_int)); self.sensitivity_textbox.blockSignals(False)

        # Update range slider's maximum and its validator
        self.pressure_range_slider.blockSignals(True)
        current_slider_min_range, _ = self.pressure_range_slider.value() # Keep current min window val
        self.pressure_range_slider.setRange(0, value_int) # Max range is sensitivity
        # Try to preserve window, clamping if necessary
        new_max_window = min(self.current_pressure_max, self.current_pressure_sensitivity)
        new_min_window = min(self.current_pressure_min, new_max_window)
        # Only set value if it differs from current to avoid recursive updates
        if self.pressure_range_slider.value() != (int(new_min_window), int(new_max_window)):
             self.pressure_range_slider.setValue((int(new_min_window), int(new_max_window)))
        self.pressure_range_slider.blockSignals(False)

        # Update the validator for the MAX pressure textbox
        self.max_pressure_validator.setTop(new_sensitivity_float)
        self.max_pressure_textbox.setValidator(self.max_pressure_validator)

        # Update the text for the MAX pressure textbox if it exceeds the new sensitivity
        if float(self.max_pressure_textbox.text()) > new_sensitivity_float:
            self.max_pressure_textbox.blockSignals(True)
            self.max_pressure_textbox.setText(f"{new_sensitivity_float:.1f}")
            self.max_pressure_textbox.blockSignals(False)


        # Also re-call the _update_pressure_range to ensure normalization/textboxes are updated
        # This uses the potentially clamped window values from above
        # This needs to happen *after* the max textbox validator/value is potentially updated
        self._update_pressure_range(new_min_window, new_max_window)


    def _update_sensitivity_from_slider(self, value):
        self._update_sensitivity(float(value))

    def _update_sensitivity_from_textbox(self):
        try: value = float(self.sensitivity_textbox.text())
        except ValueError: value = self.current_pressure_sensitivity # Revert on bad input
        self._update_sensitivity(value)

    # --- Gaussian Sigma Control Logic (Internal Slots) ---
    def _update_gaussian_sigma(self, new_value: float):
        min_sigma, max_sigma = self.sigma_slider.minimum(), self.sigma_slider.maximum()
        new_value = max(float(min_sigma), min(float(max_sigma), new_value))
        value_int = int(new_value)
        new_sigma_float = float(value_int)

        if abs(self.current_gaussian_sigma - new_sigma_float) < 1e-6: return # No change

        self.current_gaussian_sigma = new_sigma_float
        logger.info(f"Heatmap Gaussian sigma updated to: {self.current_gaussian_sigma}")

        self.sigma_slider.blockSignals(True); self.sigma_slider.setValue(value_int); self.sigma_slider.blockSignals(False)
        self.sigma_textbox.blockSignals(True); self.sigma_textbox.setText(str(value_int)); self.sigma_textbox.blockSignals(False)

        self._recompute_gaussian_factors() # Recalculate factors when sigma changes

    def _update_gaussian_sigma_from_slider(self, value):
        self._update_gaussian_sigma(float(value))

    def _update_sigma_from_textbox(self):
        try: value = float(self.sigma_textbox.text())
        except ValueError: value = self.current_gaussian_sigma # Revert on bad input
        self._update_gaussian_sigma(value)


    def _recompute_gaussian_factors(self):
        logger.debug("Heatmap: Recomputing Gaussian factors...")
        try:
            # Recalculate using the current sigma
            new_factors = self._precompute_gaussian_factors()
            # Only update if the calculation was successful (returned a valid array)
            if new_factors is not None and new_factors.size > 0:
                 self.precomputed_gaussian_factors = new_factors
                 logger.debug("Heatmap: Gaussian factors recomputed.")
            else:
                 logger.warning("Heatmap: Gaussian factor recomputation resulted in empty factors. Check grid/mask.")
        except Exception as e:
             logger.error(f"Heatmap: Failed to recompute gaussian factors: {e}")
             # Optionally invalidate factors on error, e.g., self.precomputed_gaussian_factors = None


    # --- Pressure Range Control Logic (Internal Slots) ---
    def _update_pressure_range(self, min_val: float, max_val: float):
        """Core logic to update pressure window min/max and related controls."""
        effective_max_limit = self.current_pressure_sensitivity # Window cannot exceed sensitivity
        slider_min_limit = 0.0 # Typically 0

        # Clamp max_val first
        max_val = max(slider_min_limit, min(effective_max_limit, max_val))
        # Clamp min_val based on slider min and the clamped max_val
        min_val = max(slider_min_limit, min(max_val, min_val))

        # Check if values actually changed
        if abs(self.current_pressure_min - min_val) < 1e-6 and abs(self.current_pressure_max - max_val) < 1e-6:
            return # No change

        self.current_pressure_min = min_val
        self.current_pressure_max = max_val

        # Update normalization object used for rendering
        self.norm = mcolors.Normalize(vmin=self.current_pressure_min, vmax=self.current_pressure_max, clip=True)
        logger.info(f"Heatmap pressure window updated to: Min={self.current_pressure_min:.1f}, Max={self.current_pressure_max:.1f}")

        # Update range slider (prevent signal loops)
        self.pressure_range_slider.blockSignals(True)
        # Check if the value needs updating before setting it
        if self.pressure_range_slider.value() != (int(min_val), int(max_val)):
            self.pressure_range_slider.setValue((int(min_val), int(max_val)))
        self.pressure_range_slider.blockSignals(False)


        # Update min textbox validator (max is the current window max)
        self.min_pressure_validator.setTop(self.current_pressure_max)
        self.min_pressure_textbox.setValidator(self.min_pressure_validator) # Reapply validator

        # Update textboxes (prevent signal loops)
        # Check if text needs updating before setting it
        if self.min_pressure_textbox.text() != f"{self.current_pressure_min:.1f}":
             self.min_pressure_textbox.blockSignals(True); self.min_pressure_textbox.setText(f"{self.current_pressure_min:.1f}"); self.min_pressure_textbox.blockSignals(False)

        # Max textbox validator max limit already updated by sensitivity change if needed
        if self.max_pressure_textbox.text() != f"{self.current_pressure_max:.1f}":
             self.max_pressure_textbox.blockSignals(True); self.max_pressure_textbox.setText(f"{self.current_pressure_max:.1f}"); self.max_pressure_textbox.blockSignals(False)


        # Trigger a visual update since normalization changed
        # (Assuming update_component will be called shortly by the main timer anyway)


    def _update_pressure_range_from_slider(self, value_tuple):
        min_val_int, max_val_int = value_tuple # QRangeSlider emits an int tuple
        self._update_pressure_range(float(min_val_int), float(max_val_int))

    def _update_range_from_textboxes(self):
        try: min_val_f = float(self.min_pressure_textbox.text())
        except ValueError: min_val_f = self.current_pressure_min # Revert on bad input
        try: max_val_f = float(self.max_pressure_textbox.text())
        except ValueError: max_val_f = self.current_pressure_max # Revert on bad input
        self._update_pressure_range(min_val_f, max_val_f)

    # --- Colormap Selection Slot (Internal) ---
    def _update_colormap(self, cmap_name):
        try:
            self.cmap = matplotlib.colormaps[cmap_name]
            self.current_cmap_name = cmap_name
            logger.info(f"Heatmap colormap changed to: {self.current_cmap_name}")
             # Trigger a visual update since colormap changed
            # (Assuming update_component will be called shortly by the main timer anyway)
        except KeyError:
             logger.error(f"Heatmap: Invalid colormap selected in dropdown: {cmap_name}. Keeping previous.")
             # Revert dropdown to previous valid value
             self.cmap_combobox.blockSignals(True)
             self.cmap_combobox.setCurrentText(self.current_cmap_name)
             self.cmap_combobox.blockSignals(False)


# --- SingleValueDisplayComponent ---
class SingleValueDisplayComponent(BaseGuiComponent):
    def __init__(self, config: Dict[str, Any],
                 data_buffers_ref: Dict[str, List[Tuple[float, float]]],
                 device_config_ref: DeviceConfig,
                 parent: Optional[QWidget] = None):
        super().__init__(config, data_buffers_ref, device_config_ref, parent)
        self.data_type_to_monitor = self.config.get("data_type")
        self.display_label = self.config.get("label", self.data_type_to_monitor)
        self.format_string = self.config.get("format", "{:.2f}")
        self.units = self.config.get("units", "")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.title_label = QLabel(f"{self.display_label}: ")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_label = QLabel("--")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

        self.setLayout(layout)

        self.setFixedHeight(30)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


    def update_component(self, current_relative_time: float, is_flowing: bool):
        if plotting_paused: self.value_label.setText("(Paused)"); return

        value_found = False
        is_uuid_missing = False # Flag to check if UUID is missing

        if self.data_type_to_monitor:
             # Check if UUID is missing
             uuid = self.device_config_ref.get_uuid_for_data_type(self.data_type_to_monitor)
             # Attempt to access the missing UUID set from the GuiManager via parent traversal
             missing_uuids = set()
             try:
                 # Traverse up widget hierarchy to find MainWindow -> GuiManager
                 current_widget = self
                 while current_widget is not None:
                     if isinstance(current_widget, MainWindow):
                         missing_uuids = current_widget.gui_manager.active_missing_uuids
                         break
                     current_widget = current_widget.parent()
             except AttributeError:
                  logger.warning("Could not access active_missing_uuids from parent.", exc_info=False)


             if uuid and uuid in missing_uuids:
                 self.value_label.setText("(UUID Missing)")
                 is_uuid_missing = True # Set the flag
                 # return # Don't return yet, check buffer below just in case

             if not is_uuid_missing and self.data_type_to_monitor in self.data_buffers_ref:
                 buffer = self.data_buffers_ref[self.data_type_to_monitor]
                 if buffer:
                     latest_value = buffer[-1][1]
                     try: display_text = self.format_string.format(latest_value) + f" {self.units}"
                     except (ValueError, TypeError): display_text = f"{latest_value} {self.units}" # Fallback if format fails
                     self.value_label.setText(display_text)
                     value_found = True

        # Only show '--' if not paused, not missing UUID, and value not found
        if not value_found and not is_uuid_missing and not plotting_paused:
             self.value_label.setText("--")


    def clear_component(self):
        self.value_label.setText("--")

    def get_required_data_types(self) -> Set[str]:
        # This component requires only the specified data type for display
        return {self.data_type_to_monitor} if self.data_type_to_monitor else set()

    # get_loggable_data_types will inherit default behavior (logs required types if enabled)

    def get_log_filename_suffix(self) -> str:
        """Provides a filename suffix based on the component's label or data type."""
        if self.is_loggable:
            # Use label if available, otherwise data type
            name_part = self.display_label if self.display_label else self.data_type_to_monitor
            if not name_part: name_part = f'ValueDisplay_{id(self)}'
            safe_suffix = re.sub(r'[^\w\-]+', '_', name_part).strip('_')
            return f"value_{safe_suffix}" if safe_suffix else f"value_{id(self)}"
        return ""

    def handle_missing_uuids(self, missing_uuids_for_component: Set[str]):
        """ Overrides base to update display immediately if UUID goes missing/found """
        # Update the display if our specific data type's UUID is affected
        is_missing = False
        if self.data_type_to_monitor:
            uuid = self.device_config_ref.get_uuid_for_data_type(self.data_type_to_monitor)
            if uuid and uuid in missing_uuids_for_component:
                is_missing = True

        if is_missing:
            self.value_label.setText("(UUID Missing)")
        else:
             # Force update from buffer if UUID is no longer missing
             # Use dummy time, is_flowing=False to get latest non-flowing value
             # Need a way to get current time if available, otherwise use 0
             current_time = 0.0
             if start_time: current_time = (datetime.datetime.now() - start_time).total_seconds()
             self.update_component(current_time, False)


# --- Tab Layout Configuration ---
# List of dictionaries, each defining a tab.
# Each dictionary contains 'tab_title' and 'layout'.
# 'layout' is a list of component definitions for that tab's grid.
# 'enable_logging': True/False to component configs where logging is desired
tab_configs = [
    {
        'tab_title': 'Insole View',
        'layout': [
            {   'component_class': PressureHeatmapComponent,
                'row': 0, 'col': 0, 'rowspan': 2, # Heatmap takes more vertical space
                'config': { # Configuration for the heatmap
                    'title': 'Insole Pressure Heatmap', # Used for log filename
                    'component_width': 450,
                    'component_height': 700,
                    # Optional: override defaults like initial_sensitivity, image_path etc.
                     'initial_sensitivity': 500, # Example override
                     'initial_gaussian_sigma': 110, # Example override
                    # 'initial_colormap': 'turbo',
                    # ... other config options, check component class for details
                    'grid_resolution': 12, # in pixels, default is 10
                    'enable_logging': True # Log the 11 FSR pressure values (keys from HEATMAP_KEYS)
                }
            },
            {   'component_class': TimeSeriesPlotComponent,
                'row': 0, 'col': 1, # Plot for estimated weight
                'config': {
                    'title': 'Estimated Weight vs Time', 'xlabel': 'Time [s]', 'ylabel': 'Estimated Weight [kg]', # Adjusted units
                    'plot_height': 450, # Adjust height as needed
                    'plot_width': 650, # Adjust width as needed
                    'datasets': [{'data_type': 'estimated_weight', 'label': 'Weight', 'color': 'b'}],
                    'enable_logging': True # Log estimated weight
                }
            },
        ]
    },
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
                                 {'data_type': 'orientation_z', 'label': 'Z (Yaw)', 'color': 'b'}],
                    'enable_logging': True # Enable logging for this specific plot
                }
            },
            {   'component_class': TimeSeriesPlotComponent,
                'row': 0, 'col': 1,
                'config': {
                    'title': 'Angular Velocity vs Time','xlabel': 'Time [s]','ylabel': 'Degrees/s',
                    'plot_height': 300, 'plot_width': 600,
                    'datasets': [{'data_type': 'gyro_x', 'label': 'X', 'color': 'r'},
                                 {'data_type': 'gyro_y', 'label': 'Y', 'color': 'g'},
                                 {'data_type': 'gyro_z', 'label': 'Z', 'color': 'b'}],
                    'enable_logging': False # Logging disabled for this plot (default if key omitted)
                }
            },
            {   'component_class': SingleValueDisplayComponent,
                'row': 1, 'col': 0,
                'config': {
                    'label': 'Current Roll', # Display name
                    'data_type': 'orientation_x', # Data source
                    'format': '{:.1f}', # Format string
                    'units': '°', # Units string
                    'enable_logging': True # Also log this single value
                }
            },
            {   'component_class': SingleValueDisplayComponent,
                'row': 1, 'col': 1,
                'config': {
                    'label': 'Current Yaw Rate',
                    'data_type': 'gyro_z',
                    'format': '{:.1f}',
                    'units': '°/s',
                    'enable_logging': False # Don't log this one
                }
            }
        ]
    },
    {
        'tab_title': 'IMU Acceleration',
        'layout': [
            {   'component_class': TimeSeriesPlotComponent,
                'row': 0, 'col': 0,
                'config': {
                    'title': 'Linear Acceleration vs Time','xlabel': 'Time [s]','ylabel': 'm/s²',
                    'plot_height': 300,
                    # No size constraints -> uses default Expanding policy
                    'datasets': [{'data_type': 'lin_accel_x', 'label': 'X', 'color': 'r'},
                                 {'data_type': 'lin_accel_y', 'label': 'Y', 'color': 'g'},
                                 {'data_type': 'lin_accel_z', 'label': 'Z', 'color': 'b'}],
                    'enable_logging': True # Log this plot's data
                }
            },
            {   'component_class': TimeSeriesPlotComponent,
                'row': 1, 'col': 0,
                'config': {
                    'title': 'Raw Acceleration vs Time','xlabel': 'Time [s]','ylabel': 'm/s²',
                    'plot_height': 300, # Height constraint
                    'datasets': [{'data_type': 'accel_x', 'label': 'X', 'color': 'r'},
                                 {'data_type': 'accel_y', 'label': 'Y', 'color': 'g'},
                                 {'data_type': 'accel_z', 'label': 'Z', 'color': 'b'}]
                    # enable_logging defaults to False if omitted
                }
            },
            {   'component_class': TimeSeriesPlotComponent,
                'row': 0, 'col': 1,
                'config': {
                     'title': 'Gravity vs Time','xlabel': 'Time [s]','ylabel': 'm/s²',
                     'plot_height': 300,
                     'datasets': [{'data_type': 'gravity_x', 'label': 'X', 'color': 'r'},
                                  {'data_type': 'gravity_y', 'label': 'Y', 'color': 'g'},
                                  {'data_type': 'gravity_z', 'label': 'Z', 'color': 'b'}],
                     'enable_logging': True # Log this one too
                }
            }
        ]
    },
    {
        'tab_title': 'Other Sensors',
        'layout': [
             {  'component_class': TimeSeriesPlotComponent,
                 'row': 1, 'col': 1,
                 'config': {
                     'title': 'Magnetic Field vs Time','xlabel': 'Time [s]','ylabel': 'µT',
                     'plot_height': 350, 'plot_width': 600, # Both constraints
                     'datasets': [{'data_type': 'mag_x', 'label': 'X', 'color': 'r'},
                                  {'data_type': 'mag_y', 'label': 'Y', 'color': 'g'},
                                  {'data_type': 'mag_z', 'label': 'Z', 'color': 'b'}],
                     'enable_logging': True # Log magnetometer data
                 }
             },
             # Example of placing another loggable value display
             {   'component_class': SingleValueDisplayComponent,
                 'row': 1, 'col': 0,
                 'config': {
                    'label': 'Current Mag X', 'data_type': 'mag_x',
                    'format': '{:.1f}', 'units': 'µT',
                    'enable_logging': True # Log this specific value as well
                }
            },
                {  'component_class': TimeSeriesPlotComponent,
                'row': 0, 'col': 0,
                'config': {
                    'title': 'Impedance Magnitude vs Time','xlabel': 'Time [s]','ylabel': 'Impedance Magnitude',
                    'plot_height': 350, 'plot_width': 600,
                    'datasets': [{'data_type': 'impedance_magnitude_ohm', 'label': 'Magnitude','color': 'r'},],
                    'enable_logging': True
                }
            },
                {  'component_class': TimeSeriesPlotComponent,
                'row': 0, 'col': 1,
                'config': {
                    'title': 'Impdance Phase in Degrees vs Time','xlabel': 'Time [s]','ylabel': 'Impdance Phase [°]]',
                    'plot_height': 350, 'plot_width': 600,
                    'datasets': [{'data_type': 'impedance_phase_rad', 'label': 'Phase', 'color': 'r'},],
                    'enable_logging': True
                }
            },

        ]
    },
    {
        'tab_title': 'Optical Flow',
        'layout': [
            {
                'component_class': TimeSeriesPlotComponent,
                'row': 0, 'col': 0,
                'config': {
                    'title': 'Optical Flow Δ vs Time',
                    'xlabel': 'Time [s]',
                    'ylabel': 'Pixel Δ',
                    'plot_height': 350,
                    'plot_width': 600,
                    'datasets': [
                        {'data_type': 'opt_dx',    'label': 'ΔX',    'color': 'r'},
                        {'data_type': 'opt_dy',    'label': 'ΔY',    'color': 'g'}
                    ],
                    'enable_logging': True
                }
            },
            {
                'component_class': TimeSeriesPlotComponent,
                'row': 0, 'col': 1,
                'config': {
                    'title': 'Cumulative Optical Flow vs Time',
                    'xlabel': 'Time [s]',
                    'ylabel': 'Pixels',
                    'plot_height': 350,
                    'plot_width': 600,
                    'datasets': [
                        {'data_type': 'opt_cum_x', 'label': 'Cum X', 'color': 'r'},
                        {'data_type': 'opt_cum_y', 'label': 'Cum Y', 'color': 'g'}
                    ],
                    'enable_logging': True
                }
            }
        ]
    },
    {
        'tab_title': 'ToF Sensor',
        'layout': [
            {
                'component_class': TimeSeriesPlotComponent,
                'row': 0, 'col': 0,
                'config': {
                    'title': 'Distance vs Time',
                    'xlabel': 'Time [s]',
                    'ylabel': 'Distance [mm]',
                    'plot_height': 350,
                    'plot_width': 350,
                    'datasets': [
                        {'data_type': 'tof_distance_mm', 'label': 'Distance', 'color': 'b'}
                    ],
                    'enable_logging': True
                }
            },
            {
                'component_class': TimeSeriesPlotComponent,
                'row': 0, 'col': 1,
                'config': {
                    'title': 'Darkness vs Time',
                    'xlabel': 'Time [s]',
                    'ylabel': 'Darkness [kcps/spad]',
                    'plot_height': 350,
                    'plot_width': 350,
                    'datasets': [
                        {'data_type': 'tof_brightness_kcps', 'label': 'Darkness', 'color': 'k'}
                    ],
                    'enable_logging': True
                }
            }
        ]
    },
    {
        'tab_title': 'Ankle Angles',
        'layout': [
            {   'component_class': TimeSeriesPlotComponent,
                'row': 0, 'col': 0,
                'config': {
                    'title': 'Ankle XZ vs Time',
                    'xlabel': 'Time [s]',
                    'ylabel': 'Angle [°]',
                    'plot_height': 300, 'plot_width': 450,
                    'datasets': [
                        {'data_type': 'ankle_xz', 'label': 'XZ', 'color': 'r'}
                    ],
                    'enable_logging': True
                }
            },
            {   'component_class': TimeSeriesPlotComponent,
                'row': 0, 'col': 1,
                'config': {
                    'title': 'Ankle YZ vs Time',
                    'xlabel': 'Time [s]',
                    'ylabel': 'Angle [°]',
                    'plot_height': 300, 'plot_width': 450,
                    'datasets': [
                        {'data_type': 'ankle_yz', 'label': 'YZ', 'color': 'g'}
                    ],
                    'enable_logging': True
                }
            }
        ]
    }
]


#####################################################################################################################
# End of customizable section
#####################################################################################################################
# DONT CHANGE ANYTHING BELOW THIS LINE UNLESS YOU KNOW WHAT YOU'RE DOING

# --- GUI Manager ---
class GuiManager:
    #Manages the creation, layout, and updating of GUI components across tabs.

    def __init__(self, tab_widget: QTabWidget, tab_configs: List[Dict], data_buffers_ref: Dict[str, List[Tuple[float, float]]], device_config_ref: DeviceConfig):
        self.tab_widget = tab_widget
        self.tab_configs = tab_configs
        self.data_buffers_ref = data_buffers_ref
        self.device_config_ref = device_config_ref
        self.all_components: List[BaseGuiComponent] = []
        self.active_missing_uuids: Set[str] = set()
        self.create_gui_layout()

    def create_gui_layout(self):
        for tab_index, tab_config in enumerate(self.tab_configs):
            tab_title = tab_config.get('tab_title', f'Tab {tab_index + 1}')
            component_layout_defs = tab_config.get('layout', [])
            tab_content_widget = QWidget()
            grid_layout = QGridLayout(tab_content_widget)
            if not component_layout_defs:
                empty_label = QLabel(f"No components configured for '{tab_title}'")
                empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                grid_layout.addWidget(empty_label, 0, 0)
            else:
                for comp_index, comp_def in enumerate(component_layout_defs):
                    comp_class: Type[BaseGuiComponent] = comp_def.get('component_class')
                    config = comp_def.get('config', {})
                    row = comp_def.get('row', 0)
                    col = comp_def.get('col', 0)
                    rowspan = comp_def.get('rowspan', 1)
                    colspan = comp_def.get('colspan', 1)
                    if not comp_class or not issubclass(comp_class, BaseGuiComponent):
                        logger.error(f"Invalid or missing 'component_class' in tab '{tab_title}', row {row}, col {col}. Skipping.")
                        error_widget = QLabel(f"Error:\nInvalid Component\n(Row {row}, Col {col})")
                        error_widget.setStyleSheet("QLabel { color: red; border: 1px solid red; }")
                        grid_layout.addWidget(error_widget, row, col, rowspan, colspan)
                        continue
                    try:
                        component_instance = comp_class(config, self.data_buffers_ref, self.device_config_ref)
                        component_instance.tab_index = tab_index
                        self.all_components.append(component_instance)
                        widget_to_add = component_instance.get_widget()
                        grid_layout.addWidget(widget_to_add, row, col, rowspan, colspan)
                        log_status = "LOGGING ENABLED" if component_instance.is_loggable else "logging disabled"
                        logger.debug(f"Added component {comp_class.__name__} to tab '{tab_title}' at ({row}, {col}) - {log_status}")
                    except Exception as e:
                        logger.error(f"Failed to instantiate/add component {comp_class.__name__} in tab '{tab_title}': {e}", exc_info=True)
                        error_widget = QLabel(f"Error:\n{comp_class.__name__}\nFailed to load\n(Row {row}, Col {col})")
                        error_widget.setStyleSheet("QLabel { color: red; border: 1px solid red; }")
                        grid_layout.addWidget(error_widget, row, col, rowspan, colspan)
            tab_content_widget.setLayout(grid_layout)
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setWidget(tab_content_widget)
            self.tab_widget.addTab(scroll_area, tab_title)

    def update_all_components(self, current_relative_time: float, is_flowing: bool):
        if plotting_paused or start_time is None:
            return
        for component in self.all_components:
            try:
                component.update_component(current_relative_time, is_flowing)
            except Exception as e:
                logger.error(f"Error updating component {type(component).__name__}: {e}", exc_info=True)

    def clear_all_components(self):
        logger.info("GuiManager clearing all components.")
        self.active_missing_uuids.clear()
        for component in self.all_components:
            try:
                component.clear_component()
                component.handle_missing_uuids(set())
            except Exception as e:
                logger.error(f"Error clearing component {type(component).__name__}: {e}", exc_info=True)

    def notify_missing_uuids(self, missing_uuids_set: Set[str]):
        logger.info(f"GuiManager received missing UUIDs: {missing_uuids_set if missing_uuids_set else 'None'}")
        self.active_missing_uuids = missing_uuids_set
        for component in self.all_components:
            required_types = component.get_required_data_types()
            if not required_types:
                continue
            relevant_missing_uuids_for_comp = set()
            for data_type in required_types:
                uuid = self.device_config_ref.get_uuid_for_data_type(data_type)
                if uuid and uuid in self.active_missing_uuids:
                    relevant_missing_uuids_for_comp.add(uuid)
            try:
                component.handle_missing_uuids(relevant_missing_uuids_for_comp)
            except Exception as e:
                 logger.error(f"Error notifying component {type(component).__name__} about missing UUIDs: {e}", exc_info=True)

# --- Bluetooth Protocol Handling ---
class GuiSignalEmitter(QObject):
    state_change_signal = pyqtSignal(str)
    scan_throbber_signal = pyqtSignal(str)
    connection_status_signal = pyqtSignal(str)
    show_error_signal = pyqtSignal(str, str)
    missing_uuids_signal = pyqtSignal(set)

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
    loop = asyncio.get_event_loop()
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

async def find_device(device_config_current: DeviceConfig) -> Optional[BleakClient]:
    found_event = asyncio.Event()
    target_device = None
    scan_cancelled = False
    def detection_callback(device, advertisement_data):
        nonlocal target_device, found_event
        if not found_event.is_set():
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
        service_uuids=[device_config_current.service_uuid]
    )

    logger.info(f"Starting scanner for {device_config_current.name} (Service: {device_config_current.service_uuid})...")
    gui_emitter.emit_scan_throbber("Scanning...")
    try:
        await scanner.start()
        try:
            await asyncio.wait_for(found_event.wait(), timeout=device_config_current.find_timeout)
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
        if scanner is not None: 
            try:
                await scanner.stop()
                logger.info(f"Scanner stopped for {device_config_current.name}.")
            except BleakError as e: 
                logger.warning(f"BleakError stopping scanner: {e}")
            except Exception as e:
                logger.warning(f"Error encountered during explicit scanner stop: {e}", exc_info=False)
        else:
            logger.debug("Scanner object was None in finally block, skipping stop.")

    if scan_cancelled: raise asyncio.CancelledError
    return target_device

async def connection_task():
    global client, last_received_time, state, device_config, stop_flag
    found_char_configs: List[CharacteristicConfig] = []
    while state == "scanning" and not stop_flag: 
        target_device = None
        found_char_configs = []
        current_device_config = device_config
        try:
            target_device = await find_device(current_device_config)
        except asyncio.CancelledError:
            logger.info("connection_task: Scan was cancelled.")
            break
        except Exception as e:
            logger.error(f"Error during scanning phase: {e}")
            gui_emitter.emit_show_error("Scan Error", f"Scan failed: {e}")
            await asyncio.sleep(3)
            continue
        
        if stop_flag: break 

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
             if state != "scanning" or stop_flag: logger.info("Connection attempt aborted, state changed or stop_flag set."); break
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
        
        if stop_flag: break 

        if not connection_successful:
             logger.error("Max connection attempts reached or connection aborted.")
             gui_emitter.emit_missing_uuids(set())
             if state == "scanning":
                 logger.info("Retrying scan...")
                 gui_emitter.emit_scan_throbber("Connection failed. Retrying scan...")
                 await asyncio.sleep(1)
                 continue
             else:
                 logger.info("Exiting connection task as state is no longer 'scanning'.")
                 break
        notification_errors = False
        missing_uuids = set()
        try:
            logger.info(f"Checking characteristics for service {current_device_config.service_uuid}...")
            if not client or not client.is_connected:
                logger.error("Client not available or not connected before characteristic check.")
                gui_emitter.emit_state_change("disconnecting")
                notification_errors = True
            else:
                service = client.services.get_service(current_device_config.service_uuid)
                if not service:
                    logger.error(f"Service {current_device_config.service_uuid} not found on connected device.")
                    gui_emitter.emit_show_error("Connection Error", f"Service UUID\n{current_device_config.service_uuid}\nnot found on device.")
                    gui_emitter.emit_state_change("disconnecting")
                    notification_errors = True
                else:
                    logger.info("Service found. Checking configured characteristics...")
                    found_char_configs = []
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
                    gui_emitter.emit_missing_uuids(missing_uuids)
                    if not found_char_configs:
                        logger.error("No usable (found and notifiable) characteristics from config. Disconnecting.")
                        gui_emitter.emit_show_error("Connection Error", "None of the configured characteristics\nwere found or support notifications.")
                        gui_emitter.emit_state_change("disconnecting")
                        notification_errors = True
                    else:
                        logger.info(f"Starting notifications for {len(found_char_configs)} found characteristics...")
                        notify_tasks = []
                        for char_config in found_char_configs:
                            handler_with_char = partial(notification_handler, char_config)
                            if client and client.is_connected:
                                notify_tasks.append(client.start_notify(char_config.uuid, handler_with_char))
                            else:
                                logger.error(f"Client disconnected before starting notify for {char_config.uuid}")
                                notification_errors = True; break 
                        
                        if notification_errors: 
                            gui_emitter.emit_state_change("disconnecting")
                        else:
                            results = await asyncio.gather(*notify_tasks, return_exceptions=True)
                            all_notifications_started = True
                            for i, result in enumerate(results):
                                if isinstance(result, Exception):
                                    char_uuid = found_char_configs[i].uuid
                                    logger.error(f"Failed to start notification for {char_uuid}: {result}")
                                    all_notifications_started = False; notification_errors = True
                                    missing_uuids.add(char_uuid)
                            if not all_notifications_started:
                                logger.error("Could not start all required notifications. Disconnecting.")
                                gui_emitter.emit_missing_uuids(missing_uuids)
                                gui_emitter.emit_state_change("disconnecting")
                            else:
                                logger.info("Notifications started successfully. Listening...")
                                last_received_time = time.time()
                                disconnected_event.clear()
                                while state == "connected" and not stop_flag: 
                                    try:
                                        await asyncio.wait_for(disconnected_event.wait(), timeout=0.2) 
                                        logger.info("Disconnected event received while listening.")
                                        gui_emitter.emit_state_change("disconnecting")
                                        break
                                    except asyncio.TimeoutError:
                                        current_time = time.time()
                                        if current_time - last_received_time > current_device_config.data_timeout:
                                            logger.warning(f"No data received for {current_time - last_received_time:.1f}s (timeout: {current_device_config.data_timeout}s). Assuming disconnect.")
                                            gui_emitter.emit_state_change("disconnecting")
                                            break
                                        if client and not client.is_connected:
                                            logger.warning("Bleak client reported disconnected during listening loop.")
                                            disconnected_event.set() 
                                            gui_emitter.emit_state_change("disconnecting")
                                            break
                                        continue
                                    except asyncio.CancelledError:
                                        logger.info("Listening loop cancelled.")
                                        gui_emitter.emit_state_change("disconnecting"); raise
                                    except Exception as e:
                                        logger.error(f"Error during notification listening loop: {e}")
                                        gui_emitter.emit_state_change("disconnecting"); notification_errors = True; break
        except asyncio.CancelledError:
             logger.info("Notification setup or listening task was cancelled.")
             if state == "connected": gui_emitter.emit_state_change("disconnecting")
        except Exception as e: 
             logger.error(f"Error during characteristic check or notification handling: {e}")
             gui_emitter.emit_state_change("disconnecting"); notification_errors = True
        finally:
            logger.info("Performing cleanup for connection task...")
            local_client_ref = client 
            client = None 

            if local_client_ref:
                is_conn_at_cleanup_start = False
                try: is_conn_at_cleanup_start = local_client_ref.is_connected
                except Exception as check_err: logger.warning(f"Error checking client connection status during cleanup: {check_err}")

                if is_conn_at_cleanup_start:
                    logger.info("Attempting to stop notifications and disconnect client...")
                    stop_notify_tasks = []
                    for char_config in found_char_configs: 
                        try:
                            if local_client_ref.is_connected:
                                logger.debug(f"Preparing stop_notify for {char_config.uuid}")
                                stop_notify_tasks.append(local_client_ref.stop_notify(char_config.uuid))
                            else:
                                logger.debug(f"Client disconnected before stop_notify for {char_config.uuid}, skipping.")
                                break 
                        except Exception as prep_err:
                            logger.warning(f"Error preparing stop_notify for {char_config.uuid}: {prep_err}")
                    
                    if stop_notify_tasks:
                        try:
                            results = await asyncio.gather(*stop_notify_tasks, return_exceptions=True)
                            logger.info(f"Notifications stop attempts completed for {len(stop_notify_tasks)} characteristics.")
                            for i, result in enumerate(results):
                                if isinstance(result, Exception):
                                    logger.warning(f"Error stopping notification {found_char_configs[i].uuid if i < len(found_char_configs) else 'unknown_uuid'}: {result}")
                        except Exception as gather_err:
                            logger.error(f"Error during asyncio.gather for stop_notify: {gather_err}")

                    
                    try:
                        if local_client_ref.is_connected:
                            await asyncio.wait_for(local_client_ref.disconnect(), timeout=5.0)
                            logger.info("Client disconnected.")
                        else:
                            logger.info("Client was already disconnected before explicit disconnect call in cleanup.")
                    except asyncio.TimeoutError:
                        logger.error("Timeout during client disconnect.")
                    except Exception as e:
                        logger.error(f"Error during client disconnect: {e}")
                else:
                    logger.info("Client object existed but was not connected at start of cleanup.")
            else:
                logger.info("No active client object to cleanup at start of finally block.")

            if state in ["connected", "disconnecting"] or not connection_successful:
                 gui_emitter.emit_missing_uuids(set())
            if state in ["connected", "disconnecting"] and not stop_flag : 
                 logger.info("Signalling state change to idle after cleanup.")
                 gui_emitter.emit_state_change("idle")
            
            disconnected_event.clear()
            found_char_configs = []
        if state != "scanning" or stop_flag: 
            logger.info(f"Exiting connection_task as state is '{state}' or stop_flag is set.")
            break
    logger.info("Connection task loop finished.")


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
    # Signal to request an update
    request_plot_update_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modular BLE Data GUI")
        self.setGeometry(100, 100, 1400, 950)

        # --- Capture State ---
        self._shutting_down = False 
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
        self.clear_button = QPushButton("Clear GUI"); self.clear_button.clicked.connect(self.clear_gui_action); self.button_layout.addWidget(self.clear_button)
        self.status_label = QLabel("On Standby"); self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter); self.status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred); self.button_layout.addWidget(self.status_label)
        self.main_layout.addWidget(self.button_bar)

        # --- Tab Area (Managed by GuiManager) ---
        self.tab_widget = QTabWidget()
        self.tab_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.gui_manager = GuiManager(self.tab_widget, tab_configs, data_buffers, device_config)
        self.main_layout.addWidget(self.tab_widget)

        # --- Bottom Control Bar ---
        self.bottom_bar = QWidget()
        self.bottom_layout = QHBoxLayout(self.bottom_bar)
        self.bottom_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.flowing_mode_check = QCheckBox("Flowing Mode"); self.flowing_mode_check.setChecked(True); self.bottom_layout.addWidget(self.flowing_mode_check)
        self.interval_label = QLabel("Interval (s):"); self.bottom_layout.addWidget(self.interval_label)
        self.interval_entry = QLineEdit(str(flowing_interval)); self.interval_entry.setFixedWidth(50); self.bottom_layout.addWidget(self.interval_entry)
        self.apply_interval_button = QPushButton("Apply Interval"); self.apply_interval_button.clicked.connect(self.apply_interval); self.bottom_layout.addWidget(self.apply_interval_button)
        self.data_log_check = QCheckBox("Log Raw Data to Console"); self.data_log_check.setChecked(False); self.data_log_check.stateChanged.connect(self.toggle_data_log); self.bottom_layout.addWidget(self.data_log_check)
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
        self.plot_update_timer.setInterval(50)
        self.plot_update_timer.timeout.connect(self.trigger_gui_update)
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
        gui_emitter.missing_uuids_signal.connect(self.gui_manager.notify_missing_uuids)
        self.request_plot_update_signal.connect(self._update_gui_now)
        self.handle_state_change("idle")

    def append_log_message(self, message):
        self.log_text_box.append(message)

    def trigger_gui_update(self):
        self.request_plot_update_signal.emit()

    def _update_gui_now(self):
        if start_time is not None:
            current_relative = (datetime.datetime.now() - start_time).total_seconds()
            is_flowing = self.flowing_mode_check.isChecked()
            self.gui_manager.update_all_components(current_relative, is_flowing)

    def animate_scan_throbber(self):
        if state == "scanning":
            text = "Scanning... " + self.throbber_chars[self.throbber_index]
            self.status_label.setText(text)
            self.throbber_index = (self.throbber_index + 1) % len(self.throbber_chars)
        else:
            self.scan_throbber_timer.stop()

    def update_target_device(self, selected_name: str):
        global device_config
        if device_config.name != selected_name:
            logger.info(f"Target device changed via GUI: {selected_name}")
            device_config.update_name(selected_name)
            logger.info("Device config name updated.")

    def handle_state_change(self, new_state: str):
        global state, plotting_paused, start_time
        logger.info(f"GUI received state change: {new_state}")
        previous_state = state
        state = new_state
        if new_state != "scanning" and self.scan_throbber_timer.isActive():
            self.scan_throbber_timer.stop()
        is_idle = (new_state == "idle")
        self.device_combo.setEnabled(is_idle)
        self.scan_button.setEnabled(True)
        if new_state == "idle":
            self.scan_button.setText("Start Scan")
            self.led_indicator.set_color("red"); self.status_label.setText("On Standby")
            self.pause_resume_button.setEnabled(False); self.pause_resume_button.setText("Pause Plotting")
            self.capture_button.setEnabled(False); self.capture_button.setText("Start Capture")
            plotting_paused = True
            if previous_state != "disconnecting": 
                logger.info("State changed to idle. Automatically clearing GUI and state.")
                self.clear_gui_action(confirm=False)
            if self.is_capturing:
                 logger.warning("Capture was active when state became idle (likely disconnect). Files were NOT generated automatically by clear.")
        elif new_state == "scanning":
            self.scan_button.setText("Stop Scan")
            self.led_indicator.set_color("orange"); self.throbber_index = 0
            if not self.scan_throbber_timer.isActive(): self.scan_throbber_timer.start()
            self.pause_resume_button.setEnabled(False)
            self.capture_button.setEnabled(False)
        elif new_state == "connected":
            self.scan_button.setText("Disconnect")
            self.led_indicator.set_color("lightgreen"); self.status_label.setText(f"Connected to: {device_config.name}")
            if not self.is_capturing:
                self.pause_resume_button.setEnabled(True)
            self.pause_resume_button.setText("Pause Plotting")
            self.capture_button.setEnabled(True)
            plotting_paused = False
        elif new_state == "disconnecting":
            self.scan_button.setText("Disconnecting..."); self.scan_button.setEnabled(False)
            self.led_indicator.set_color("red"); self.status_label.setText("Status: Disconnecting...")
            self.pause_resume_button.setEnabled(False); self.pause_resume_button.setText("Pause Plotting")
            self.capture_button.setEnabled(False)
            plotting_paused = True

    def update_scan_status(self, text: str):
         if state == "scanning": self.status_label.setText(text)
    def update_connection_status(self, text: str):
         if state != "connected" and state != "idle": self.status_label.setText(text)
    def show_message_box(self, title: str, message: str):
        QMessageBox.warning(self, title, message)

    @qasync.asyncSlot()
    async def toggle_scan(self):
        global current_task, state, data_buffers, start_time
        if self._shutting_down: return

        if state == "idle":
            event_loop = asyncio.get_event_loop()
            if event_loop and event_loop.is_running():
                logger.info("Clearing state before starting scan...")
                self.clear_gui_action(confirm=False)
                self.handle_state_change("scanning")
                current_task = asyncio.create_task(connection_task())
            else:
                logger.error("Asyncio loop not running!")
                self.show_message_box("Error", "Asyncio loop is not running.")
        elif state == "scanning":
            if current_task and not current_task.done():
                logger.info("Requesting scan cancellation...")
                self.scan_button.setEnabled(False) 
                await self.cancel_and_wait_task(current_task)
                self.scan_button.setEnabled(True)
                if state == "scanning": 
                    self.handle_state_change("idle")
            else:
                logger.warning("Stop scan requested, but no task was running/done.")
                self.handle_state_change("idle")
            current_task = None
        elif state == "connected":
            self.handle_state_change("disconnecting") 
            if client and client.is_connected:
                logger.info("Requesting disconnection via disconnected_event...")
                disconnected_event.set() 
            elif current_task and not current_task.done():
                logger.info("Requesting disconnect via task cancellation...")
                await self.cancel_and_wait_task(current_task) 
            else:
                logger.warning("Disconnect requested but no active connection/task found.")
                self.handle_state_change("idle") 

    def toggle_pause_resume(self):
        global plotting_paused
        if not self.pause_resume_button.isEnabled():
            logger.warning("Pause/Resume toggled while button disabled. Ignoring.")
            return
        plotting_paused = not plotting_paused
        self.pause_resume_button.setText("Resume Plotting" if plotting_paused else "Pause Plotting")
        logger.info(f"Plotting {'paused' if plotting_paused else 'resumed'}")
        if not plotting_paused:
            self.trigger_gui_update()

    def toggle_capture(self):
        global start_time
        if not self.is_capturing:
            if state != "connected" or start_time is None:
                self.show_message_box("Capture Error", "Must be connected and receiving data.")
                return
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
                return
            self.is_capturing = True
            self.capture_button.setText("Stop Capture && Export")
            self.capture_t0_absolute = datetime.datetime.now()
            self.capture_start_relative_time = (self.capture_t0_absolute - start_time).total_seconds()
            self.pause_resume_button.setEnabled(False)
            logger.info("Pause/Resume plotting disabled during capture.")
            logger.info(f"Capture started. Abs t0: {self.capture_t0_absolute}, Rel t0: {self.capture_start_relative_time:.3f}s.")
        else:
            self.stop_and_generate_files()

    def stop_and_generate_files(self):
        if not self.is_capturing or start_time is None:
            logger.warning("stop_and_generate called but capture inactive or start_time missing.")
            if state == "connected": self.pause_resume_button.setEnabled(True)
            else: self.pause_resume_button.setEnabled(False)
            self.is_capturing = False
            self.capture_button.setText("Start Capture")
            self.capture_button.setEnabled(state == "connected")
            return
        logger.info("Stopping capture, generating PDF & CSV.")
        output_dir = self.capture_output_base_dir
        start_rel_time = self.capture_start_relative_time
        capture_end_relative_time = (datetime.datetime.now() - start_time).total_seconds()
        self.is_capturing = False
        self.capture_button.setText("Start Capture")
        self.capture_button.setEnabled(state == "connected")
        if state == "connected":
            self.pause_resume_button.setEnabled(True)
            logger.info("Pause/Resume plotting re-enabled after capture (still connected).")
        else:
            self.pause_resume_button.setEnabled(False)
            logger.info("Pause/Resume plotting remains disabled after capture (not connected).")
        if output_dir and start_rel_time is not None:
            if not data_buffers:
                 logger.warning("No data captured during the active period. Skipping PDF/CSV generation.")
                 self.capture_output_base_dir = None; self.capture_start_relative_time = None
                 self.capture_t0_absolute = None; self.capture_timestamp = None
                 return
            pdf_subdir = os.path.join(output_dir, "pdf_plots")
            csv_subdir = os.path.join(output_dir, "csv_files")
            try: os.makedirs(pdf_subdir, exist_ok=True); os.makedirs(csv_subdir, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create PDF/CSV subdirs: {e}")
                self.show_message_box("File Gen Error", f"Could not create output subdirs:\n{e}")
                self.capture_output_base_dir = None; self.capture_start_relative_time = None
                self.capture_t0_absolute = None; self.capture_timestamp = None
                return
            gen_errors = []
            try:
                self.generate_pdf_plots_from_buffer(pdf_subdir, start_rel_time)
            except Exception as e: logger.error(f"PDF generation failed: {e}", exc_info=True); gen_errors.append(f"PDF: {e}")
            try:
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
        self.capture_output_base_dir = None
        self.capture_start_relative_time = None
        self.capture_t0_absolute = None
        self.capture_timestamp = None

    def generate_pdf_plots_from_buffer(self, pdf_dir: str, capture_start_relative_time: float):
        global data_buffers
        logger.info(f"Generating PDF plots (t=0 at capture start, t_offset={capture_start_relative_time:.3f}s). Dir: {pdf_dir}")
        if not data_buffers: logger.warning("Data buffer empty, skipping PDF generation."); return
        try:
            plt.style.use('science')
            plt.rcParams.update({'text.usetex': False, 'figure.figsize': [5.5, 3.5], 'legend.fontsize': 9,
                                 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.titlesize': 11})
        except Exception as style_err:
            logger.warning(f"Could not apply 'science' style: {style_err}. Using default.")
            plt.rcParams.update({'figure.figsize': [6.0, 4.0]})
        gen_success = False
        for component in self.gui_manager.all_components:
            if not isinstance(component, TimeSeriesPlotComponent):
                continue
            plot_config = component.config
            plot_title = plot_config.get('title', 'UntitledPlot')
            datasets = plot_config.get('datasets', [])
            if not datasets: continue
            required_uuids_for_plot = set()
            required_types = component.get_required_data_types()
            for dtype in required_types:
                uuid = self.gui_manager.device_config_ref.get_uuid_for_data_type(dtype)
                if uuid: required_uuids_for_plot.add(uuid)
            missing_uuids_for_this_plot = required_uuids_for_plot.intersection(self.gui_manager.active_missing_uuids)
            if missing_uuids_for_this_plot:
                logger.warning(f"Skipping PDF for plot '{plot_title}' as it depends on missing UUID(s): {missing_uuids_for_this_plot}")
                continue
            fig, ax = plt.subplots()
            ax.set_title(plot_config.get('title', 'Plot'));
            ax.set_xlabel(plot_config.get('xlabel', 'Time [s]'));
            ax.set_ylabel(plot_config.get('ylabel', 'Value'))
            plot_created = False
            for dataset in datasets:
                data_type = dataset['data_type']
                if data_type in data_buffers and data_buffers[data_type]:
                    full_data = data_buffers[data_type]
                    plot_data = [(item[0] - capture_start_relative_time, item[1])
                                    for item in full_data if item[0] >= capture_start_relative_time]
                    if plot_data:
                        try:
                            times_rel_capture = [p[0] for p in plot_data]
                            values = [p[1] for p in plot_data]
                            ax.plot(times_rel_capture, values, label=dataset.get('label', data_type), color=dataset.get('color', 'k'), linewidth=1.2)
                            plot_created = True
                        except Exception as plot_err: logger.error(f"Error plotting {data_type} for PDF '{plot_title}': {plot_err}")
            if plot_created:
                ax.legend(); ax.grid(True, linestyle='--', alpha=0.6); fig.tight_layout(pad=0.5)
                safe_suffix = component.get_log_filename_suffix()
                prefix = f"{self.capture_timestamp}_" if self.capture_timestamp else ""
                pdf_filename = f"{prefix}{safe_suffix}.pdf"
                pdf_filepath = os.path.join(pdf_dir, pdf_filename)
                try: fig.savefig(pdf_filepath, bbox_inches='tight'); logger.info(f"Saved PDF: {pdf_filename}"); gen_success = True
                except Exception as save_err: logger.error(f"Error saving PDF {pdf_filename}: {save_err}"); raise RuntimeError(f"Save PDF failed: {save_err}") from save_err
            else: logger.info(f"Skipping PDF '{plot_title}' (no data in capture window).")
            plt.close(fig)
        if gen_success: logger.info(f"PDF generation finished. Dir: {pdf_dir}")
        else: logger.warning("PDF done, but no plots saved (no data / missing UUIDs / no plot components?).")

    def generate_csv_files_from_buffer(self, csv_dir: str, filter_start_rel_time: float, filter_end_rel_time: float, time_offset: float):
        global data_buffers
        logger.info(f"Generating CSVs (data {filter_start_rel_time:.3f}s-{filter_end_rel_time:.3f}s rel session, t=0 at capture start offset={time_offset:.3f}s). Dir: {csv_dir}")
        if not data_buffers: logger.warning("Data buffer empty, skipping CSV generation."); return
        def get_series(dt, start, end):
             uuid = self.gui_manager.device_config_ref.get_uuid_for_data_type(dt)
             if uuid and uuid in self.gui_manager.active_missing_uuids:
                 logger.debug(f"Excluding series '{dt}' from CSV (UUID {uuid} was missing).")
                 return None
             if dt in data_buffers and data_buffers[dt]:
                 filt = [item for item in data_buffers[dt] if start <= item[0] <= end]
                 if filt:
                     try:
                         return pd.Series([i[1] for i in filt], index=pd.Index([i[0] for i in filt], name='TimeRelSession'), name=dt)
                     except Exception as e:
                         logger.error(f"Error creating Pandas Series for {dt}: {e}")
             return None
        indiv_gen = False
        for component in self.gui_manager.all_components:
            if not component.is_loggable:
                continue
            loggable_data_types = component.get_loggable_data_types()
            log_filename_suffix = component.get_log_filename_suffix()
            if not loggable_data_types or not log_filename_suffix:
                logger.warning(f"Skipping individual CSV for component {type(component).__name__} (ID: {id(component)}): No loggable types or filename suffix.")
                continue
            logger.info(f"Processing Individual CSV for component: {log_filename_suffix}")
            series_list = [s for dt in sorted(list(loggable_data_types)) if (s := get_series(dt, filter_start_rel_time, filter_end_rel_time)) is not None]
            if not series_list:
                logger.warning(f"Skipping Individual CSV '{log_filename_suffix}': No valid series data found in window or UUIDs missing.")
                continue
            try:
                component_df = pd.concat(series_list, axis=1, join='outer').sort_index()
                component_df.insert(0, 'Time (s)', component_df.index - time_offset)
                prefix = f"{self.capture_timestamp}_" if self.capture_timestamp else ""
                csv_fname = f"{prefix}{log_filename_suffix}.csv"
                csv_fpath = os.path.join(csv_dir, csv_fname)
                component_df.to_csv(csv_fpath, index=False, float_format='%.6f')
                logger.info(f"Saved Individual CSV: {csv_fname}"); indiv_gen = True
            except Exception as e:
                logger.error(f"Error generating Individual CSV '{log_filename_suffix}': {e}", exc_info=True)
                raise RuntimeError(f"Individual CSV generation failed: {e}") from e
        master_gen = False
        for tab_index, tab_config in enumerate(self.gui_manager.tab_configs):
            tab_title = tab_config.get('tab_title', f"Tab_{tab_index}")
            logger.info(f"Processing Master CSV for tab: '{tab_title}'")
            tab_loggable_types = set()
            components_in_tab = [comp for comp in self.gui_manager.all_components if comp.tab_index == tab_index and comp.is_loggable]
            if not components_in_tab:
                 logger.warning(f"Skipping Master CSV '{tab_title}': No loggable components found in this tab."); continue
            for component in components_in_tab:
                tab_loggable_types.update(component.get_loggable_data_types())
            if not tab_loggable_types:
                logger.warning(f"Skipping Master CSV '{tab_title}': No loggable data types found among loggable components in this tab."); continue
            series_list = [s for dt in sorted(list(tab_loggable_types)) if (s := get_series(dt, filter_start_rel_time, filter_end_rel_time)) is not None]
            if not series_list:
                logger.warning(f"Skipping Master CSV '{tab_title}': No valid series data found for loggable types in window or UUIDs missing."); continue
            try:
                master_df = pd.concat(series_list, axis=1, join='outer').sort_index()
                master_df.insert(0, 'Master Time (s)', master_df.index - time_offset)
                safe_t_title = re.sub(r'[^\w\-]+', '_', tab_title).strip('_')
                prefix = f"{self.capture_timestamp}_" if self.capture_timestamp else ""
                csv_fname = f"{prefix}master_tab_{safe_t_title}.csv"
                csv_fpath = os.path.join(csv_dir, csv_fname)
                master_df.to_csv(csv_fpath, index=False, float_format='%.6f')
                logger.info(f"Saved Master CSV: {csv_fname}"); master_gen = True
            except Exception as e:
                logger.error(f"Error generating Master CSV '{tab_title}': {e}", exc_info=True)
                raise RuntimeError(f"Master CSV generation failed: {e}") from e
        if master_gen or indiv_gen: logger.info(f"CSV generation finished. Dir: {csv_dir}")
        else: logger.warning("CSV generation done, but no files were saved (no loggable components / no valid data?).")

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
        else: do_clear = True
        if do_clear:
            logger.info("Confirmed clearing GUI, data buffers, resetting start time, and UUID status.")
            if self.is_capturing:
                 logger.warning("Capture active during clear. Stopping capture WITHOUT generating files.")
                 self.is_capturing = False
                 self.capture_button.setText("Start Capture")
                 self.capture_button.setEnabled(state == "connected")
                 if state == "connected": self.pause_resume_button.setEnabled(True)
                 else: self.pause_resume_button.setEnabled(False)
                 self.capture_output_base_dir = None; self.capture_start_relative_time = None
                 self.capture_t0_absolute = None; self.capture_timestamp = None
            data_buffers.clear()
            start_time = None
            self.gui_manager.clear_all_components()

    def apply_interval(self):
        global flowing_interval
        try:
            new_interval = float(self.interval_entry.text())
            if new_interval > 0:
                flowing_interval = new_interval
                logger.info(f"Flowing interval updated to {new_interval}s")
                if self.flowing_mode_check.isChecked():
                    self._update_gui_now()
            else: self.show_message_box("Invalid Input", "Interval must be positive.")
        except ValueError: self.show_message_box("Invalid Input", "Please enter a valid number for the interval.")

    def toggle_data_log(self, check_state_value):
        is_checked = (check_state_value == Qt.CheckState.Checked.value)
        if is_checked:
            data_console_handler.setLevel(logging.INFO)
            logger.info("Raw data logging (INFO level) to console enabled.")
        else:
            data_console_handler.setLevel(logging.WARNING)
            logger.info("Raw data logging (INFO level) to console disabled.")
    

    # --- Close Event Handling ---

    def closeEvent(self, event):
        global stop_flag
        if self._shutting_down:
            event.accept()
            return

        logger.info("Close event triggered. Initiating asynchronous shutdown.")
        self._shutting_down = True
        stop_flag = True 
        event.ignore()  
        asyncio.create_task(self.async_shutdown_operations())

    async def async_shutdown_operations(self):
        global current_task, client
        logger.info("Async shutdown: Starting...")
        if current_task and not current_task.done():
            logger.info("Async shutdown: Requesting cancellation of active BLE task...")
            if not current_task.cancelled():
                await self.cancel_and_wait_task(current_task)
                logger.info("Async shutdown: BLE task cancellation completed.")
            else:
                logger.info("Async shutdown: BLE task was already cancelled.")
        else:
            logger.info("Async shutdown: No active BLE task or task already done.")
        
        current_client_ref = client 
        if current_client_ref and current_client_ref.is_connected:
            logger.info("Async shutdown: Attempting to disconnect client explicitly...")
            try:
                await current_client_ref.disconnect()
                logger.info("Async shutdown: Client disconnected successfully.")
            except Exception as e:
                logger.error(f"Async shutdown: Error during explicit client disconnect: {e}")
        
        logger.info("Async shutdown: Performing GUI cleanup...")
        self.plot_update_timer.stop()
        self.scan_throbber_timer.stop()
        if self.log_handler:
            logger.info("Async shutdown: Removing GUI log handler...")
            root_logger = logging.getLogger()
            if self.log_handler in root_logger.handlers:
                root_logger.removeHandler(self.log_handler)
            self.log_handler.close() 
            self.log_handler = None 
            logger.info("Async shutdown: GUI log handler removed and closed.")
        
        logger.info("Async shutdown: Clearing GUI components...")
        try:
            self.clear_gui_action(confirm=False)
        except Exception as e:
            logger.error(f"Async shutdown: Error clearing GUI: {e}", exc_info=True)

        if self.is_capturing:
            logger.warning("Async shutdown: Capture still active. Generating files.")
            try:
                self.stop_and_generate_files()
            except Exception as e:
                logger.error(f"Async shutdown: Error generating files on close: {e}", exc_info=True)
            self.is_capturing = False

        logger.info("Async shutdown: All cleanup done. Quitting application.")
        QApplication.instance().quit()

    async def cancel_and_wait_task(self, task_to_cancel):
        if task_to_cancel and not task_to_cancel.done():
            if not task_to_cancel.cancelled(): 
                 task_to_cancel.cancel()
            try:
                await task_to_cancel
            except asyncio.CancelledError:
                logger.info("Async task successfully cancelled and awaited.")
            except Exception as e: 
                logger.error(f"Exception while awaiting cancelled task (e.g. BleakError during disconnect): {e}", exc_info=False)


# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    qasync_loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(qasync_loop)
    main_window = MainWindow()
    main_window.show()
    exit_code = 0 
    try:
        logger.info("Starting qasync event loop (integrates Qt and asyncio)...")
        with qasync_loop:
            qasync_loop.run_forever()
        exit_code = 0 
        logger.info("qasync event loop finished.")
    except KeyboardInterrupt:
        logger.info("Application interrupted by user (KeyboardInterrupt).")
        if not main_window._shutting_down:
            stop_flag = True 
            if qasync_loop.is_running():
                 asyncio.run_coroutine_threadsafe(main_window.async_shutdown_operations(), qasync_loop)
            else: 
                logger.warning("KeyboardInterrupt after loop stopped. May not clean up fully.")

    except SystemExit: 
        logger.info("SystemExit caught, application quitting.")
        exit_code = 0 
    except Exception as e:
        logger.critical(f"Unhandled exception in main execution: {e}", exc_info=True)
        exit_code = 1 
    finally:
        stop_flag = True
        logger.info("Performing final application cleanup...")
        try:
            plt.style.use('default')
            logger.debug("Reset matplotlib style.")
        except Exception as e:
            logger.warning(f"Could not reset matplotlib style: {e}")
        
        logging.shutdown()
        logger.info(f"Application exiting with code {exit_code}.")
        sys.exit(exit_code)

# <<< END OF FILE >>>