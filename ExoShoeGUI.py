##################################################################################
#
#                   *** BLE Sensor Data Acquisition GUI***
#
##################################################################################
# Author:    Arvin Parvizinia
# Date:      2025-05-16
# Version:   1.0 
#
# Description:
#   Provides a modular GUI for connecting to Bluetooth Low Energy (BLE)
#   devices for sensor data acquisition. Features:
#     - Real-time plotting: time-series, heat/pressuremaps, 3D IMU, Impedance
#     - Data logging to CSV
#     - PDF export of plots
#     - Customizable device profiles, data handlers, GUI layouts
################################################################################
# Dependencies: (Top level only, see requirements.txt for full list)
#   pip install qasync bleak pandas numpy matplotlib scienceplots \
#               PyQt6 pyqtgraph superqt numpy-stl PyOpenGL
################################################################################
# License (MIT):
#   Copyright (c) 2025 Arvin Parvizinia
#
# Permission is granted to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, subject to the following:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM THE SOFTWARE OR
# ITS USE.
################################################################################

import asyncio
import qasync
from bleak import BleakScanner, BleakClient, BleakError
import logging
from typing import Optional, Callable, Dict, List, Tuple, Set, Any, Type
import time
from functools import partial
from collections import deque # for sensor fusion and potential future optimization if lists grow huge
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
from pathlib import Path # For easier path manipulation in PDF export


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
    QComboBox, QSlider, QDialog, QTreeWidget, QTreeWidgetItem,
    QSplitter, QTextBrowser, QFileDialog
)
from PyQt6.QtGui import (
    QColor, QPainter, QBrush, QPen, QPixmap, QImage, QPolygonF,
    QIntValidator, QDoubleValidator, QSurfaceFormat, QQuaternion,
    QVector3D, QImage, QPainter, QFont, QFontInfo )
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QMetaObject, QThread, QPointF


# --- PyQtGraph Import ---
import pyqtgraph as pg
import pyqtgraph.opengl as gl


# --- SuperQT Import for RangeSlider ---
from superqt.sliders import QRangeSlider # Needed for heatmap pressure range

# for Impedance Plotter snapshot
from concurrent.futures import ThreadPoolExecutor

from help_window import HelpWindow 

# Apply PyQtGraph global options for background/foreground
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOptions(antialias=True) # Ensure anti-aliasing is enabled

# subfolder named 'assets' in the same directory as this script.
SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR / "assets"

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

from abc import ABC, abstractmethod

class DataSource(ABC):
    @abstractmethod
    async def start(self):
        pass

    @abstractmethod
    async def stop(self):
        pass

class BleDataSource(DataSource):
    def __init__(self, device_config_ref: 'DeviceConfig', gui_emitter_ref: 'GuiSignalEmitter'):
        self._device_config = device_config_ref
        self._gui_emitter = gui_emitter_ref
        self._client: Optional[BleakClient] = None
        self._stop_requested = False
        self._active_char_configs: List[CharacteristicConfig] = []


    async def start(self):
        global stop_flag, disconnected_event, state, last_received_time
        
        self._stop_requested = False
        original_stop_flag_val = stop_flag
        stop_flag = False 

        target_device = None
        
        try:
            target_device = await find_device(self._device_config)
        except asyncio.CancelledError:
            logger.info("BleDataSource: Scan was cancelled by stop_requested.")
            self._gui_emitter.emit_state_change("idle")
            stop_flag = original_stop_flag_val
            return
        except Exception as e:
            logger.error(f"BleDataSource: Error during find_device: {e}")
            self._gui_emitter.emit_show_error("Scan Error", f"Scan failed: {e}")
            self._gui_emitter.emit_state_change("idle")
            stop_flag = original_stop_flag_val
            return

        if self._stop_requested:
            logger.info("BleDataSource: Stop requested after find_device.")
            self._gui_emitter.emit_state_change("idle")
            stop_flag = original_stop_flag_val
            return
        
        if not target_device:
            logger.info(f"BleDataSource: Device '{self._device_config.name}' not found.")
            self._gui_emitter.emit_scan_throbber(f"Device '{self._device_config.name}' not found.")
            self._gui_emitter.emit_state_change("idle")
            stop_flag = original_stop_flag_val
            return

        self._gui_emitter.emit_connection_status(f"Found {self._device_config.name}. Connecting...")
        
        connection_successful = False
        for attempt in range(3):
            if self._stop_requested:
                logger.info("BleDataSource: Stop requested during connection attempts.")
                break
            try:
                logger.info(f"BleDataSource: Connecting (attempt {attempt + 1})...")
                self._client = BleakClient(target_device, disconnected_callback=disconnected_callback)
                await self._client.connect(timeout=10.0)
                logger.info("BleDataSource: Connected successfully.")
                connection_successful = True
                self._gui_emitter.emit_state_change("connected")
                break
            except Exception as e:
                logger.error(f"BleDataSource: Connection attempt {attempt + 1} failed: {e}")
                if self._client:
                    try: await self._client.disconnect()
                    except Exception as disconnect_err: logger.warning(f"BleDataSource: Error during disconnect after failed connection: {disconnect_err}")
                self._client = None
                if attempt < 2 and not self._stop_requested: await asyncio.sleep(2)
        
        if self._stop_requested and not connection_successful:
             logger.info("BleDataSource: Stop requested and connection failed or aborted.")
             if self._client:
                 try: await self._client.disconnect()
                 except Exception as e: logger.warning(f"BleDataSource: Error disconnecting client after stop during connect: {e}")
                 self._client = None
             self._gui_emitter.emit_state_change("idle")
             stop_flag = original_stop_flag_val
             return

        if not connection_successful:
            logger.error("BleDataSource: Max connection attempts reached or connection aborted.")
            self._gui_emitter.emit_missing_uuids(set())
            self._gui_emitter.emit_scan_throbber("Connection failed.")
            self._gui_emitter.emit_state_change("idle")
            stop_flag = original_stop_flag_val
            return

        missing_uuids = set()
        self._active_char_configs = [] 
        try:
            logger.info(f"BleDataSource: Checking characteristics for service {self._device_config.service_uuid}...")
            if not self._client or not self._client.is_connected:
                raise Exception("Client not available or not connected before characteristic check.")

            service = self._client.services.get_service(self._device_config.service_uuid)
            if not service:
                logger.error(f"BleDataSource: Service {self._device_config.service_uuid} not found.")
                self._gui_emitter.emit_show_error("Connection Error", f"Service UUID\n{self._device_config.service_uuid}\nnot found on device.")
                raise Exception("Service not found")

            logger.info("BleDataSource: Service found. Checking configured characteristics...")
            for char_config_iter in self._device_config.characteristics:
                bleak_char = service.get_characteristic(char_config_iter.uuid)
                if bleak_char:
                    if "notify" in bleak_char.properties or "indicate" in bleak_char.properties:
                        self._active_char_configs.append(char_config_iter)
                    else:
                        missing_uuids.add(char_config_iter.uuid)
                else:
                    missing_uuids.add(char_config_iter.uuid)
            
            self._gui_emitter.emit_missing_uuids(missing_uuids)

            if not self._active_char_configs:
                logger.error("BleDataSource: No usable characteristics found. Disconnecting.")
                self._gui_emitter.emit_show_error("Connection Error", "None of the configured characteristics\nwere found or support notifications.")
                raise Exception("No usable characteristics")

            logger.info(f"BleDataSource: Starting notifications for {len(self._active_char_configs)} characteristics...")
            notify_tasks = []
            for char_conf in self._active_char_configs:
                handler_with_char = partial(notification_handler, char_conf)
                if self._client and self._client.is_connected:
                    notify_tasks.append(self._client.start_notify(char_conf.uuid, handler_with_char))
                else:
                    raise Exception(f"Client disconnected before starting notify for {char_conf.uuid}")
            
            results = await asyncio.gather(*notify_tasks, return_exceptions=True)
            all_notifications_started = True
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    char_uuid_err = self._active_char_configs[i].uuid
                    logger.error(f"BleDataSource: Failed to start notification for {char_uuid_err}: {result}")
                    all_notifications_started = False
                    missing_uuids.add(char_uuid_err)
            
            if not all_notifications_started:
                self._gui_emitter.emit_missing_uuids(missing_uuids)
                raise Exception("Could not start all required notifications")

            logger.info("BleDataSource: Notifications started. Listening...")
            last_received_time = time.time()
            disconnected_event.clear()
            
            while not self._stop_requested and self._client and self._client.is_connected:
                try:
                    await asyncio.wait_for(disconnected_event.wait(), timeout=0.2)
                    logger.info("BleDataSource: Disconnected event received while listening.")
                    break 
                except asyncio.TimeoutError:
                    current_time_loop = time.time()
                    if current_time_loop - last_received_time > self._device_config.data_timeout:
                        logger.warning(f"BleDataSource: No data timeout ({self._device_config.data_timeout}s). Assuming disconnect.")
                        break
                    if not (self._client and self._client.is_connected):
                        logger.warning("BleDataSource: Client reported disconnected during listening loop.")
                        break
                    continue
                except asyncio.CancelledError:
                    logger.info("BleDataSource: Listening loop cancelled by stop_requested.")
                    break 
                except Exception as e:
                    logger.error(f"BleDataSource: Error during notification listening: {e}")
                    break
            
        except asyncio.CancelledError:
            logger.info("BleDataSource: Start operation cancelled by stop_requested.")
        except Exception as e:
            logger.error(f"BleDataSource: Error during setup/listen: {e}")
            if state == "connected": self._gui_emitter.emit_state_change("disconnecting")
        finally:
            logger.info("BleDataSource: Start method entering finally block for cleanup.")
            local_client_ref = self._client  # Capture the client instance used by this start() call
            self._client = None  # Indicate this BleDataSource instance no longer manages this client internally

            active_chars_to_clean = list(self._active_char_configs) # Make a copy for safe iteration if needed
            self._active_char_configs = [] # Clear instance's list early

            if local_client_ref:
                client_was_connected_at_cleanup_start = False
                try:
                    if hasattr(local_client_ref, 'is_connected') and callable(local_client_ref.is_connected):
                        client_was_connected_at_cleanup_start = local_client_ref.is_connected
                    else:
                        logger.warning("BleDataSource: local_client_ref is not a valid BleakClient object in finally.")
                except BleakError as be: 
                    logger.warning(f"BleDataSource: BleakError when checking local_client_ref.is_connected in finally: {be}")
                except Exception as e: 
                    logger.warning(f"BleDataSource: Unexpected error checking local_client_ref.is_connected in finally: {e}")

                if client_was_connected_at_cleanup_start:
                    logger.info("BleDataSource: Client was connected at start of finally. Stopping notifications...")
                    stop_notify_tasks_final = []
                    for char_conf in active_chars_to_clean:
                        try:
                            if local_client_ref.is_connected: # Check before each attempt
                                stop_notify_tasks_final.append(local_client_ref.stop_notify(char_conf.uuid))
                            else:
                                logger.warning(f"BleDataSource: Client disconnected before stop_notify for {char_conf.uuid} could be added to tasks.")
                                break 
                        except Exception as e:
                            logger.warning(f"BleDataSource: Error preparing stop_notify for {char_conf.uuid} in finally: {e}")
                    
                    if stop_notify_tasks_final:
                        logger.info(f"BleDataSource: Attempting to stop {len(stop_notify_tasks_final)} notifications...")
                        stop_results = await asyncio.gather(*stop_notify_tasks_final, return_exceptions=True)
                        for i, res in enumerate(stop_results):
                            if isinstance(res, Exception):
                                char_uuid_err = active_chars_to_clean[i].uuid if i < len(active_chars_to_clean) else "unknown_uuid"
                                logger.warning(f"BleDataSource: Error during stop_notify for {char_uuid_err}: {res}")
                        logger.info("BleDataSource: Finished attempting to stop notifications.")
                    else:
                        logger.info("BleDataSource: No notifications to stop or client was already disconnected.")

                    logger.info("BleDataSource: Attempting to disconnect client in finally block...")
                    try:
                        if local_client_ref.is_connected: # Final check
                            # Give more time for a clean disconnect. The GUI is already hidden.
                            await asyncio.wait_for(local_client_ref.disconnect(), timeout=5.0)
                            logger.info("BleDataSource: Client disconnected successfully via local_client_ref in finally block.")
                        else:
                            logger.info("BleDataSource: Client was already disconnected before final disconnect call in finally block.")
                    except asyncio.TimeoutError:
                        # This will now only trigger if disconnect takes longer than 5.0s
                        logger.warning(f"BleDataSource: Timeout ({5.0}s) disconnecting client. Proceeding with shutdown.")
                    except BleakError as be:
                        logger.error(f"BleDataSource: BleakError during client disconnect in finally: {be}")
                    except Exception as e:
                        logger.error(f"BleDataSource: Generic error during client disconnect in finally: {e}")
                else:
                    logger.info("BleDataSource: local_client_ref existed but was not connected (or invalid) at start of finally block.")
            else:
                logger.info("BleDataSource: local_client_ref was None at start of finally block.")

            disconnected_event.clear()
            
            # If the start() method is ending NOT because _stop_requested was true 
            # (e.g. connection lost, scan failed and not retrying within start itself),
            # ensure GUI state is updated to idle.
            # If _stop_requested is true, the _ble_source_done_callback will typically handle the final idle state.
            if not self._stop_requested and state != "idle":
                logger.info("BleDataSource: Stop not externally requested, but start() is ending. Emitting idle state.")
                self._gui_emitter.emit_state_change("idle")
            
            stop_flag = original_stop_flag_val # Restore global stop_flag for other potential uses
            logger.info("BleDataSource: Start method's finally block completed.")

    async def stop(self):
        logger.info("BleDataSource: Stop requested.")
        self._stop_requested = True
        global stop_flag # To ensure the global stop_flag used by find_device is also set
        stop_flag = True
        
        if self._client and self._client.is_connected:
            logger.info("BleDataSource: Client is connected, setting disconnected_event to break listen loop.")
            disconnected_event.set()
        else:
            logger.info("BleDataSource: Client not connected or None, stop will primarily affect scan/connect phases.")

        logger.info("BleDataSource.stop() has signaled the active data source task to terminate and clean up.")


class CsvReplaySource(DataSource):
    def __init__(self, filepath: str, **kwargs): # emit_interval_ms no longer used directly for emission
        self.df = pd.read_csv(filepath)
        
        found_time_col = None
        exact_match_time_s = 'Time (s)'
        
        if exact_match_time_s in self.df.columns:
            found_time_col = exact_match_time_s
        else:
            # Attempt to find a case-insensitive match or 'Master Time (s)'
            for col in self.df.columns:
                if col.lower() == 'time (s)':
                    found_time_col = col
                    logger.info(f"Replay CSV: Found case-insensitive 'Time (s)' column: '{col}'. Will rename.")
                    break
                elif col.lower() == 'master time (s)':
                    found_time_col = col
                    logger.info(f"Replay CSV: Found 'Master Time (s)' column: '{col}'. Will rename.")
                    break
        
        if not found_time_col:
            err_msg = "CSV file for replay must contain a 'Time (s)' or 'Master Time (s)' column."
            logger.error(err_msg + f" Columns found: {list(self.df.columns)}")
            raise ValueError(err_msg)
            
        if found_time_col != exact_match_time_s:
            self.df = self.df.rename(columns={found_time_col: exact_match_time_s})
            logger.info(f"Replay CSV: Renamed column '{found_time_col}' to '{exact_match_time_s}'.")

        self.df.attrs['filepath'] = filepath 
        self._finished_event = asyncio.Event()
        self._stop_requested = False # For future use if we want cancellable bulk load

    async def start(self):
        global start_time, data_buffers
        
        self._stop_requested = False
        self._finished_event.clear()
        
        # Set a conceptual start_time for the session.
        # The actual timestamps in data_buffers will come directly from the CSV's 'Time (s)' column.
        start_time = datetime.datetime.now() 
        
        logger.info(f"CsvReplaySource: Starting bulk load from {os.path.basename(self.df.attrs.get('filepath', 'Unknown CSV'))}.")

        try:
            # Process all rows in the DataFrame
            for idx, row in self.df.iterrows():
                if self._stop_requested:
                    logger.info("CsvReplaySource: Bulk load interrupted by stop request.")
                    break

                # Use the 'Time (s)' column directly from the CSV for the timestamp
                # This ensures the replayed data retains its original relative timing.
                csv_row_time_original = row['Time (s)']
                
                for col_name, value in row.items():
                    if col_name == 'Time (s)': # Skip the time column itself
                        continue
                    
                    data_type = col_name
                    try:
                        numeric_value = float(value) # Ensure value is float
                        if pd.isna(numeric_value): # Handle NaN values if any
                            # logger.debug(f"CsvReplaySource: NaN value for {data_type} at time {csv_row_time_original:.3f}s. Skipping.")
                            continue

                        if data_type not in data_buffers:
                            data_buffers[data_type] = []
                        
                        # Append with the original CSV time
                        data_buffers[data_type].append((csv_row_time_original, numeric_value))

                    except ValueError:
                        logger.warning(f"CsvReplaySource: Could not convert value '{value}' for data_type '{data_type}' at CSV time {csv_row_time_original:.2f}s to float. Skipping.")
                    except Exception as e:
                        logger.error(f"CsvReplaySource: Error processing column {col_name} at CSV time {csv_row_time_original:.2f}s: {e}")
                
                # Call derived data computation after each conceptual "sample" is loaded
                compute_all_derived_data(csv_row_time_original)

            # After loading, sort each buffer by time to ensure correctness for bisect, etc.
            for data_type in data_buffers.keys():
                data_buffers[data_type].sort(key=lambda x: x[0])
            
            logger.info("CsvReplaySource: Bulk load completed.")

        except Exception as e:
            logger.error(f"CsvReplaySource: Error during bulk data load: {e}", exc_info=True)
        finally:
            # Signal that processing is done
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(self._finished_event.set)
            else:
                self._finished_event.set()
        
        await self._finished_event.wait() # The event will be set very quickly
        logger.info("CsvReplaySource start method finished after bulk load.")

    async def stop(self):
        logger.info("CsvReplaySource: Stop requested (during bulk load or if it were continuous).")
        self._stop_requested = True # This flag can be checked during bulk load
        
        # Ensure finished_event is set if not already
        if not self._finished_event.is_set():
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(self._finished_event.set)
            else:
                self._finished_event.set()



# --- Global variables ---
disconnected_event = asyncio.Event()
last_received_time = 0

data_buffers: Dict[str, List[Tuple[float, float]]] = {}
start_time: Optional[datetime.datetime] = None 
stop_flag = False
state = "idle"
current_task: Optional[asyncio.Task] = None
loop: Optional[asyncio.AbstractEventLoop] = None

plotting_paused = False
flowing_interval = 10.0 

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


# --- Derived / Fused Data Framework ----------------------------------------------
class DerivedDataDefinition:
    """
    data_type          – name of the new (virtual) data-series that will appear in
                         data_buffers and can be referenced by any GUI component.
    dependencies       – list of RAW OR DERIVED data_type strings the function needs
                         (they, in turn, resolve to concrete UUIDs through
                         device_config or other DerivedDataDefinitions).
    compute_func()     – function that returns the **latest** value of the derived
                         quantity (or None if it cannot be computed yet).
                         It MUST read its own inputs from the global `data_buffers`
                         and MUST NOT modify anything except appending its result.
    """
    def __init__(self,
                 data_type: str,
                 dependencies: List[str],
                 compute_func: Callable[[], Optional[float]]):
        self.data_type = data_type
        self.dependencies = dependencies
        self.compute_func = compute_func

# global registry of all derived data
derived_data_definitions: Dict[str, DerivedDataDefinition] = {}

def register_derived_data(defn: DerivedDataDefinition):
    if defn.data_type in derived_data_definitions:
        logger.warning(f"Derived data '{defn.data_type}' already registered – overriding.")
    derived_data_definitions[defn.data_type] = defn
    logger.info(f"Registered derived data: {defn.data_type} ; depends on {defn.dependencies}")

def _all_dependency_uuids(dep_list: List[str], dev_cfg: DeviceConfig) -> Set[str]:
    """Returns the concrete UUIDs needed (recursively) for the given dependency list."""
    req_uuids: Set[str] = set()
    for dt in dep_list:
        uuid = dev_cfg.get_uuid_for_data_type(dt)
        if uuid:
            req_uuids.add(uuid)
        elif dt in derived_data_definitions:
            req_uuids |= _all_dependency_uuids(derived_data_definitions[dt].dependencies, dev_cfg)
    return req_uuids

def compute_all_derived_data(current_relative_time: float):
    """
    Call this every time new raw data arrive.
    Any derived value that CAN be computed is pushed to data_buffers
    with the SAME timestamp as the triggering raw sample.
    """
    for defn in derived_data_definitions.values():
        # need at least ONE sample available for each dependency
        deps_ok = all(d in data_buffers and data_buffers[d] for d in defn.dependencies)
        if not deps_ok:
            continue
        try:
            val = defn.compute_func()
        except Exception as e:
            logger.error(f"Derived '{defn.data_type}' compute error: {e}")
            continue
        if val is None:
            continue
        buf = data_buffers.setdefault(defn.data_type, [])
        buf.append((current_relative_time, val))
# ---------------------------------------------------------------------------------


def get_value_at_time(data_type_key: str, target_time: float, buffers: Dict[str, List[Tuple[float, float]]]) -> Optional[float]:
    if data_type_key not in buffers or not buffers[data_type_key]:
        return None
    
    buffer = buffers[data_type_key]
    times = [item[0] for item in buffer]
    
    idx = bisect.bisect_right(times, target_time)
    
    if idx == 0:
        if buffer: return buffer[0][1] 
        return None
        
    return buffer[idx-1][1]

#####################################################################################################################
# Start of customizable section
#####################################################################################################################
"""
Section for customizing device configuration, data handling, and plotting.

The following sections outline the steps to set up the GUI and device configuration:
1. Data handlers for different characteristics
2. Data fusion/derived handlers for derived data 
3. Register the derived data handlers and their dependencies
4. Device configuration (add UUIDs AND `produces_data_types`)
5. Define GUI Component Classes (e.g., plot class, indicator class, etc.)
6. Define Tab Layout Configuration using the components
"""


# 1. --- Data Handlers for Different Characteristics ---

# --- Constants needed for Insole Data Handling ---
ADC_MAX_VOLTAGE = 3.3           # Maximal voltage expected from ADC
# --- Define the keys specifically for the heatmap sensors ---
HEATMAP_KEYS = ["A0C0", "A1C0", "A2C0", "A0C1", "A1C1", "A2C1", "A0C2", "A1C2", "A2C2", "A1C3", "A2C3"]
NUM_HEATMAP_SENSORS = len(HEATMAP_KEYS)


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
], dtype=np.float32) # (A0C3 is ommited, broken flex sensor)


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

# optical_flow_handler.py

# module‐level accumulators,!!!NEED TO BE explicitly CLEARED in "clear_gui_action"!!!
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
    estimated weight (from sum).

    Args:
        data: The raw bytearray received via BLE.

    Returns:
        A dictionary containing keys for:
        - Each FSR sensor ('A0C0'...'A2C3') with its pressure value (relative to initial sensitivity).
        - 'estimated_weight' with the calculated weight value.
        Returns an empty dictionary if parsing or critical calculation fails.
    """
    try:
        data_string = data.decode('utf-8')
    except UnicodeDecodeError:
        data_logger.warning(f"Received non-UTF8 raw bytes for insole: {data}")
        return {}

    output_dict: Dict[str, float] = {}
    summed_gained_voltage = 0.0             # Accumulator for gained voltages (heatmap only)
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

            # --- Check if it's one of the Heatmap Sensors ---
            if key in HEATMAP_KEYS:
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
            else: data_logger.debug(f"Skipping unrecognized key: '{key}'")

    except ValueError as e:
        data_logger.warning(f"ValueError parsing part '{part}': {e} in string: {data_string}")
        return {} # Indicate failure
    except Exception as e:
        data_logger.error(f"Error parsing data string '{data_string}' (part: '{part}'): {e}")
        return {} # Indicate failure

    # --- Perform final calculations ---

    # Weight Estimation
    output_dict['estimated_weight'] = summed_gained_voltage * VOLTAGE_TO_WEIGHT_FACTOR


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
            "impedance_phase_deg": phase_Z * 180.0 / math.pi,  # Convert to degrees
            "real_part_kohm": real_kohm,
            "imag_part_kohm": imag_kohm,
        }

    except (ValueError, ZeroDivisionError) as e:
        data_logger.error(f"Error processing ADC data '{text}': {e}")
        return {}


def handle_quaternion_data(data: bytearray) -> dict:
    try:
        text = data.decode("utf-8").strip()
        parts = text.split(",")
        if len(parts) != 4:
            data_logger.error("Invalid quaternion payload: expected 4 parts, got %d", len(parts))
            return {}
        w = float(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
        data_logger.info(f"Quaternion Data: w={w:.3f}, x={x:.3f}, y={y:.3f}, z={z:.3f}") # Optional: for verbose logging
        return {"quat_w": w, "quat_x": x, "quat_y": y, "quat_z": z}
    except ValueError as e:
        data_logger.error(f"Error parsing quaternion data (ValueError): {e} - Data: '{data.decode('utf-8', errors='ignore')}'")
        return {}
    except Exception as e:
        data_logger.error(f"Error parsing quaternion data (Exception): {e} - Data: '{data.decode('utf-8', errors='ignore')}'")
        return {}
# ------------------------------------------------------------------


# 2. --- Derived/Fusion Data Handlers ---

# derived data example:  |Z| change-speed  (ABS(Δ|Z|) / Δt   in  Ω/s)
# dependencies:  |Z| (Ω)  (from ADC characteristic)
def _compute_dZ_dt(min_span_sec: float = 0.08,
                   window_sec: float   = 0.40,
                   history_maxlen: int = 12
                  ) -> Optional[float]:
    """
    Returns d|Z|/dt (Ω/s), signed:
      • positive when |Z| rises
      • negative when |Z| falls

    Uses a sliding window (≤ window_sec) of at most history_maxlen samples,
    computes a least-squares slope, and only returns a value once the time
    span ≥ min_span_sec.
    """

    # initialize persistent history on first call
    hist_attr = "_compute_dZ_dt_history"
    if not hasattr(_compute_dZ_dt, hist_attr):
        setattr(_compute_dZ_dt, hist_attr,
                deque(maxlen=history_maxlen))
    history: deque = getattr(_compute_dZ_dt, hist_attr)

    # get latest |Z|
    buf = data_buffers.get('impedance_magnitude_ohm', [])
    if not buf:
        return None
    t_now, z_now = buf[-1]

    # append to history
    history.append((t_now, z_now))

    # drop samples older than window_sec
    while history and (t_now - history[0][0]) > window_sec:
        history.popleft()

    # need at least 3 points for a stable slope
    if len(history) < 3:
        return None

    # check total timespan
    span = history[-1][0] - history[0][0]
    if span < min_span_sec:
        return None

    # prepare for least-squares slope
    times = np.array([pt[0] for pt in history], dtype=np.float64)
    mags  = np.array([pt[1] for pt in history], dtype=np.float64)
    t_mean, m_mean = times.mean(), mags.mean()
    t_c = times - t_mean
    m_c = mags  - m_mean

    denom = np.dot(t_c, t_c)
    if denom <= 1e-12:
        return None

    slope = np.dot(t_c, m_c) / denom  # Ω/s, signed
    return float(slope)

# ------------------------------------------------------------------

# 3. --- Register the Derived Data Handlers and their Dependencies ---

# Register the derived / fused data definitions
register_derived_data(
    DerivedDataDefinition(
        data_type='impedance_change_speed_ohm_per_s',
        dependencies=['impedance_magnitude_ohm'],
        compute_func=_compute_dZ_dt,
    )
)
# ------------------------------------------------------------------

# 4. --- Device Configuration (Initial Device Name still set here) ---

# This object's 'name' attribute will be updated by the dropdown menu
device_config = DeviceConfig(
    name="Nano33IoT", # Initial default name
    service_uuid="19B10000-E8F2-537E-4F6C-D104768A1214",
    characteristics=[
        # IMU Characteristics
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
                             produces_data_types=HEATMAP_KEYS + ['estimated_weight']),
        CharacteristicConfig(uuid="19B10009-E8F2-537E-4F6C-D104768A1214",
            handler=handle_adc_data,
            produces_data_types=['impedance_magnitude_ohm', 'impedance_phase_deg', 'real_part_kohm', 'imag_part_kohm']
        ),
        CharacteristicConfig(uuid="19B10012-E8F2-537E-4F6C-D104768A1214",
            handler=handle_optical_xy_data,
            produces_data_types=['opt_dx', 'opt_dy', 'opt_cum_x', 'opt_cum_y']
        ),
        CharacteristicConfig(uuid="19B10014-E8F2-537E-4F6C-D104768A1214",
            handler=handle_tof_data,
            produces_data_types=['tof_distance_mm', 'tof_brightness_kcps']
        ),
        CharacteristicConfig(uuid="19B10016-E8F2-537E-4F6C-D104768A1214",
            handler=handle_ankle_angle_data,
            produces_data_types=['ankle_xz', 'ankle_yz']
        ),
        CharacteristicConfig(uuid="19B10020-E8F2-537E-4F6C-D104768A1214", handler=handle_quaternion_data,
                     produces_data_types=['quat_w', 'quat_x', 'quat_y', 'quat_z']),
    ],
    find_timeout=10.0,
    data_timeout=1.0
)

# List of available device names for the dropdown
AVAILABLE_DEVICE_NAMES = ["Nano33IoT", "NanoESP32"]
# ------------------------------------------------------------------

# 5. --- GUI Component Classes ---

# This is the base class for all GUI components.
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
            text_content = "Missing UUID(s):\n" + "\n".join(sorted(missing_uuids_for_component))


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


    def handle_missing_replay_data(self, missing_data_types_for_component: Set[str]):
        """
        Called by GuiManager during replay mode if required data types are not in data_buffers.
        """
        if missing_data_types_for_component:
            # Create a comma-separated list, limit length for display
            missing_types_str = ", ".join(sorted(list(missing_data_types_for_component)))
            if len(missing_types_str) > 100: # Limit display length
                missing_types_str = missing_types_str[:97] + "..."
            text_content = f"Data Not Loaded From CSV for:\n{missing_types_str}"

            if not self.uuid_missing_overlay: # Re-use the same overlay QLabel object
                self.uuid_missing_overlay = QLabel(text_content, self)
                self.uuid_missing_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.uuid_missing_overlay.setStyleSheet("background-color: rgba(120, 120, 50, 200); color: white; font-weight: bold; border-radius: 5px; padding: 10px;") # Slightly different color
                self.uuid_missing_overlay.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
                self.uuid_missing_overlay.adjustSize()
                self.uuid_missing_overlay.raise_()
                self.uuid_missing_overlay.setVisible(True)
                self._position_overlay()
            else:
                self.uuid_missing_overlay.setText(text_content)
                self.uuid_missing_overlay.setStyleSheet("background-color: rgba(120, 120, 50, 200); color: white; font-weight: bold; border-radius: 5px; padding: 10px;") # Ensure style
                self.uuid_missing_overlay.adjustSize()
                self._position_overlay()
                self.uuid_missing_overlay.setVisible(True)
                self.uuid_missing_overlay.raise_()
        else: # No missing data types for this component
            # If the overlay is visible and it's a "CSV Not Loaded" message, hide it.
            # This check prevents hiding a "UUID Missing" message if both were somehow active.
            if self.uuid_missing_overlay and "CSV" in self.uuid_missing_overlay.text():                self.uuid_missing_overlay.setVisible(False)

# --- Specific Component Implementations ---

# --- TimeSeriesPlotComponent ---
class TimeSeriesPlotComponent(BaseGuiComponent):
    """A GUI component that displays time-series data using pyqtgraph."""

    SLIDER_FLOAT_PRECISION_FACTOR = 100  # For 2 decimal places (e.g., 0.01s resolution)

    def __init__(self, config: Dict[str, Any], data_buffers_ref: Dict[str, List[Tuple[float, float]]], device_config_ref: DeviceConfig, parent: Optional[QWidget] = None):
        super().__init__(config, data_buffers_ref, device_config_ref, parent)

        self.plot_widget = pg.PlotWidget()
        self.plot_item: pg.PlotItem = self.plot_widget.getPlotItem()
        self.lines: Dict[str, pg.PlotDataItem] = {} # data_type -> PlotDataItem
        # self.uuid_not_found_text: Optional[pg.TextItem] = None # Use base class overlay instead
        self._required_data_types: Set[str] = set() # Internal store
        self.missing_relevant_uuids: Set[str] = set() # UUIDs this plot needs that are missing

        self._setup_plot()



        # --- Main Layout ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget)



        # --- Replay Slider Controls ---
        self.replay_controls_container = QWidget() # Main container for slider and export button
        replay_controls_main_layout = QVBoxLayout(self.replay_controls_container)
        replay_controls_main_layout.setContentsMargins(0,0,0,0)
        replay_controls_main_layout.setSpacing(3) # Small spacing between slider row and button

        self.replay_slider_widget = QWidget() # Widget for the slider and its labels
        slider_labels_layout = QHBoxLayout(self.replay_slider_widget)
        slider_labels_layout.setContentsMargins(5,0,5,0) 

        self.min_time_display = QLabel("0.00") # Changed from QLineEdit to QLabel
        self.min_time_display.setFixedWidth(50) # Adjusted width
        self.min_time_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.replay_time_slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.replay_time_slider.setRange(0, 0) 

        self.max_time_display = QLabel("0.00") # Changed from QLineEdit to QLabel
        self.max_time_display.setFixedWidth(50) # Adjusted width
        self.max_time_display.setAlignment(Qt.AlignmentFlag.AlignCenter)

        slider_labels_layout.addWidget(QLabel("View:"))
        slider_labels_layout.addWidget(self.min_time_display)
        slider_labels_layout.addWidget(self.replay_time_slider)
        slider_labels_layout.addWidget(self.max_time_display)
        slider_labels_layout.addWidget(QLabel("s"))
        
        replay_controls_main_layout.addWidget(self.replay_slider_widget)

        # --- Export Button for Replay ---
        self.export_replay_pdf_button = QPushButton("Export Visible Plot to PDF")
        self.export_replay_pdf_button.clicked.connect(self._request_replay_pdf_export)
        # Center the button
        export_button_layout = QHBoxLayout()
        export_button_layout.addStretch()
        export_button_layout.addWidget(self.export_replay_pdf_button)
        export_button_layout.addStretch()
        replay_controls_main_layout.addLayout(export_button_layout)


        self.replay_controls_container.setVisible(False) # Initially hidden
        layout.addWidget(self.replay_controls_container)
        self.setLayout(layout)

        # --- Connect Signals ---
        self.replay_time_slider.valueChanged.connect(self._on_replay_slider_changed)
        # Removed connections for min/max_time_textbox.editingFinished


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


        # The replay slider (previously export_slider) functionality
        # is now handled by self.replay_time_slider and its associated widgets,
        # which are set up in __init__ and added to the layout there.
        # This self.export_slider is no longer needed for replay scrubbing.
        # If a separate export-only slider is ever needed, it would require
        # its own distinct setup and handler.
        pass # Or simply remove these lines if no other purpose for an 'export_slider' here.


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
        # If in flowing mode AND plotting is paused, then return.
        # Otherwise (e.g., not flowing (slider mode), or flowing and not paused), proceed to update.
        if plotting_paused and is_flowing:
            return



        # --- Determine Time Axis Range and Slider Configuration ---
        filter_min_time_float: float
        filter_max_time_float: float

        if is_flowing:

            filter_min_time_float = max(0, current_relative_time - flowing_interval)
            filter_max_time_float = current_relative_time
            self.replay_controls_container.setVisible(False) # Hide the whole replay controls container
            self.plot_item.setXRange(filter_min_time_float, filter_max_time_float, padding=0.02)

        else:  # Not flowing mode: use slider to control view
            self.replay_controls_container.setVisible(True) # Show the whole replay controls container


            actual_min_data_time_float = float('inf')
            actual_max_data_time_float = 0.0 
            found_any_data_for_slider = False

            for data_type_slider in self.get_required_data_types():
                uuid_slider = self.device_config_ref.get_uuid_for_data_type(data_type_slider)
                if uuid_slider and uuid_slider in self.missing_relevant_uuids:
                    continue
                if data_type_slider in self.data_buffers_ref and self.data_buffers_ref[data_type_slider]:
                    series_times = [item[0] for item in self.data_buffers_ref[data_type_slider]]
                    if series_times:
                        actual_min_data_time_float = min(actual_min_data_time_float, series_times[0])
                        actual_max_data_time_float = max(actual_max_data_time_float, series_times[-1])
                        found_any_data_for_slider = True
            
            if not found_any_data_for_slider:
                actual_min_data_time_float = 0.0
                actual_max_data_time_float = max(flowing_interval, 0.1) 
            
            # Ensure max is greater than min for float times
            if actual_max_data_time_float <= actual_min_data_time_float:
                actual_max_data_time_float = actual_min_data_time_float + 0.1 # Ensure a small valid range

            # Convert float times to scaled integers for the QRangeSlider
            slider_abs_min_range_int = int(math.floor(actual_min_data_time_float * self.SLIDER_FLOAT_PRECISION_FACTOR))
            slider_abs_max_range_int = int(math.ceil(actual_max_data_time_float * self.SLIDER_FLOAT_PRECISION_FACTOR))

            # Ensure max_int > min_int for slider range
            if slider_abs_max_range_int <= slider_abs_min_range_int:
                slider_abs_max_range_int = slider_abs_min_range_int + 1 

            self.replay_time_slider.blockSignals(True)
            current_slider_min_int, current_slider_max_int = self.replay_time_slider.value()
            range_changed = False
            if self.replay_time_slider.minimum() != slider_abs_min_range_int or \
               self.replay_time_slider.maximum() != slider_abs_max_range_int:
                self.replay_time_slider.setRange(slider_abs_min_range_int, slider_abs_max_range_int)
                # If overall range changes, reset slider handles to cover the full new range
                self.replay_time_slider.setValue((slider_abs_min_range_int, slider_abs_max_range_int))
                current_slider_min_int, current_slider_max_int = self.replay_time_slider.value() # Update current values
                range_changed = True
            self.replay_time_slider.blockSignals(False)

            # Get current slider selection (scaled integers)
            selected_min_int, selected_max_int = self.replay_time_slider.value()
            filter_min_time_float = float(selected_min_int) / self.SLIDER_FLOAT_PRECISION_FACTOR
            filter_max_time_float = float(selected_max_int) / self.SLIDER_FLOAT_PRECISION_FACTOR

            # If the overall range changed, text displays need update from slider's new full range
            # Also, ensure text displays are in sync if they weren't the source of a change
            # Convert current label text to float for comparison, handle potential errors if label is not a number
            try:
                min_label_val = float(self.min_time_display.text())
                max_label_val = float(self.max_time_display.text())
            except ValueError: # If labels contain non-numeric text (e.g. "--"), force update
                min_label_val = -1.0 
                max_label_val = -1.0

            if range_changed or \
               abs(min_label_val - filter_min_time_float) > 1e-3 or \
               abs(max_label_val - filter_max_time_float) > 1e-3:
                self.min_time_display.setText(f"{filter_min_time_float:.2f}")
                self.max_time_display.setText(f"{filter_max_time_float:.2f}")
            
            self.plot_item.setXRange(filter_min_time_float, filter_max_time_float, padding=0.02)



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
                # Filter data based on filter_min_time_float and filter_max_time_float determined earlier
                # 'data' here refers to the data_series for the current data_type
                start_idx = bisect.bisect_left(data, filter_min_time_float, key=lambda x: x[0])
                end_idx = bisect.bisect_right(data, filter_max_time_float, key=lambda x: x[0])
                plot_data_tuples = data[start_idx:end_idx]

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




        # Determine if any overlay (UUID or CSV missing) is active for this plot's data
        is_any_data_issue_overlay_active = False
        if self.uuid_missing_overlay and self.uuid_missing_overlay.isVisible():
            # Check if the overlay is due to UUIDs missing for this plot
            if self.missing_relevant_uuids: 
                is_any_data_issue_overlay_active = True
            # Check if the overlay is due to CSV data not loaded for this plot
            elif "CSV" in self.uuid_missing_overlay.text():
                for dt_check in self.get_required_data_types():
                     if dt_check not in self.data_buffers_ref or not self.data_buffers_ref[dt_check]:
                         is_any_data_issue_overlay_active = True
                         break
        
        if data_updated_in_plot and not is_any_data_issue_overlay_active:
            self.plot_item.enableAutoRange(axis='y', enable=True)
        elif is_any_data_issue_overlay_active:
             self.plot_item.enableAutoRange(axis='y', enable=False) # Disable auto Y range
             self.plot_item.setYRange(-1, 1, padding=0.1) # Set a default Y range
             # Clear lines if an overlay is active
             for data_type, line in self.lines.items():
                 line.setData(x=[], y=[])
        # If !data_updated_in_plot AND no overlay, auto-range might be okay, or let it be.
        # If data_buffers are empty for this plot's types, lines will be empty anyway.



    def clear_component(self):
        """Clears the plot lines, resets axes, and hides replay controls."""
        for line in self.lines.values():
            line.setData(x=[], y=[])

        # Hide overlay via base class method (handles None check)
        self.handle_missing_uuids(set())
        self.missing_relevant_uuids.clear() # Also clear internal set

        # Reset view ranges for "live" or "flowing" mode
        self.plot_item.setXRange(0, flowing_interval, padding=0.02)
        self.plot_item.setYRange(-1, 1, padding=0.1) 
        self.plot_item.enableAutoRange(axis='y', enable=True)


        # --- Explicitly hide and reset replay slider components ---
        if hasattr(self, 'replay_controls_container'): # Check for the new container
            self.replay_controls_container.setVisible(False)
        if hasattr(self, 'replay_time_slider'):
            self.replay_time_slider.blockSignals(True)
            self.replay_time_slider.setRange(0,0) 
            self.replay_time_slider.setValue((0,0)) 
            self.replay_time_slider.blockSignals(False)
        if hasattr(self, 'min_time_display'): # Changed from textbox to display
            self.min_time_display.setText("0.00")
        if hasattr(self, 'max_time_display'): # Changed from textbox to display
            self.max_time_display.setText("0.00") # Or a default like str(flowing_interval)
        

        # Reset validators to a broad initial range if necessary,
        # or let update_component handle it when new data comes in live.
        # For now, just resetting text is probably fine.
        logger.debug(f"TimeSeriesPlotComponent '{self.config.get('title', 'Plot')}' cleared, replay controls hidden.")



    def _on_replay_slider_changed(self, value_tuple: Tuple[int, int]):
        t0_int, t1_int = value_tuple  # Integer values from the slider

        # Scale to float based on precision factor
        t0_float = float(t0_int) / self.SLIDER_FLOAT_PRECISION_FACTOR
        t1_float = float(t1_int) / self.SLIDER_FLOAT_PRECISION_FACTOR
        
        # Ensure t0 is not greater than t1 if slider handles cross (shouldn't happen with QRangeSlider)
        t0_float = min(t0_float, t1_float)


        # Update text displays
        self.min_time_display.setText(f"{t0_float:.2f}")
        self.max_time_display.setText(f"{t1_float:.2f}")


        logger.debug(f"Replay slider changed for '{self.config.get('title', 'Plot')}': [{t0_float:.2f}s, {t1_float:.2f}s]")
        
        # Request GUI update to refresh the plot based on the new slider range
        self._request_gui_update_for_yrange()

    def _request_replay_pdf_export(self):
        if not self.replay_time_slider.isVisible(): # Should only be callable when slider is visible
            logger.warning("Replay PDF export requested but slider is not visible.")
            return

        min_t_int, max_t_int = self.replay_time_slider.value()
        min_t_float = float(min_t_int) / self.SLIDER_FLOAT_PRECISION_FACTOR
        max_t_float = float(max_t_int) / self.SLIDER_FLOAT_PRECISION_FACTOR

        # Try to find the MainWindow instance to call its export handler
        # This assumes the component is eventually parented to MainWindow
        main_window_instance = self.window() 
        if not isinstance(main_window_instance, MainWindow):
            # Fallback search if self.window() is not directly MainWindow (e.g., if nested in complex ways)
            for widget in QApplication.topLevelWidgets():
                if isinstance(widget, MainWindow):
                    main_window_instance = widget
                    break
            else: # If loop finishes without break
                main_window_instance = None
        
        if isinstance(main_window_instance, MainWindow):
            logger.info(f"Requesting PDF export for '{self.config.get('title', 'Plot')}' "
                        f"from {min_t_float:.2f}s to {max_t_float:.2f}s.")
            main_window_instance.handle_component_replay_export(self, min_t_float, max_t_float)
        else:
            logger.error("Could not find MainWindow instance to handle replay PDF export.")
            QMessageBox.critical(self, "Export Error", "Could not find main application window to handle export request.")

    # def _update_slider_from_textboxes(self):
    #     # This method is no longer needed as text displays are not editable.
    #     # If it were to be kept for programmatic updates, it would need significant changes.
    #     pass

# --- HEATMAP COMPONENT ---

class PressureHeatmapComponent(BaseGuiComponent):
    """ Displays a pressure heatmap based on sensor data. """
    SLIDER_FLOAT_PRECISION_FACTOR_HEATMAP = 100 # For 0.01s resolution, adjust as needed

    # --- Constants specific to this component ---
    DEFAULT_INSOLE_IMAGE_PATH = str(ASSETS_DIR / 'Sohle_rechts.png')
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


        self.time_slider_heatmap = QSlider(Qt.Orientation.Horizontal)

        self.current_replay_time_label_heatmap = QLabel("0.00s")
        self.current_replay_time_label_heatmap.setFixedWidth(50) # Adjust width as needed
        self.current_replay_time_label_heatmap.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.time_slider_heatmap.setRange(0, 0)
        self.time_slider_heatmap.valueChanged.connect(self.on_time_slider_heatmap)
        # Slider itself is not set to visible/invisible directly anymore
        
        # Create a container widget for the scrub time controls
        self.scrub_time_widget_heatmap = QWidget()
        time_slider_layout = QHBoxLayout(self.scrub_time_widget_heatmap)
        time_slider_layout.setContentsMargins(0,0,0,0) # Minimize extra space
        time_slider_layout.addWidget(QLabel("Scrub Time:"))
        time_slider_layout.addWidget(self.time_slider_heatmap)

        time_slider_layout.addWidget(self.current_replay_time_label_heatmap)

        self.controls_layout.addWidget(self.scrub_time_widget_heatmap)
        self.scrub_time_widget_heatmap.setVisible(False) # Initially hidden


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


    def _update_controls_based_on_data_status(self):
        """Helper to enable/disable controls based on any active overlay."""
        is_overlay_active = self.uuid_missing_overlay and self.uuid_missing_overlay.isVisible()
        
        if hasattr(self, 'controls_widget'):
            self.controls_widget.setEnabled(not is_overlay_active)
        if is_overlay_active:
            # If an overlay is active (either UUID or CSV data missing), clear the visual component
            # But don't call full clear_component as that might reset slider ranges we want to keep.
            self.pressure_values.fill(0.0)
            self.center_of_pressure = None
            self.cop_trail.clear()
            if self.heatmap_qimage: # Check if precomputation succeeded
                calculated_pressures = self._calculate_pressure_fast() # Will be zeros
                self._render_heatmap_to_buffer(calculated_pressures)
                self._update_display_pixmap()


    def handle_missing_uuids(self, missing_uuids_for_component: Set[str]):
        """ Shows overlay and disables controls if the required UUID is missing. """
        super().handle_missing_uuids(missing_uuids_for_component)
        self._update_controls_based_on_data_status()

    def handle_missing_replay_data(self, missing_data_types_for_component: Set[str]):
        super().handle_missing_replay_data(missing_data_types_for_component)
        self._update_controls_based_on_data_status()

    # --- Update and Clear ---
    def update_component(self, current_relative_time: float, is_flowing: bool):

        # Directly use is_flowing argument, and global plotting_paused
        if plotting_paused and is_flowing: # If live and paused, do nothing.
             return
        if not self.heatmap_qimage: return # Skip update if precomputation failed


        if not is_flowing: # This implies replay mode or "Flowing Mode" checkbox is unchecked
            self.scrub_time_widget_heatmap.setVisible(True)
            
            actual_min_data_time_hm = float('inf')
            actual_max_data_time_hm = 0.0
            found_any_data_hm = False

            for key_hm_slider in HEATMAP_KEYS: # HEATMAP_KEYS are the required data types
                if key_hm_slider in self.data_buffers_ref and self.data_buffers_ref[key_hm_slider]:
                    series_times_hm = [item[0] for item in self.data_buffers_ref[key_hm_slider]]
                    if series_times_hm:
                        actual_min_data_time_hm = min(actual_min_data_time_hm, series_times_hm[0])
                        actual_max_data_time_hm = max(actual_max_data_time_hm, series_times_hm[-1])
                        found_any_data_hm = True
            
            if not found_any_data_hm:
                actual_min_data_time_hm = 0.0
                actual_max_data_time_hm = 1.0 
            
            if actual_max_data_time_hm <= actual_min_data_time_hm:
                actual_max_data_time_hm = actual_min_data_time_hm + 0.1 

            slider_min_int = int(math.floor(actual_min_data_time_hm * self.SLIDER_FLOAT_PRECISION_FACTOR_HEATMAP))
            slider_max_int = int(math.ceil(actual_max_data_time_hm * self.SLIDER_FLOAT_PRECISION_FACTOR_HEATMAP))

            if slider_max_int <= slider_min_int:
                slider_max_int = slider_min_int + 1 

            self.time_slider_heatmap.blockSignals(True)
            if self.time_slider_heatmap.minimum() != slider_min_int or \
               self.time_slider_heatmap.maximum() != slider_max_int:
                current_val = self.time_slider_heatmap.value()
                self.time_slider_heatmap.setRange(slider_min_int, slider_max_int)
                # Try to keep value if within new range, else set to min
                if slider_min_int <= current_val <= slider_max_int:
                    self.time_slider_heatmap.setValue(current_val)
                else:
                    self.time_slider_heatmap.setValue(slider_min_int)
                # Trigger initial render for the new range/value if not user-driven
                # self.on_time_slider_heatmap(self.time_slider_heatmap.value()) 
            self.time_slider_heatmap.blockSignals(False)
            
            # If in replay/slider mode, rendering is driven by on_time_slider_heatmap()
            # or an initial call after range setting if needed.
            # For now, let slider interaction trigger the render.
            # If no interaction yet, it might show last state or be blank until slider moved.
            # To ensure it shows data for current_relative_time (if that's desired concept for "latest" in paused mode):
            if plotting_paused: # This means we are in slider mode
                 # Render for the current slider value. If slider was just set up, this renders initial view.
                 self.render_heatmap_for_time(float(self.time_slider_heatmap.value()) / self.SLIDER_FLOAT_PRECISION_FACTOR_HEATMAP)

        else: # is_flowing is True (Live mode)
            self.scrub_time_widget_heatmap.setVisible(False)
            # --- Get Latest Pressure Data AND RESCALE IT (Live Mode Path) ---
            data_found_count = 0
            rescaled_pressures = np.zeros(self.NUM_SENSORS, dtype=np.float32)
            for i, key in enumerate(HEATMAP_KEYS):
                if key in self.data_buffers_ref and self.data_buffers_ref[key]:
                    initial_pressure = self.data_buffers_ref[key][-1][1]
                    if INITIAL_PRESSURE_SENSITIVITY > 1e-6:
                        current_pressure = initial_pressure * (self.current_pressure_sensitivity / INITIAL_PRESSURE_SENSITIVITY)
                    else: current_pressure = 0.0
                    rescaled_pressures[i] = max(0.0, current_pressure)
                    data_found_count += 1
            if data_found_count > 0: self.pressure_values = rescaled_pressures

            # --- Live Heatmap Calculation and Rendering ---
            current_cop_qpoint = self._calculate_center_of_pressure() # Uses self.pressure_values
            self.center_of_pressure = current_cop_qpoint
            if current_cop_qpoint is not None:
                is_different = True
                if self.cop_trail:
                    last_point = self.cop_trail[-1]
                    if abs(current_cop_qpoint.x() - last_point.x()) < 0.1 and abs(current_cop_qpoint.y() - last_point.y()) < 0.1: is_different = False
                if is_different: self.cop_trail.append(current_cop_qpoint)

            calculated_pressures = self._calculate_pressure_fast()
            self._render_heatmap_to_buffer(calculated_pressures)
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
        super().handle_missing_replay_data(set()) # Ensure replay overlay is also cleared


        # --- Reset time slider and hide its container ---
        if hasattr(self, 'time_slider_heatmap'):
            self.time_slider_heatmap.blockSignals(True)
            self.time_slider_heatmap.setRange(0, 0) # Reset range
            self.time_slider_heatmap.setValue(0)    # Reset value
            self.time_slider_heatmap.blockSignals(False)
            # The slider's visibility is controlled by its parent scrub_time_widget_heatmap

            
        # Update the replay time label to reflect the current slider value
        current_slider_val_for_label = self.time_slider_heatmap.value()
        time_sec_for_label = float(current_slider_val_for_label) / self.SLIDER_FLOAT_PRECISION_FACTOR_HEATMAP
        if hasattr(self, 'current_replay_time_label_heatmap'):
            self.current_replay_time_label_heatmap.setText(f"{time_sec_for_label:.2f}s")

        if hasattr(self, 'scrub_time_widget_heatmap'):
            self.scrub_time_widget_heatmap.setVisible(False)

        if hasattr(self, 'current_replay_time_label_heatmap'):
            self.current_replay_time_label_heatmap.setText("0.00s")

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

    def _calculate_center_of_pressure(self, pressures_array: Optional[np.ndarray] = None) -> Optional[QPointF]:
        # Use the provided pressures_array if given, otherwise use self.pressure_values
        pressures_to_use = pressures_array if pressures_array is not None else self.pressure_values
        
        # Ensure pressures_to_use is a valid numpy array before proceeding
        if pressures_to_use is None or not isinstance(pressures_to_use, np.ndarray) or pressures_to_use.size == 0:
            return None

        pressures = np.maximum(pressures_to_use, 0.0) # Ensure non-negative
        total_pressure = np.sum(pressures)

        if total_pressure < 1e-9: # Use a smaller epsilon for float comparisons
            return None
            
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


        is_paused_or_replaying = plotting_paused or (state == "replay_active") # Simpler check for replay state
        if is_paused_or_replaying and self.scrub_time_widget_heatmap.isVisible(): # Check visibility of the container
            current_slider_int_value = self.time_slider_heatmap.value()
            current_slider_time_sec = float(current_slider_int_value) / self.SLIDER_FLOAT_PRECISION_FACTOR_HEATMAP
            logger.info(f"Heatmap snapshot: Using data from slider time {current_slider_time_sec:.2f}s (raw slider: {current_slider_int_value}).")
            # Ensure render_heatmap_for_time is called *before* drawing to the save_pixmap
            self.render_heatmap_for_time(current_slider_time_sec) 
        else:
            logger.info("Heatmap snapshot: Using live/latest data (not from slider).")
            # For live data, self.pressure_values, self.center_of_pressure, self.cop_trail
            # should already reflect the latest state from update_component.


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



    def _update_colormap(self, cmap_name):
        try:
            self.cmap = matplotlib.colormaps[cmap_name]
            self.current_cmap_name = cmap_name
            logger.info(f"Heatmap colormap changed to: {self.current_cmap_name}")
             
        except KeyError:
             logger.error(f"Heatmap: Invalid colormap selected in dropdown: {cmap_name}. Keeping previous.")
             
             self.cmap_combobox.blockSignals(True)
             self.cmap_combobox.setCurrentText(self.current_cmap_name)
             self.cmap_combobox.blockSignals(False)

    def on_time_slider_heatmap(self, slider_value_int: int):
        global plotting_paused
        if not plotting_paused: # If we are entering scrub mode via slider interaction
            plotting_paused = True 
            main_window = self.window()
            if isinstance(main_window, MainWindow):
                if main_window.pause_resume_button.text() != "Resume Plotting":
                     main_window.pause_resume_button.setText("Resume Plotting")
        
        time_sec_float = float(slider_value_int) / self.SLIDER_FLOAT_PRECISION_FACTOR_HEATMAP

        if hasattr(self, 'current_replay_time_label_heatmap'):
            self.current_replay_time_label_heatmap.setText(f"{time_sec_float:.2f}s")

        self.render_heatmap_for_time(time_sec_float)



    def render_heatmap_for_time(self, time_sec: float):
        if not self.heatmap_qimage: return

        # --- 1. Calculate and set self.pressure_values for heatmap coloration at time_sec ---
        pressures_for_heatmap_at_time_sec = np.zeros(self.NUM_SENSORS, dtype=np.float32)
        data_found_for_heatmap = False
        for i, key in enumerate(HEATMAP_KEYS):
            val_at_time = get_value_at_time(key, time_sec, self.data_buffers_ref)
            if val_at_time is not None:
                initial_pressure = val_at_time
                rescaled_val = 0.0
                if INITIAL_PRESSURE_SENSITIVITY > 1e-6:
                    rescaled_val = initial_pressure * (self.current_pressure_sensitivity / INITIAL_PRESSURE_SENSITIVITY)
                pressures_for_heatmap_at_time_sec[i] = max(0.0, rescaled_val)
                data_found_for_heatmap = True
        
        self.pressure_values = pressures_for_heatmap_at_time_sec # Used by _calculate_pressure_fast for heatmap colors

        if not data_found_for_heatmap and time_sec > 0.01: # Avoid log spam for t=0 if no data
             logger.debug(f"Heatmap render_for_time({time_sec:.2f}s): No data for heatmap colors.")
        
        # --- 2. Reconstruct CoP trail leading up to time_sec ---
        self.cop_trail.clear()
        self.center_of_pressure = None # Will be set to the last point of the trail

        all_heatmap_keys_timestamps = []
        for key in HEATMAP_KEYS:
            if key in self.data_buffers_ref and self.data_buffers_ref[key]:
                all_heatmap_keys_timestamps.extend([item[0] for item in self.data_buffers_ref[key]])
        
        if all_heatmap_keys_timestamps:
            unique_sorted_times = sorted(list(set(all_heatmap_keys_timestamps)))
            
            # Find the index of the actual data point at or just before time_sec
            idx_current_render_time = bisect.bisect_right(unique_sorted_times, time_sec) - 1

            if idx_current_render_time >= 0:
                # Determine the range of timestamps for the trail
                trail_end_idx = idx_current_render_time
                trail_start_idx = max(0, trail_end_idx - self.COP_TRAIL_MAX_LEN + 1)
                timestamps_for_trail = unique_sorted_times[trail_start_idx : trail_end_idx + 1]

                temp_trail_cops = []
                for t_hist in timestamps_for_trail:
                    historical_pressures_at_t_hist = np.zeros(self.NUM_SENSORS, dtype=np.float32)
                    data_found_for_trail_point = False
                    for i, key_trail in enumerate(HEATMAP_KEYS):
                        val_at_t_hist = get_value_at_time(key_trail, t_hist, self.data_buffers_ref)
                        if val_at_t_hist is not None:
                            initial_pressure = val_at_t_hist
                            rescaled_val = 0.0
                            if INITIAL_PRESSURE_SENSITIVITY > 1e-6:
                                 rescaled_val = initial_pressure * (self.current_pressure_sensitivity / INITIAL_PRESSURE_SENSITIVITY)
                            historical_pressures_at_t_hist[i] = max(0.0, rescaled_val)
                            data_found_for_trail_point = True
                    
                    if data_found_for_trail_point:
                        cop_at_t_hist = self._calculate_center_of_pressure(pressures_array=historical_pressures_at_t_hist)
                        if cop_at_t_hist:
                            temp_trail_cops.append(cop_at_t_hist)
                
                if temp_trail_cops:
                    self.cop_trail.extend(temp_trail_cops)
                    self.center_of_pressure = self.cop_trail[-1] # Main CoP dot is the last in the trail
            else: # time_sec is before any data
                self.pressure_values.fill(0.0) 
        else: # No timestamps at all for heatmap keys
            self.pressure_values.fill(0.0)

        # --- 3. Render heatmap and CoP ---
        calculated_pressures_for_heatmap = self._calculate_pressure_fast() # Uses self.pressure_values
        self._render_heatmap_to_buffer(calculated_pressures_for_heatmap)
        self._update_display_pixmap()



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
        is_csv_data_missing = False
        if state == "replay_active" and self.data_type_to_monitor not in self.data_buffers_ref:
            is_csv_data_missing = True
            self.value_label.setText("(CSV Data Missing)")

        # Only show '--' if not paused, not missing UUID, not missing CSV data, and value not found
        if not value_found and not is_uuid_missing and not is_csv_data_missing and not plotting_paused:
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

    def handle_missing_replay_data(self, missing_data_types_for_component: Set[str]):
        # For SingleValueDisplay, if its specific data_type is in the missing set for replay
        if self.data_type_to_monitor in missing_data_types_for_component:
            self.value_label.setText("(CSV Data Missing)")
        else:
            # If the overlay was visible and was for CSV data, but our type is now loaded, update
            if self.uuid_missing_overlay and self.uuid_missing_overlay.isVisible() and "CSV" in self.uuid_missing_overlay.text():
                 current_time = 0.0
                 if start_time: current_time = (datetime.datetime.now() - start_time).total_seconds()
                 self.update_component(current_time, False) # Attempt to update with current data
        # The generic overlay is handled by super if we call it:
        # super().handle_missing_replay_data(missing_data_types_for_component)
        # However, for this simple component, updating its own label might be cleaner.

# --- Nyquist Plot Component ---
class NyquistPlotComponent(BaseGuiComponent):
    # Constants
    SLIDER_FLOAT_PRECISION_FACTOR_NYQUIST = 100 # For 0.01s resolution, adjust as needed


    TRAIL_MAX_LEN = 50
    TRAIL_COLOR = QColor(243, 100, 248, 255)  # Bright pink
    MAIN_POINT_COLOR = QColor(0, 0, 0, 255)    # Black
    TRAIL_POINT_RADIUS = 3.0
    MAIN_POINT_RADIUS = 5.0
    SPLINE_SAMPLES_PER_SEGMENT = 10
    SPLINE_LINE_WIDTH = 3
    SNAPSHOT_DPI = 200

    def __init__(self, config: Dict[str, Any], data_buffers_ref: Dict[str, List[Tuple[float, float]]], device_config_ref: DeviceConfig, parent: Optional[QWidget] = None):
        super().__init__(config, data_buffers_ref, device_config_ref, parent)

        self._required_data_types: Set[str] = {"real_part_kohm", "imag_part_kohm"}
        self.missing_relevant_uuids: Set[str] = set()

        self.trail_data: deque[Tuple[float, float]] = deque(maxlen=self.TRAIL_MAX_LEN)

        self.snapshot_dir = self.config.get('snapshot_dir', "nyquist_snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.snapshot_executor = ThreadPoolExecutor(max_workers=1)

        self.plot_widget = pg.PlotWidget()
        self.plot_item: pg.PlotItem = self.plot_widget.getPlotItem()
        self._setup_plot_appearance()

        self.spline_items: List[pg.PlotDataItem] = []
        for _ in range(self.TRAIL_MAX_LEN - 1):
            item = pg.PlotDataItem(pen=None)
            self.plot_item.addItem(item)
            self.spline_items.append(item)

        self.trail_scatter_item = pg.ScatterPlotItem()
        self.current_scatter_item = pg.ScatterPlotItem()
        self.plot_item.addItem(self.trail_scatter_item)
        self.plot_item.addItem(self.current_scatter_item)

        component_layout = QVBoxLayout(self)
        component_layout.setContentsMargins(0, 0, 0, 0)
        component_layout.addWidget(self.plot_widget)


        self.time_slider_nyquist = QSlider(Qt.Orientation.Horizontal)

        self.current_replay_time_label_nyquist = QLabel("0.00s")
        self.current_replay_time_label_nyquist.setFixedWidth(50) # Adjust width as needed
        self.current_replay_time_label_nyquist.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.time_slider_nyquist.setRange(0,0)
        self.time_slider_nyquist.valueChanged.connect(self.on_time_slider_nyquist)

        # Create a container widget for the scrub time controls
        self.scrub_time_widget_nyquist = QWidget()
        time_slider_layout_nq = QHBoxLayout(self.scrub_time_widget_nyquist)
        time_slider_layout_nq.setContentsMargins(0,0,0,0) # Minimize extra space
        time_slider_layout_nq.addWidget(QLabel("Scrub Time:"))
        time_slider_layout_nq.addWidget(self.time_slider_nyquist)

        time_slider_layout_nq.addWidget(self.current_replay_time_label_nyquist)
        
        # Add the container widget directly to the main component layout
        component_layout.addWidget(self.scrub_time_widget_nyquist)
        self.scrub_time_widget_nyquist.setVisible(False) # Initially hidden



        self.snapshot_button = QPushButton("Take Snapshot")
        self.snapshot_button.clicked.connect(self._trigger_snapshot_creation)
        component_layout.addWidget(self.snapshot_button)

        self.setLayout(component_layout)
        self.clear_component()

    def _setup_plot_appearance(self):
        plot_height = self.config.get('plot_height')
        plot_width = self.config.get('plot_width')
        if plot_height is not None: self.setFixedHeight(plot_height)
        if plot_width is not None: self.setFixedWidth(plot_width)
        
        if plot_height is not None and plot_width is None:
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        elif plot_width is not None and plot_height is None:
            self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        elif plot_width is None and plot_height is None:
             self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        else: 
             self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.plot_item.setTitle(self.config.get('title', 'Nyquist Plot'), size='10pt')
        self.plot_item.setLabel('bottom', self.config.get('xlabel', "Real Z' [kOhm]"))
        self.plot_item.setLabel('left', self.config.get('ylabel', "-Imag Z'' [kOhm]"))
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)
        self.plot_item.getViewBox().setAspectLocked(True)
        self.plot_item.getViewBox().setDefaultPadding(0.05)

    def get_required_data_types(self) -> Set[str]:
        return self._required_data_types

    def get_log_filename_suffix(self) -> str:
        if self.is_loggable:
            title = self.config.get('title', 'NyquistPlot')
            safe_suffix = re.sub(r'[^\w\-]+', '_', title).strip('_')
            return f"nyquist_{safe_suffix}" if safe_suffix else f"nyquist_{id(self)}"
        return ""

    def update_component(self, current_relative_time: float, is_flowing: bool):


        if (plotting_paused and is_flowing) or self.missing_relevant_uuids: # plotting_paused global, is_flowing arg
            return

        if not is_flowing: # Replay mode or "Flowing Mode" checkbox unchecked
            self.scrub_time_widget_nyquist.setVisible(True)
            
            actual_min_data_time_nq = float('inf')
            actual_max_data_time_nq = 0.0
            found_any_data_nq = False

            for key_nq_slider in self._required_data_types: 
                if key_nq_slider in self.data_buffers_ref and self.data_buffers_ref[key_nq_slider]:
                    series_times_nq = [item[0] for item in self.data_buffers_ref[key_nq_slider]]
                    if series_times_nq:
                        actual_min_data_time_nq = min(actual_min_data_time_nq, series_times_nq[0])
                        actual_max_data_time_nq = max(actual_max_data_time_nq, series_times_nq[-1])
                        found_any_data_nq = True
            
            if not found_any_data_nq:
                actual_min_data_time_nq = 0.0
                actual_max_data_time_nq = 1.0 
            
            if actual_max_data_time_nq <= actual_min_data_time_nq:
                actual_max_data_time_nq = actual_min_data_time_nq + 0.1

            slider_min_int = int(math.floor(actual_min_data_time_nq * self.SLIDER_FLOAT_PRECISION_FACTOR_NYQUIST))
            slider_max_int = int(math.ceil(actual_max_data_time_nq * self.SLIDER_FLOAT_PRECISION_FACTOR_NYQUIST))

            if slider_max_int <= slider_min_int:
                slider_max_int = slider_min_int + 1

            self.time_slider_nyquist.blockSignals(True)
            if self.time_slider_nyquist.minimum() != slider_min_int or \
               self.time_slider_nyquist.maximum() != slider_max_int:
                current_val = self.time_slider_nyquist.value()
                self.time_slider_nyquist.setRange(slider_min_int, slider_max_int)
                if slider_min_int <= current_val <= slider_max_int:
                    self.time_slider_nyquist.setValue(current_val)
                else:
                    self.time_slider_nyquist.setValue(slider_min_int)
            self.time_slider_nyquist.blockSignals(False)

            
            # Update the replay time label to reflect the current slider value
            current_slider_val_for_label_nq = self.time_slider_nyquist.value()
            time_sec_for_label_nq = float(current_slider_val_for_label_nq) / self.SLIDER_FLOAT_PRECISION_FACTOR_NYQUIST
            if hasattr(self, 'current_replay_time_label_nyquist'):
                self.current_replay_time_label_nyquist.setText(f"{time_sec_for_label_nq:.2f}s")
            
            if plotting_paused: # This means we are in slider mode
                 self.render_nyquist_for_time(float(self.time_slider_nyquist.value()) / self.SLIDER_FLOAT_PRECISION_FACTOR_NYQUIST)

        else: # is_flowing is True (Live mode)
            self.scrub_time_widget_nyquist.setVisible(False)
            # Live update logic
            real_key, imag_key = "real_part_kohm", "imag_part_kohm"
            new_data_point = None
            if real_key in self.data_buffers_ref and self.data_buffers_ref[real_key] and \
               imag_key in self.data_buffers_ref and self.data_buffers_ref[imag_key]:
                try:
                    latest_real_val = self.data_buffers_ref[real_key][-1][1]
                    latest_imag_val = self.data_buffers_ref[imag_key][-1][1]
                    new_data_point = (float(latest_real_val), -float(latest_imag_val))
                except (IndexError, ValueError) as e: logger.debug(f"NyquistPlot: Error fetching latest data: {e}")
                except Exception as e: logger.warning(f"NyquistPlot: Unexpected error processing data: {e}")

            if new_data_point:
                self.trail_data.append(new_data_point)
                self._refresh_plot_graphics()


    def _refresh_plot_graphics(self):
        points = list(self.trail_data)
        num_points = len(points)

        if not num_points:
            self.trail_scatter_item.clear()
            self.current_scatter_item.clear()
            for spline in self.spline_items:
                spline.clear()
            return

        if num_points > 1:
            alphas_for_trail = (np.linspace(0.1, 1.0, num_points) ** 1.5) * 255
        else:
            alphas_for_trail = np.array([255.0])
        alphas_for_trail = np.clip(alphas_for_trail, 0, 255).astype(int)

        base_rgb = (self.TRAIL_COLOR.red(), self.TRAIL_COLOR.green(), self.TRAIL_COLOR.blue())

        trail_plot_data = []
        for i in range(num_points):
            pt = points[i]; alpha = alphas_for_trail[i]
            trail_plot_data.append({'pos': pt, 'size': self.TRAIL_POINT_RADIUS * 2,
                                    'brush': pg.mkBrush(QColor(*base_rgb, alpha)), 'pen': None})
        self.trail_scatter_item.setData(trail_plot_data)

        self.current_scatter_item.setData([{'pos': points[-1], 'size': self.MAIN_POINT_RADIUS * 2,
                                           'brush': pg.mkBrush(self.MAIN_POINT_COLOR),
                                           'pen': pg.mkPen(QColor(255,255,255,150), width=0.5)}])
        for i in range(len(self.spline_items)):
            if i < num_points - 1:
                p1 = points[i]; p2 = points[i+1]
                p0 = points[i-1] if i > 0 else p1
                p3 = points[i+2] if i < num_points - 2 else p2
                segment_coords = self._catmull_rom_segment_static(p0, p1, p2, p3, self.SPLINE_SAMPLES_PER_SEGMENT)
                xs = [c[0] for c in segment_coords]; ys = [c[1] for c in segment_coords]
                segment_alpha = alphas_for_trail[i]
                pen = pg.mkPen(QColor(*base_rgb, segment_alpha), width=self.SPLINE_LINE_WIDTH)
                self.spline_items[i].setData(xs, ys, pen=pen)
            else:
                self.spline_items[i].clear()
        
        if not self.missing_relevant_uuids and num_points > 0:
            self.plot_item.enableAutoRange(axis='xy', enable=True)

    @staticmethod
    def _catmull_rom_point_static(t: float, p0_tuple, p1_tuple, p2_tuple, p3_tuple):
        t2 = t * t; t3 = t2 * t
        p0x, p0y = p0_tuple; p1x, p1y = p1_tuple; p2x, p2y = p2_tuple; p3x, p3y = p3_tuple
        x = 0.5 * ((2 * p1x) + (-p0x + p2x) * t + (2 * p0x - 5 * p1x + 4 * p2x - p3x) * t2 + (-p0x + 3 * p1x - 3 * p2x + p3x) * t3)
        y = 0.5 * ((2 * p1y) + (-p0y + p2y) * t + (2 * p0y - 5 * p1y + 4 * p2y - p3y) * t2 + (-p0y + 3 * p1y - 3 * p2y + p3y) * t3)
        return (x, y)

    @classmethod
    def _catmull_rom_segment_static(cls, p0, p1, p2, p3, samples: int):
        return [cls._catmull_rom_point_static(i / samples, p0, p1, p2, p3) for i in range(samples + 1)]

    def clear_component(self):
        self.trail_data.clear()
        self._refresh_plot_graphics()

        super().handle_missing_uuids(set())
        super().handle_missing_replay_data(set()) # Ensure replay overlay is also cleared        
        self.missing_relevant_uuids.clear()
        self.plot_item.setXRange(-1, 1, padding=0.1)
        self.plot_item.setYRange(-1, 1, padding=0.1)

        self.plot_item.enableAutoRange(axis='xy', enable=True)
        self.snapshot_button.setEnabled(True)
        self.snapshot_button.setText("Take Snapshot")

        # --- Reset time slider and hide its container ---
        if hasattr(self, 'time_slider_nyquist'):
            self.time_slider_nyquist.blockSignals(True)
            self.time_slider_nyquist.setRange(0, 0) # Reset range
            self.time_slider_nyquist.setValue(0)    # Reset value
            self.time_slider_nyquist.blockSignals(False)
            # The slider's visibility is controlled by its parent scrub_time_widget_nyquist

        if hasattr(self, 'scrub_time_widget_nyquist'):
            self.scrub_time_widget_nyquist.setVisible(False)

        if hasattr(self, 'current_replay_time_label_nyquist'):
            self.current_replay_time_label_nyquist.setText("0.00s")


    def handle_missing_uuids(self, missing_uuids_for_component: Set[str]):
        super().handle_missing_uuids(missing_uuids_for_component)
        self.missing_relevant_uuids = missing_uuids_for_component # Keep track of missing UUIDs
        self._update_controls_based_on_data_status()


    def _update_controls_based_on_data_status(self):
        is_overlay_active = self.uuid_missing_overlay and self.uuid_missing_overlay.isVisible()
        if hasattr(self, 'snapshot_button'):
            self.snapshot_button.setEnabled(not is_overlay_active)
        
        if is_overlay_active:
            self.trail_data.clear()
            self._refresh_plot_graphics() # Clear visual plot
            self.plot_item.enableAutoRange(axis='xy', enable=False)
            self.plot_item.setXRange(-1, 1, padding=0.1)
            self.plot_item.setYRange(-1, 1, padding=0.1)
        else:
            if self.trail_data: # Only enable autorange if there's data and no overlay
                 self.plot_item.enableAutoRange(axis='xy', enable=True)


    def handle_missing_replay_data(self, missing_data_types_for_component: Set[str]):
        super().handle_missing_replay_data(missing_data_types_for_component)
        self._update_controls_based_on_data_status()


    def _trigger_snapshot_creation(self):
        if not self.snapshot_button.isEnabled() or not self.trail_data:
            if not self.trail_data: logger.info("NyquistPlot: No data for snapshot.")
            return
        

        is_paused_or_replaying_nq = plotting_paused or (state == "replay_active")
        if is_paused_or_replaying_nq and self.scrub_time_widget_nyquist.isVisible():
            current_slider_int_value_nq = self.time_slider_nyquist.value()
            current_slider_time_nq_sec = float(current_slider_int_value_nq) / self.SLIDER_FLOAT_PRECISION_FACTOR_NYQUIST
            logger.info(f"Nyquist snapshot: Using data from slider time {current_slider_time_nq_sec:.2f}s (raw slider: {current_slider_int_value_nq}).")
            # Ensure render_nyquist_for_time updates self.trail_data before PDF generation
            self.render_nyquist_for_time(current_slider_time_nq_sec)
        else:
             logger.info("Nyquist snapshot: Using live/latest trail data (not from slider).")
             # For live data, self.trail_data should be current from update_component.


        self.snapshot_button.setEnabled(False)
        self.snapshot_button.setText("Saving Snapshot...")
        points_to_snapshot = list(self.trail_data)
        current_view_range = self.plot_item.getViewBox().viewRange()
        num_pts = len(points_to_snapshot)
        alphas_snapshot = (np.linspace(0.1, 1.0, num_pts) ** 1.5) if num_pts > 1 else (np.array([1.0]) if num_pts ==1 else np.array([]))
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"nyquist_snapshot_{timestamp_str}.pdf"
        filepath = os.path.join(self.snapshot_dir, filename)
        loop = asyncio.get_running_loop() # Assumes qasync loop is running
        future = loop.run_in_executor(
            self.snapshot_executor, self._build_snapshot_pdf_sync,
            points_to_snapshot, current_view_range, alphas_snapshot, filepath,
            self.config.get('title', 'Nyquist Plot'),
            self.config.get('xlabel', "Real Z' [kOhm]"),
            self.config.get('ylabel', "-Imag Z'' [kOhm]")
        )
        future.add_done_callback(self._snapshot_done_callback)

    def _snapshot_done_callback(self, future_result):
        self.snapshot_button.setEnabled(True)
        self.snapshot_button.setText("Take Snapshot")
        try:
            result_path = future_result.result()
            if result_path: logger.info(f"NyquistPlot: Snapshot saved to {result_path}")
        except Exception as e:
            logger.error(f"NyquistPlot: Error retrieving snapshot result: {e}", exc_info=True)




    def on_time_slider_nyquist(self, slider_value_int: int):
        global plotting_paused
        if not plotting_paused: # If we are entering scrub mode via slider interaction
            plotting_paused = True
            main_window = self.window()
            if isinstance(main_window, MainWindow):
                if main_window.pause_resume_button.text() != "Resume Plotting":
                    main_window.pause_resume_button.setText("Resume Plotting")
        
        time_sec_float = float(slider_value_int) / self.SLIDER_FLOAT_PRECISION_FACTOR_NYQUIST

        if hasattr(self, 'current_replay_time_label_nyquist'):
            self.current_replay_time_label_nyquist.setText(f"{time_sec_float:.2f}s")

        self.render_nyquist_for_time(time_sec_float)

    def render_nyquist_for_time(self, time_sec: float):
        self.trail_data.clear()

        real_key = "real_part_kohm"
        imag_key = "imag_part_kohm"

        # Check if the required data types are even in data_buffers
        if real_key not in self.data_buffers_ref or imag_key not in self.data_buffers_ref or \
           not self.data_buffers_ref[real_key] or not self.data_buffers_ref[imag_key]:
            logger.debug(f"Nyquist render_for_time({time_sec:.2f}s): Missing or empty buffers for {real_key} or {imag_key}.")
            self._refresh_plot_graphics() # Refresh to show an empty plot
            return

        # Combine all timestamps from real and imag parts to find relevant historical points
        all_nyquist_timestamps = []
        all_nyquist_timestamps.extend([item[0] for item in self.data_buffers_ref[real_key]])
        all_nyquist_timestamps.extend([item[0] for item in self.data_buffers_ref[imag_key]])
        
        if not all_nyquist_timestamps:
            self._refresh_plot_graphics()
            return

        unique_sorted_times = sorted(list(set(all_nyquist_timestamps)))
        
        # Find the index of the actual data point at or just before time_sec
        idx_current_render_time = bisect.bisect_right(unique_sorted_times, time_sec) -1

        if idx_current_render_time < 0: # time_sec is before any data
            self._refresh_plot_graphics() # Refresh to show an empty plot
            return

        # Determine the range of timestamps for the trail
        trail_end_idx = idx_current_render_time
        trail_start_idx = max(0, trail_end_idx - self.TRAIL_MAX_LEN + 1)
        timestamps_for_trail = unique_sorted_times[trail_start_idx : trail_end_idx + 1]
        
        temp_trail_points = []
        for t_hist in timestamps_for_trail:
            real_val_hist = get_value_at_time(real_key, t_hist, self.data_buffers_ref)
            imag_val_hist = get_value_at_time(imag_key, t_hist, self.data_buffers_ref)

            if real_val_hist is not None and imag_val_hist is not None:
                temp_trail_points.append((float(real_val_hist), -float(imag_val_hist)))
        
        if temp_trail_points:
            self.trail_data.extend(temp_trail_points)
        
        self._refresh_plot_graphics()


    @classmethod
    def _build_snapshot_pdf_sync(cls, points, view_range, alphas, path, title_str, xlabel_str, ylabel_str):
        try:
            (x_min, x_max), (y_min, y_max) = view_range
            base_color_tuple = (cls.TRAIL_COLOR.redF(), cls.TRAIL_COLOR.greenF(), cls.TRAIL_COLOR.blueF())
            
            # Use scienceplots style if available, otherwise default
            plot_style = 'science' if 'science' in plt.style.available else 'default'

            with plt.style.context([plot_style]):
                if plot_style == 'science':
                    plt.rcParams.update({'text.usetex': False, 'figure.figsize': [6, 5],
                                         'legend.fontsize': 9, 'axes.labelsize': 10, 
                                         'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.titlesize': 11})
                else: # Basic defaults if scienceplots not used
                    plt.rcParams.update({'figure.figsize': [6, 5], 'axes.grid': True})

                fig, ax = plt.subplots(figsize=(6, 5), dpi=cls.SNAPSHOT_DPI)
                num_snapshot_points = len(points)
                if num_snapshot_points > 0:
                    for i in range(num_snapshot_points - 1):
                        p1 = points[i]; p2 = points[i+1]
                        p0 = points[i-1] if i > 0 else p1
                        p3 = points[i+2] if i < num_snapshot_points - 2 else p2
                        seg_coords = cls._catmull_rom_segment_static(p0, p1, p2, p3, cls.SPLINE_SAMPLES_PER_SEGMENT)
                        xs, ys = zip(*seg_coords)
                        ax.plot(xs, ys, color=base_color_tuple, alpha=float(alphas[i]), linewidth=cls.SPLINE_LINE_WIDTH / 1.5)
                    ax.scatter([points[-1][0]], [points[-1][1]], color=base_color_tuple,
                               s=cls.MAIN_POINT_RADIUS * 10, edgecolor="black", linewidth=0.5, zorder=5)
                
                ax.set_xlabel(xlabel_str); ax.set_ylabel(ylabel_str)
                ax.set_title(title_str + " (Snapshot)"); ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_aspect("equal", adjustable="box"); ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
                fig.tight_layout(pad=0.5); plt.savefig(path, format="pdf", bbox_inches="tight", dpi=cls.SNAPSHOT_DPI)
                plt.close(fig)
            return path
        except Exception as e:
            logger.error(f"NyquistPlot: Failed to build snapshot PDF '{path}': {e}", exc_info=True)
            return None

    def __del__(self):
        if hasattr(self, 'snapshot_executor') and self.snapshot_executor:
            self.snapshot_executor.shutdown(wait=False)


# --- IMU Visualizer Component ---
class IMUVisualizerComponent(BaseGuiComponent):

    SLIDER_FLOAT_PRECISION_FACTOR_IMU = 100 # For 0.01s resolution


    # STL and Box rendering constants
    _DEFAULT_BOX_SIZE = (1.5, 3, 0.5)
    _DEFAULT_BOX_EDGE_COLOR = (0.5, 0.5, 0.5, 1.0)
    _DEFAULT_BOX_FACE_COLORS = np.array([ # RGBA (0-1) for 6 faces
        (0.6, 0.6, 0.6, 1.0), (0.9, 0.9, 0.9, 1.0), (0.8, 0.2, 0.2, 1.0),
        (0.8, 0.5, 0.2, 1.0), (0.2, 0.8, 0.2, 1.0), (0.2, 0.2, 0.8, 1.0)
    ], dtype=np.float32)
    _STL_DEFAULT_FACE_COLOR = (0.7, 0.7, 0.7, 1.0) # Default if STL has no color and none configured



    # Axis and Grid constants (can remain as they are if not needing changes)
    AXIS_LENGTH = 3.0
    AXIS_RADIUS = 0.03
    ARROW_RADIUS = 0.08
    ARROW_LENGTH = 0.3
    AXIS_COLORS = {'x': (1, 0, 0, 1), 'y': (0, 1, 0, 1), 'z': (0, 0, 1, 1)} # R, G, B, Alpha (0-1)
    GLVIEW_BACKGROUND_COLOR = QColor(200, 200, 200)
    GRID_COLOR = QColor(0, 0, 0, 100) # R, G, B, Alpha (0-255)
    GRID_SCALE = 5
    SNAPSHOT_DIR_DEFAULT = "IMU_Snapshots"


    def __init__(self, config: Dict[str, Any], data_buffers_ref: Dict[str, List[Tuple[float, float]]], device_config_ref: DeviceConfig, parent: Optional[QWidget] = None):
        super().__init__(config, data_buffers_ref, device_config_ref, parent)
        self._required_data_types = {"quat_w", "quat_x", "quat_y", "quat_z"}
        self.current_quaternion = QQuaternion(1, 0, 0, 0)
        self.baseline_quaternion = QQuaternion(1, 0, 0, 0)
        
        self.snapshot_dir = self.config.get('snapshot_dir', self.SNAPSHOT_DIR_DEFAULT)
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # --- STL/Mesh Configuration ---
        self.stl_filename = self.config.get("stl_filename", None)
        self.mirror_x = self.config.get("mirror_x", False)
        self.mirror_y = self.config.get("mirror_y", False)
        self.mirror_z = self.config.get("mirror_z", False)
        self.stl_scale = self.config.get("stl_scale", 1.0)

        self.stl_config_color = self.config.get("stl_color", None) # User-defined color for the whole STL
        self.use_stl_attributes_for_color = self.config.get("use_stl_attributes_for_color", False) # << NEW
        self.stl_draw_edges = self.config.get("stl_draw_edges", False)
        self.stl_edge_color = self.config.get("stl_edge_color", (0.2, 0.2, 0.2, 1.0))

        self.mesh_data_item: Optional[gl.MeshData] = None # To store loaded MeshData
        self.object_mesh: Optional[gl.GLMeshItem] = None  # The GLMeshItem for STL or box

        self._setup_internal_ui()
        self._load_and_prepare_mesh_data() # Load mesh data before setting up scene elements
        self._setup_scene_elements()
        self.clear_component()



    def _setup_internal_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0,0,0,0)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(self.GLVIEW_BACKGROUND_COLOR)
        self.view.setCameraPosition(distance=10, elevation=20, azimuth=45)
        main_layout.addWidget(self.view, 1) # View takes most space

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(5,5,5,5)

        self.orientation_status_label = QLabel("Yaw: 0.0° Pitch: 0.0° Roll: 0.0°")
        self.orientation_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font_orientation = self.orientation_status_label.font(); font_orientation.setPointSize(11)
        self.orientation_status_label.setFont(font_orientation)
        controls_layout.addWidget(self.orientation_status_label)

        self.time_slider_imu = QSlider(Qt.Orientation.Horizontal)

        self.current_replay_time_label_imu = QLabel("0.00s")
        self.current_replay_time_label_imu.setFixedWidth(50) # Adjust width as needed
        self.current_replay_time_label_imu.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.time_slider_imu.setRange(0,0)
        self.time_slider_imu.valueChanged.connect(self.on_time_slider_imu)
        
        # Create a container widget for the scrub time controls
        self.scrub_time_widget_imu = QWidget()
        time_slider_layout_imu = QHBoxLayout(self.scrub_time_widget_imu)
        time_slider_layout_imu.setContentsMargins(0,0,0,0) # Minimize extra space
        time_slider_layout_imu.addWidget(QLabel("Scrub Time:"))

        time_slider_layout_imu.addWidget(self.time_slider_imu)


        time_slider_layout_imu.addWidget(self.current_replay_time_label_imu)

        controls_layout.addWidget(self.scrub_time_widget_imu)
        self.scrub_time_widget_imu.setVisible(False) # Initially hidden



        buttons_layout = QHBoxLayout()
        self.snapshot_button = QPushButton("Take Snapshot")
        self.snapshot_button.clicked.connect(self._take_snapshot_action)
        buttons_layout.addWidget(self.snapshot_button)

        self.reset_button = QPushButton("Reset Orientation")
        self.reset_button.clicked.connect(self._reset_orientation_action)
        buttons_layout.addWidget(self.reset_button)
        controls_layout.addLayout(buttons_layout)
        
        controls_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        main_layout.addWidget(controls_widget)
        self.setLayout(main_layout)

    def _load_and_prepare_mesh_data(self):
        """
        Loads mesh data from an STL file if specified, or creates default box data.
        Applies configured scaling and mirroring to STL data.
        Sets self.mesh_data_item with the pyqtgraph.opengl.MeshData.
        """
        logger.info("IMUVisualizer: Preparing mesh data...")
        use_stl = False
        if self.stl_filename:
            if not os.path.exists(self.stl_filename):
                logger.error(f"IMUVisualizer: STL file '{self.stl_filename}' not found. Falling back to default box.")
            else:
                try:
                    from stl import mesh as stl_mesh # Import here to catch ImportError if not installed
                    
                    stl_mesh_obj = stl_mesh.Mesh.from_file(self.stl_filename)
                    num_faces = stl_mesh_obj.vectors.shape[0]
                    if num_faces == 0:
                        logger.warning(f"IMUVisualizer: STL file '{self.stl_filename}' contains no faces. Falling back to default box.")
                    else:
                        # Each face has 3 vertices, each vertex has 3 coordinates
                        # stl_mesh_obj.vectors is (num_faces, 3_vertices_per_face, 3_coords_per_vertex)
                        # We want a flat list of vertices for each triangle for per-face coloring (N*3, 3)
                        vertices = stl_mesh_obj.vectors.reshape(num_faces * 3, 3).astype(np.float32)
                        
                        # Faces will simply index these flattened vertices: [[0,1,2], [3,4,5], ...]
                        faces = np.arange(num_faces * 3, dtype=np.uint32).reshape(num_faces, 3)

                        # Apply scaling FIRST
                        if self.stl_scale != 1.0:
                            vertices *= float(self.stl_scale) # Apply scale directly to the vertices array
                            logger.info(f"IMUVisualizer: STL vertices scaled by {self.stl_scale}")
                        
                        # Then apply mirroring to the (potentially already scaled) vertices
                        num_mirrors = 0
                        if self.mirror_x: vertices[:, 0] *= -1; num_mirrors += 1
                        if self.mirror_y: vertices[:, 1] *= -1; num_mirrors += 1
                        if self.mirror_z: vertices[:, 2] *= -1; num_mirrors += 1
                        
                        if num_mirrors % 2 != 0: # Odd number of mirrors, reverse face winding for correct normals
                            faces = faces[:, [0, 2, 1]]
                        
                        # Handle face colors
                        face_colors_array = None
                        if self.use_stl_attributes_for_color and hasattr(stl_mesh_obj, 'attr') and stl_mesh_obj.attr is not None and stl_mesh_obj.attr.shape[0] == num_faces:
                            parsed_stl_colors = []
                            colors_from_attr = 0
                            for i in range(num_faces):
                                attr_val = stl_mesh_obj.attr[i, 0]
                                if attr_val != 0:
                                    r = ((attr_val & 0xF800) >> 11) / 31.0
                                    g = ((attr_val & 0x07E0) >> 5)  / 63.0
                                    b = (attr_val & 0x001F)       / 31.0
                                    parsed_stl_colors.append((r, g, b, 1.0))
                                    colors_from_attr +=1
                                else: 
                                    parsed_stl_colors.append(self.stl_config_color if self.stl_config_color else self._STL_DEFAULT_FACE_COLOR)
                            if colors_from_attr > 0:
                                face_colors_array = np.array(parsed_stl_colors, dtype=np.float32)
                                logger.info(f"IMUVisualizer: Applied colors from STL attributes for {colors_from_attr}/{num_faces} faces.")
                        
                        if face_colors_array is None: 
                            if self.stl_config_color:
                                logger.info(f"IMUVisualizer: Using configured 'stl_color' for all STL faces.")
                                color_to_tile = np.array(self.stl_config_color, dtype=np.float32)
                            else:
                                logger.info(f"IMUVisualizer: Using default color for all STL faces.")
                                color_to_tile = np.array(self._STL_DEFAULT_FACE_COLOR, dtype=np.float32)
                            face_colors_array = np.tile(color_to_tile, (num_faces, 1))
                        
                        # Create MeshData with the processed (scaled and mirrored) vertices
                        self.mesh_data_item = gl.MeshData(vertexes=vertices, faces=faces, faceColors=face_colors_array)
                        logger.info(f"IMUVisualizer: Successfully loaded and prepared STL '{self.stl_filename}'.")
                        use_stl = True

                except ImportError:
                    logger.error("IMUVisualizer: 'numpy-stl' library is not installed. Please install it: pip install numpy-stl. Falling back to default box.")
                except Exception as e:
                    logger.error(f"IMUVisualizer: Failed to load or process STL file '{self.stl_filename}': {e}. Falling back to default box.", exc_info=True)
        
        if not use_stl: # Fallback to default box
            logger.info("IMUVisualizer: Using default box model.")
            width, height, depth = self._DEFAULT_BOX_SIZE
            w2, h2, d2 = width / 2, height / 2, depth / 2
            
            # Apply configured scale to the default box as well
            scaled_w2, scaled_h2, scaled_d2 = w2 * self.stl_scale, h2 * self.stl_scale, d2 * self.stl_scale
            if self.stl_scale != 1.0:
                logger.info(f"IMUVisualizer: Default box scaled by {self.stl_scale}")

            box_vertices = np.array([
                [-scaled_w2, -scaled_h2, -scaled_d2], [ scaled_w2, -scaled_h2, -scaled_d2], 
                [ scaled_w2,  scaled_h2, -scaled_d2], [-scaled_w2,  scaled_h2, -scaled_d2],
                [-scaled_w2, -scaled_h2,  scaled_d2], [ scaled_w2, -scaled_h2,  scaled_d2], 
                [ scaled_w2,  scaled_h2,  scaled_d2], [-scaled_w2,  scaled_h2,  scaled_d2]
            ], dtype=np.float32)
            
            box_faces_quads = np.array([ 
                [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
                [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3] 
            ], dtype=np.uint32)

            box_faces_triangles = []
            for quad in box_faces_quads:
                box_faces_triangles.append([quad[0], quad[1], quad[2]])
                box_faces_triangles.append([quad[0], quad[2], quad[3]])
            box_faces_triangles = np.array(box_faces_triangles, dtype=np.uint32)

            box_triangle_colors = np.repeat(self._DEFAULT_BOX_FACE_COLORS, 2, axis=0)
            
            self.mesh_data_item = gl.MeshData(vertexes=box_vertices, faces=box_faces_triangles, faceColors=box_triangle_colors)

    @staticmethod
    def _create_cylinder_mesh_data(radius: float, length: float, sections: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        verts = []
        faces = []
        for i in range(sections + 1):
            angle = (i / sections) * 2 * np.pi
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            verts.extend([[x, y, 0], [x, y, length]])
        for i in range(sections):
            i2 = i * 2; i2_n = ((i + 1) % sections) * 2
            faces.append([i2, i2 + 1, i2_n + 1]); faces.append([i2, i2_n + 1, i2_n])
        bottom_center_idx = len(verts); verts.append([0, 0, 0])
        top_center_idx = len(verts); verts.append([0, 0, length])
        for i in range(sections):
            i2 = i * 2; i2_n = ((i + 1) % sections) * 2
            faces.append([bottom_center_idx, i2_n, i2]); faces.append([top_center_idx, i2 + 1, i2_n + 1])
        return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.uint32)

    @staticmethod
    def _create_cone_mesh_data(radius: float, length: float, sections: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        verts = []
        faces = []
        for i in range(sections):
            angle = (i / sections) * 2 * np.pi
            x = radius * np.cos(angle); y = radius * np.sin(angle)
            verts.append([x, y, 0])
        tip_idx = len(verts); verts.append([0, 0, length])
        base_center_idx = len(verts); verts.append([0, 0, 0])
        for i in range(sections):
            i_n = (i + 1) % sections
            faces.append([tip_idx, i, i_n]); faces.append([base_center_idx, i_n, i])
        return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.uint32)

    def _setup_scene_elements(self):
        grid = gl.GLGridItem()
        grid.scale(self.GRID_SCALE, self.GRID_SCALE, 1)
        grid.setColor(self.GRID_COLOR)
        self.view.addItem(grid)

        shaft_verts, shaft_faces = self._create_cylinder_mesh_data(self.AXIS_RADIUS, self.AXIS_LENGTH)
        arrow_verts, arrow_faces = self._create_cone_mesh_data(self.ARROW_RADIUS, self.ARROW_LENGTH)

        # X Axis
        color_x = self.AXIS_COLORS['x']
        x_shaft_mesh = gl.GLMeshItem(vertexes=shaft_verts, faces=shaft_faces, color=color_x, smooth=True, computeNormals=True)
        tr_shaft_x = pg.Transform3D(); tr_shaft_x.rotate(90, 0, 1, 0); x_shaft_mesh.setTransform(tr_shaft_x)
        self.view.addItem(x_shaft_mesh)
        x_arrow_mesh = gl.GLMeshItem(vertexes=arrow_verts, faces=arrow_faces, color=color_x, smooth=True, computeNormals=True)
        tr_arrow_x = pg.Transform3D(); tr_arrow_x.translate(self.AXIS_LENGTH, 0, 0); tr_arrow_x.rotate(90, 0, 1, 0); x_arrow_mesh.setTransform(tr_arrow_x)
        self.view.addItem(x_arrow_mesh)

        # Y Axis
        color_y = self.AXIS_COLORS['y']
        y_shaft_mesh = gl.GLMeshItem(vertexes=shaft_verts, faces=shaft_faces, color=color_y, smooth=True, computeNormals=True)
        tr_shaft_y = pg.Transform3D(); tr_shaft_y.rotate(-90, 1, 0, 0); y_shaft_mesh.setTransform(tr_shaft_y)
        self.view.addItem(y_shaft_mesh)
        y_arrow_mesh = gl.GLMeshItem(vertexes=arrow_verts, faces=arrow_faces, color=color_y, smooth=True, computeNormals=True)
        tr_arrow_y = pg.Transform3D(); tr_arrow_y.translate(0, self.AXIS_LENGTH, 0); tr_arrow_y.rotate(-90, 1, 0, 0); y_arrow_mesh.setTransform(tr_arrow_y)
        self.view.addItem(y_arrow_mesh)

        # Z Axis
        color_z = self.AXIS_COLORS['z']
        z_shaft_mesh = gl.GLMeshItem(vertexes=shaft_verts, faces=shaft_faces, color=color_z, smooth=True, computeNormals=True)
        self.view.addItem(z_shaft_mesh) # No rotation needed for shaft
        z_arrow_mesh = gl.GLMeshItem(vertexes=arrow_verts, faces=arrow_faces, color=color_z, smooth=True, computeNormals=True)
        tr_arrow_z = pg.Transform3D(); tr_arrow_z.translate(0, 0, self.AXIS_LENGTH); z_arrow_mesh.setTransform(tr_arrow_z)
        self.view.addItem(z_arrow_mesh)



        # Object Mesh (STL or default box)
        if self.mesh_data_item:
            is_stl_loaded = bool(self.stl_filename and os.path.exists(self.stl_filename)) # Approx check if STL was intended

            self.object_mesh = gl.GLMeshItem(
                meshdata=self.mesh_data_item,
                smooth=False,  # Important for per-face colors from STL/Box to not be interpolated
                drawEdges=self.stl_draw_edges if is_stl_loaded else True, # Default box always draws edges
                edgeColor=self.stl_edge_color if is_stl_loaded else self._DEFAULT_BOX_EDGE_COLOR,
                computeNormals=True # Recompute normals, especially after mirroring/scaling
            )
            self.view.addItem(self.object_mesh)

        else:
            logger.error("IMUVisualizer: Mesh data was not loaded. Cannot create object mesh.")


    def update_component(self, current_relative_time: float, is_flowing: bool):


        # Directly use is_flowing argument, and global plotting_paused
        if plotting_paused and is_flowing: # If live and paused, do nothing.
            return
        
        is_uuid_missing_active = self.uuid_missing_overlay and self.uuid_missing_overlay.isVisible()

        if not is_flowing: # Replay mode or "Flowing Mode" checkbox unchecked
            self.scrub_time_widget_imu.setVisible(True)
            
            actual_min_data_time_imu = float('inf')
            actual_max_data_time_imu = 0.0
            found_any_data_imu = False

            for key_imu in self._required_data_types: # quat_w,x,y,z
                if key_imu in self.data_buffers_ref and self.data_buffers_ref[key_imu]:
                    series_times_imu = [item[0] for item in self.data_buffers_ref[key_imu]]
                    if series_times_imu:
                        actual_min_data_time_imu = min(actual_min_data_time_imu, series_times_imu[0])
                        actual_max_data_time_imu = max(actual_max_data_time_imu, series_times_imu[-1])
                        found_any_data_imu = True
            
            if not found_any_data_imu: # No data for any quat component
                actual_min_data_time_imu = 0.0
                actual_max_data_time_imu = 1.0 
            
            if actual_max_data_time_imu <= actual_min_data_time_imu: # e.g. only one data point
                actual_max_data_time_imu = actual_min_data_time_imu + 0.1

            # Scale for precision slider
            slider_min_int = int(math.floor(actual_min_data_time_imu * self.SLIDER_FLOAT_PRECISION_FACTOR_IMU))
            slider_max_int = int(math.ceil(actual_max_data_time_imu * self.SLIDER_FLOAT_PRECISION_FACTOR_IMU))

            if slider_max_int <= slider_min_int: # Ensure slider has a range > 0
                slider_max_int = slider_min_int + 1

            self.time_slider_imu.blockSignals(True)
            if self.time_slider_imu.minimum() != slider_min_int or \
               self.time_slider_imu.maximum() != slider_max_int:
                current_val = self.time_slider_imu.value()
                self.time_slider_imu.setRange(slider_min_int, slider_max_int)
                if slider_min_int <= current_val <= slider_max_int:
                    self.time_slider_imu.setValue(current_val)
                else:
                    self.time_slider_imu.setValue(slider_min_int)
            self.time_slider_imu.blockSignals(False)


            # Update the replay time label to reflect the current slider value
            current_slider_val_for_label_imu = self.time_slider_imu.value()
            time_sec_for_label_imu = float(current_slider_val_for_label_imu) / self.SLIDER_FLOAT_PRECISION_FACTOR_IMU
            if hasattr(self, 'current_replay_time_label_imu'):
                self.current_replay_time_label_imu.setText(f"{time_sec_for_label_imu:.2f}s")

            if plotting_paused: # This means we are in slider mode
                self.render_imu_for_time(float(self.time_slider_imu.value()) / self.SLIDER_FLOAT_PRECISION_FACTOR_IMU)
        
        else: # is_flowing is True (Live mode)
            self.scrub_time_widget_imu.setVisible(False)
            if is_uuid_missing_active: # If UUIDs missing in live mode, show default
                relative_quat = QQuaternion(1,0,0,0)
                self.orientation_status_label.setText("Yaw: --- Pitch: --- Roll: ---")
            else:
                q_w, q_x, q_y, q_z = None, None, None, None
                if 'quat_w' in self.data_buffers_ref and self.data_buffers_ref['quat_w']: q_w = self.data_buffers_ref['quat_w'][-1][1]
                if 'quat_x' in self.data_buffers_ref and self.data_buffers_ref['quat_x']: q_x = self.data_buffers_ref['quat_x'][-1][1]
                if 'quat_y' in self.data_buffers_ref and self.data_buffers_ref['quat_y']: q_y = self.data_buffers_ref['quat_y'][-1][1]
                if 'quat_z' in self.data_buffers_ref and self.data_buffers_ref['quat_z']: q_z = self.data_buffers_ref['quat_z'][-1][1]

                if all(q is not None for q in [q_w, q_x, q_y, q_z]):
                    self.current_quaternion = QQuaternion(float(q_w), float(q_x), float(q_y), float(q_z))
                else: # Data incomplete for live update
                    self.current_quaternion = QQuaternion(1,0,0,0) 

                relative_quat = self.baseline_quaternion.inverted() * self.current_quaternion
                yaw, pitch, roll = self._get_euler_angles_from_qt_quaternion(relative_quat)
                self.orientation_status_label.setText(f"Yaw: {yaw:6.1f}°  Pitch: {pitch:6.1f}°  Roll: {roll:6.1f}°")

            transform = pg.Transform3D()
            transform.rotate(relative_quat)
            if self.object_mesh: # Check if mesh item exists
                self.object_mesh.setTransform(transform)


    def _get_euler_angles_from_qt_quaternion(self, q: QQuaternion) -> Tuple[float, float, float]:
        euler_vector = q.toEulerAngles() # Returns QVector3D: (pitch, yaw, roll)
        pitch = euler_vector.x()
        yaw = euler_vector.y()
        roll = euler_vector.z()
        return yaw, pitch, roll

    def clear_component(self):
        self.current_quaternion = QQuaternion(1, 0, 0, 0)
        self.baseline_quaternion = QQuaternion(1, 0, 0, 0)
        
        identity_quat = QQuaternion(1,0,0,0)
        transform = pg.Transform3D()
        transform.rotate(identity_quat)
        if self.object_mesh: # Check if mesh item exists
            self.object_mesh.setTransform(transform)
        

        yaw, pitch, roll = self._get_euler_angles_from_qt_quaternion(identity_quat)
        self.orientation_status_label.setText(f"Yaw: {yaw:6.1f}°  Pitch: {pitch:6.1f}°  Roll: {roll:6.1f}°")

        super().handle_missing_uuids(set()) # Clear UUID overlay
        super().handle_missing_replay_data(set()) # Clear replay data overlay

        if hasattr(self, 'snapshot_button'): self.snapshot_button.setEnabled(True)
        if hasattr(self, 'reset_button'): self.reset_button.setEnabled(True)

        # --- Reset time slider and hide its container ---
        if hasattr(self, 'time_slider_imu'):
            self.time_slider_imu.blockSignals(True)
            self.time_slider_imu.setRange(0, 0) # Reset range
            self.time_slider_imu.setValue(0)    # Reset value
            self.time_slider_imu.blockSignals(False)
            # The slider's visibility is controlled by its parent scrub_time_widget_imu

        if hasattr(self, 'scrub_time_widget_imu'):
            self.scrub_time_widget_imu.setVisible(False)

        if hasattr(self, 'current_replay_time_label_imu'):
            self.current_replay_time_label_imu.setText("0.00s")


    def _take_snapshot_action(self):
        if not hasattr(self, 'view'): return

        is_paused_or_replaying_imu = plotting_paused or (state == "replay_active")
        if is_paused_or_replaying_imu and self.scrub_time_widget_imu.isVisible():
            current_slider_int_value_imu = self.time_slider_imu.value()
            current_slider_time_imu_sec = float(current_slider_int_value_imu) / self.SLIDER_FLOAT_PRECISION_FACTOR_IMU
            logger.info(f"IMU snapshot: Rendering view for slider time {current_slider_time_imu_sec:.2f}s (raw slider: {current_slider_int_value_imu}).")
            # Ensure render_imu_for_time updates the view to the specified time *before* snapshotting
            self.render_imu_for_time(current_slider_time_imu_sec)
        else:
            logger.info("IMU snapshot: Using live/latest orientation data (not from slider).")
            # For live data, the view should be current from update_component.

        try:
            # *** THE FIX IS HERE: Use grabFramebuffer() directly ***
            image = self.view.grabFramebuffer()
            
            if image.isNull():
                logger.warning("IMU Snapshot: grabFramebuffer returned a null image. Cannot save.")
                return

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = os.path.join(self.snapshot_dir, f"snapshot_{timestamp}.png")
            
            # *** THE FIX IS HERE: Use grabFramebuffer() directly ***
            image = self.view.grabFramebuffer()
            
            if image.isNull():
                logger.warning("IMU Snapshot: grabFramebuffer returned a null image. Cannot save.")
                return

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            # *** Save as JPEG for smaller size ***
            # Use '.jpg' extension and specify "JPG" format with a quality (e.g., 85)
            filename = os.path.join(self.snapshot_dir, f"snapshot_{timestamp}.jpg")
            quality = 85 # Adjust quality (0-100) for filesize vs detail trade-off
            
            if image.save(filename, "JPG", quality):
                logger.info(f"IMU Snapshot saved as {filename} with JPG quality {quality}.")
            else:
                logger.error(f"IMU Snapshot: Failed to save QImage to {filename} as JPG.")

        except Exception as e:
            logger.error(f"IMU Snapshot failed: {e}", exc_info=True)




    def _reset_orientation_action(self):
        self.baseline_quaternion = QQuaternion(
            self.current_quaternion.scalar(),
            self.current_quaternion.x(),
            self.current_quaternion.y(),
            self.current_quaternion.z(),
        )
        # Force an update to reflect the reset
        self.update_component(0, False) # Args might not be ideal, but triggers logic



    def on_time_slider_imu(self, slider_value_int: int): # Renamed arg for clarity
        global plotting_paused
        if not plotting_paused: # If we are entering scrub mode via slider interaction
            plotting_paused = True 
            main_window = self.window()
            if isinstance(main_window, MainWindow):
                if main_window.pause_resume_button.text() != "Resume Plotting":
                     main_window.pause_resume_button.setText("Resume Plotting")
        
        time_sec_float = float(slider_value_int) / self.SLIDER_FLOAT_PRECISION_FACTOR_IMU

        if hasattr(self, 'current_replay_time_label_imu'):
            self.current_replay_time_label_imu.setText(f"{time_sec_float:.2f}s")

        self.render_imu_for_time(time_sec_float)

    def render_imu_for_time(self, time_sec: float):
        q_w = get_value_at_time("quat_w", time_sec, self.data_buffers_ref)
        q_x = get_value_at_time("quat_x", time_sec, self.data_buffers_ref)
        q_y = get_value_at_time("quat_y", time_sec, self.data_buffers_ref)
        q_z = get_value_at_time("quat_z", time_sec, self.data_buffers_ref)

        historical_quaternion = QQuaternion(1,0,0,0) # Default to identity
        if all(q is not None for q in [q_w, q_x, q_y, q_z]):
            historical_quaternion = QQuaternion(float(q_w), float(q_x), float(q_y), float(q_z))
        
        
        relative_quat_hist = self.baseline_quaternion.inverted() * historical_quaternion
        
        yaw, pitch, roll = self._get_euler_angles_from_qt_quaternion(relative_quat_hist)
        self.orientation_status_label.setText(f"Yaw: {yaw:6.1f}°  Pitch: {pitch:6.1f}°  Roll: {roll:6.1f}° (T={time_sec:.1f}s)")

        transform = pg.Transform3D()
        transform.rotate(relative_quat_hist)
        if self.object_mesh: # Check if mesh item exists
            self.object_mesh.setTransform(transform)

    def get_widget(self) -> QWidget:
        return self

    def get_required_data_types(self) -> Set[str]:
        return self._required_data_types

    def get_log_filename_suffix(self) -> str:
        if self.is_loggable:
            title = self.config.get('title', 'IMU_3D_Visualizer')
            safe_suffix = re.sub(r'[^\w\-]+', '_', title).strip('_')
            return f"imuvis_{safe_suffix}" if safe_suffix else f"imuvis_{id(self)}"
        return ""


    def handle_missing_uuids(self, missing_uuids_for_component: Set[str]):
        super().handle_missing_uuids(missing_uuids_for_component) # Show/hide overlay
        self._update_controls_based_on_data_status()


    def _update_controls_based_on_data_status(self):
        is_overlay_active = self.uuid_missing_overlay and self.uuid_missing_overlay.isVisible()
        are_buttons_defined = hasattr(self, 'snapshot_button') and hasattr(self, 'reset_button')

        if are_buttons_defined:
            self.snapshot_button.setEnabled(not is_overlay_active)
            self.reset_button.setEnabled(not is_overlay_active)
        
        if is_overlay_active:
            # Force view to identity and update label
            self.current_quaternion = QQuaternion(1,0,0,0)
            # Do not reset baseline_quaternion here as it's user set.
            # Let update_component handle the visual reset based on current_quaternion.
            self.update_component(0, False) # Trigger visual update to identity
            self.orientation_status_label.setText("Yaw: --- Pitch: --- Roll: ---")

    def handle_missing_replay_data(self, missing_data_types_for_component: Set[str]):
        super().handle_missing_replay_data(missing_data_types_for_component)
        self._update_controls_based_on_data_status()

# ------------------------------------------------------------------

# 6. --- Tab Layout Configuration ---

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
                    'grid_resolution': 20, # in pixels, default is 10 (higher values means bigger pixels)
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
        'tab_title': 'IMU 3D View', # New Tab
        'layout': [
            {
                'component_class': IMUVisualizerComponent,
                'row': 0, 'col': 0, 'rowspan':1, 'colspan':1,
                'config': {
                    'title': 'IMU Orientation Visualizer',
                    'enable_logging': True, 
                    # --- New STL configurations ---
                    'stl_filename': ASSETS_DIR / "Sohle1V2_MIR.stl",  # Optional: Path to your .stl file
                    'stl_scale': 0.02, # Example if STL is in mm and you want meters for display
                    'use_stl_attributes_for_color': False,  # << SET TO FALSE
                    'stl_color': (0.5, 0.5, 0.5, 0),    # << SET YOUR DESIRED GREY COLOR
                    'mirror_x': False,                         # Optional: True to mirror on X
                    'mirror_y': False,                          # Optional: True to mirror on Y (e.g. for right-handed to left-handed model)
                    'mirror_z': False,                         # Optional: True to mirror on Z
                    #'stl_color': (0.8, 0.8, 0.3, 1.0),         # Optional: RGBA tuple (0-1) to color the whole STL if it has no color attributes or you want to override.
                    'stl_draw_edges': True,                    # Optional: True to draw edges on the STL model
                    'stl_edge_color': (0.1, 0.1, 0.1, 1.0)     # Optional: Color for STL edges if drawn
                    # 'component_height': 600, # Optional: set fixed size
                    # 'component_width': 800,  # Optional: set fixed size
                }
            },
        ]
    },
    {
        'tab_title': 'Impedance',
        'layout': [
            {
                'component_class': NyquistPlotComponent,
                'row': 0, 'col': 0, 'rowspan': 2, 'colspan': 1,
                'config': {
                    'title': 'Live Nyquist Plot', 'xlabel': "Re(Z) [kOhm]", 'ylabel': "-Im(Z) [kOhm]",
                    'plot_height': 600, 'plot_width': 600,  
                    'snapshot_dir': 'nyquist_snapshots', 'enable_logging': True 
                }
            },
            {
                'component_class': TimeSeriesPlotComponent, 'row': 0, 'col': 1,
                'config': {
                    'title': 'Impedance Magnitude vs Time', 'xlabel': 'Time [s]', 'ylabel': '|Z| [Ohm]',
                    'plot_height': 300, 
                    'datasets': [{'data_type': 'impedance_magnitude_ohm', 'label': '|Z|', 'color': 'purple'}],
                    'enable_logging': True
                }
            },
            {
                'component_class': TimeSeriesPlotComponent, 'row': 1, 'col': 1,
                'config': {
                    'title': 'Impedance Phase vs Time', 'xlabel': 'Time [s]', 'ylabel': 'Phase [degrees]',
                    'plot_height': 300, 
                    'datasets': [{'data_type': 'impedance_phase_deg', 'label': 'Phase', 'color': 'orange'}],
                    'enable_logging': True
                }
            },
            {
                'component_class': TimeSeriesPlotComponent, 'row': 2, 'col': 1,
                'config': {
                    'title': 'd|Z|/dt vs Time',
                    'xlabel': 'Time [s]',
                    'ylabel': 'Δ|Z|/Δt  [Ω/s]',
                    'plot_height': 300,
                    'datasets': [
                        {'data_type': 'impedance_change_speed_ohm_per_s',
                         'label': 'd|Z|/dt',
                         'color': 'r'}
                    ],
                    'enable_logging': True
                }
            }
        ]
    },
    {
        'tab_title': 'IMU Basic',
        'layout': [
            {   'component_class': TimeSeriesPlotComponent, 'row': 0, 'col': 0,
                'config': {
                    'title': 'Orientation vs Time', 'xlabel': 'Time [s]', 'ylabel': 'Degrees',
                    'plot_height': 300, 'plot_width': 450,
                    'datasets': [{'data_type': 'orientation_x', 'label': 'X (Roll)', 'color': 'r'},
                                 {'data_type': 'orientation_y', 'label': 'Y (Pitch)', 'color': 'g'},
                                 {'data_type': 'orientation_z', 'label': 'Z (Yaw)', 'color': 'b'}],
                    'enable_logging': True
                }
            },
            {   'component_class': TimeSeriesPlotComponent, 'row': 0, 'col': 1,
                'config': {
                    'title': 'Angular Velocity vs Time','xlabel': 'Time [s]','ylabel': 'Degrees/s',
                    'plot_height': 300, 'plot_width': 600,
                    'datasets': [{'data_type': 'gyro_x', 'label': 'X', 'color': 'r'},
                                 {'data_type': 'gyro_y', 'label': 'Y', 'color': 'g'},
                                 {'data_type': 'gyro_z', 'label': 'Z', 'color': 'b'}],
                    'enable_logging': False
                }
            },
            {   'component_class': SingleValueDisplayComponent, 'row': 1, 'col': 0,
                'config': {
                    'label': 'Current Roll', 'data_type': 'orientation_x',
                    'format': '{:.1f}', 'units': '°', 'enable_logging': True
                }
            },
            {   'component_class': SingleValueDisplayComponent, 'row': 1, 'col': 1,
                'config': {
                    'label': 'Current Yaw Rate', 'data_type': 'gyro_z',
                    'format': '{:.1f}', 'units': '°/s', 'enable_logging': False
                }
            }
        ]
    },
    {
        'tab_title': 'IMU Acceleration',
        'layout': [
            {   'component_class': TimeSeriesPlotComponent, 'row': 0, 'col': 0,
                'config': {
                    'title': 'Linear Acceleration vs Time','xlabel': 'Time [s]','ylabel': 'm/s²',
                    'plot_height': 300,
                    'datasets': [{'data_type': 'lin_accel_x', 'label': 'X', 'color': 'r'},
                                 {'data_type': 'lin_accel_y', 'label': 'Y', 'color': 'g'},
                                 {'data_type': 'lin_accel_z', 'label': 'Z', 'color': 'b'}],
                    'enable_logging': True
                }
            },
            {   'component_class': TimeSeriesPlotComponent, 'row': 1, 'col': 0,
                'config': {
                    'title': 'Raw Acceleration vs Time','xlabel': 'Time [s]','ylabel': 'm/s²',
                    'plot_height': 300,
                    'datasets': [{'data_type': 'accel_x', 'label': 'X', 'color': 'r'},
                                 {'data_type': 'accel_y', 'label': 'Y', 'color': 'g'},
                                 {'data_type': 'accel_z', 'label': 'Z', 'color': 'b'}]
                }
            },
            {   'component_class': TimeSeriesPlotComponent, 'row': 0, 'col': 1,
                'config': {
                     'title': 'Gravity vs Time','xlabel': 'Time [s]','ylabel': 'm/s²',
                     'plot_height': 300,
                     'datasets': [{'data_type': 'gravity_x', 'label': 'X', 'color': 'r'},
                                  {'data_type': 'gravity_y', 'label': 'Y', 'color': 'g'},
                                  {'data_type': 'gravity_z', 'label': 'Z', 'color': 'b'}],
                     'enable_logging': True
                }
            }
        ]
    },
    {
        'tab_title': 'Other Sensors',
        'layout': [
             {  'component_class': TimeSeriesPlotComponent, 'row': 0, 'col': 0,
                 'config': {
                     'title': 'Magnetic Field vs Time','xlabel': 'Time [s]','ylabel': 'µT',
                     'plot_height': 350, 'plot_width': 600,
                     'datasets': [{'data_type': 'mag_x', 'label': 'X', 'color': 'r'},
                                  {'data_type': 'mag_y', 'label': 'Y', 'color': 'g'},
                                  {'data_type': 'mag_z', 'label': 'Z', 'color': 'b'}],
                     'enable_logging': True
                 }
             },
             {   'component_class': SingleValueDisplayComponent, 'row': 1, 'col': 0,
                 'config': {
                    'label': 'Current Mag X', 'data_type': 'mag_x',
                    'format': '{:.1f}', 'units': 'µT', 'enable_logging': True
                }
            },
        ]
    },
    {
        'tab_title': 'Optical Flow',
        'layout': [
            {   'component_class': TimeSeriesPlotComponent, 'row': 0, 'col': 0,
                'config': {
                    'title': 'Optical Flow Δ vs Time', 'xlabel': 'Time [s]', 'ylabel': 'Pixel Δ',
                    'plot_height': 350, 'plot_width': 600,
                    'datasets': [ {'data_type': 'opt_dx', 'label': 'ΔX', 'color': 'r'},
                                  {'data_type': 'opt_dy', 'label': 'ΔY', 'color': 'g'}],
                    'enable_logging': True
                }
            },
            {   'component_class': TimeSeriesPlotComponent, 'row': 0, 'col': 1,
                'config': {
                    'title': 'Cumulative Optical Flow vs Time', 'xlabel': 'Time [s]', 'ylabel': 'Pixels',
                    'plot_height': 350, 'plot_width': 600,
                    'datasets': [ {'data_type': 'opt_cum_x', 'label': 'Cum X', 'color': 'r'},
                                  {'data_type': 'opt_cum_y', 'label': 'Cum Y', 'color': 'g'}],
                    'enable_logging': True
                }
            }
        ]
    },
    {
        'tab_title': 'ToF Sensor',
        'layout': [
            {   'component_class': TimeSeriesPlotComponent, 'row': 0, 'col': 0,
                'config': {
                    'title': 'Distance vs Time', 'xlabel': 'Time [s]', 'ylabel': 'Distance [mm]',
                    'plot_height': 350, 'plot_width': 450,
                    'datasets': [ {'data_type': 'tof_distance_mm', 'label': 'Distance', 'color': 'b'}],
                    'enable_logging': True
                }
            },
            {   'component_class': TimeSeriesPlotComponent, 'row': 0, 'col': 1,
                'config': {
                    'title': 'Darkness vs Time', 'xlabel': 'Time [s]', 'ylabel': 'Darkness [kcps/spad]',
                    'plot_height': 350, 'plot_width': 450,
                    'datasets': [ {'data_type': 'tof_brightness_kcps', 'label': 'Darkness', 'color': 'k'}],
                    'enable_logging': True
                }
            }
        ]
    },
    {
        'tab_title': 'Ankle Angles',
        'layout': [
            {   'component_class': TimeSeriesPlotComponent, 'row': 0, 'col': 0,
                'config': {
                    'title': 'Ankle XZ vs Time', 'xlabel': 'Time [s]', 'ylabel': 'Angle [°]',
                    'plot_height': 300, 'plot_width': 450,
                    'datasets': [ {'data_type': 'ankle_xz', 'label': 'XZ', 'color': 'r'}],
                    'enable_logging': True
                }
            },
            {   'component_class': TimeSeriesPlotComponent, 'row': 0, 'col': 1,
                'config': {
                    'title': 'Ankle YZ vs Time', 'xlabel': 'Time [s]', 'ylabel': 'Angle [°]',
                    'plot_height': 300, 'plot_width': 450,
                    'datasets': [ {'data_type': 'ankle_yz', 'label': 'YZ', 'color': 'g'}],
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
        if start_time is None: # Only return if there's no session start time (e.g., before any data)
            return
        # The plotting_paused logic will be handled by individual components
        # based on whether they are in a "flowing" state or a "static/slider" state.
        for component in self.all_components:
            try:
                component.update_component(current_relative_time, is_flowing)
            except Exception as e:
                logger.error(f"Error updating component {type(component).__name__}: {e}", exc_info=True)

    def clear_components(self, clear_only_connected_data: bool):
        if clear_only_connected_data:
            logger.info("GuiManager clearing data for connected components only (preserving overlays for unconnected)...")
        else:
            logger.info("GuiManager clearing all components fully (including overlays)...")

        for component in self.all_components:
            try:
                if clear_only_connected_data:
                    # Determine if this component is "fully connected"
                    # A component is fully connected if none of its required data types
                    # map to UUIDs that are currently in self.active_missing_uuids.
                    is_this_component_fully_connected = True
                    if not component.get_required_data_types(): # Component requires no data, so it's "connected"
                        pass
                    else:
                        for dtype in component.get_required_data_types():
                            # _missing_uuids_for_dtype returns the set of *actually missing* UUIDs for this dtype
                            if self._missing_uuids_for_dtype(dtype): # If any required dtype is missing UUIDs
                                is_this_component_fully_connected = False
                                break
                    
                    if is_this_component_fully_connected:
                        logger.debug(f"Clearing connected component: {type(component).__name__} (Title: {component.config.get('title', 'N/A')})")
                        component.clear_component()
                        # Ensure its overlays are cleared as it's considered connected
                        component.handle_missing_uuids(set())
                        component.handle_missing_replay_data(set())
                    else:
                        # For unconnected components, do not call clear_component().
                        # Their data in data_buffers is cleared globally.
                        # Their "missing UUID" overlay (already set by notify_missing_uuids) should remain.
                        # We might want to clear any replay-specific overlay if it exists.
                        logger.debug(f"Skipping clear for unconnected component (preserving UUID overlay): {type(component).__name__} (Title: {component.config.get('title', 'N/A')})")
                        component.handle_missing_replay_data(set()) # Clear replay overlay even if UUID overlay stays
                else:
                    # Clear all components fully (behavior when disconnected/idle)
                    logger.debug(f"Fully clearing component: {type(component).__name__} (Title: {component.config.get('title', 'N/A')})")
                    component.clear_component()
                    component.handle_missing_uuids(set())
                    component.handle_missing_replay_data(set())
            except Exception as e:
                logger.error(f"Error during component clear ({'connected_only' if clear_only_connected_data else 'all'}): {type(component).__name__} - {e}", exc_info=True)

        if not clear_only_connected_data:
            # Only clear the GuiManager's active_missing_uuids set if we are doing a full clear (e.g., on disconnect)
            logger.debug("Full clear: Resetting GuiManager.active_missing_uuids.")
            self.active_missing_uuids.clear()
        # If clear_only_connected_data is True, self.active_missing_uuids should persist.


    # ------------------------------------------------------------------
    def _missing_uuids_for_dtype(self, dtype: str) -> Set[str]:
        """Recursively resolve which underlying UUIDs are absent for a data_type and fusion."""
        uuid = self.device_config_ref.get_uuid_for_data_type(dtype)
        if uuid:
            return {uuid} if uuid in self.active_missing_uuids else set()
        if dtype in derived_data_definitions:
            miss: Set[str] = set()
            for dep in derived_data_definitions[dtype].dependencies:
                miss |= self._missing_uuids_for_dtype(dep)
            return miss
        return set()
    # ------------------------------------------------------------------



    def notify_missing_uuids(self, missing_uuids_set: Set[str]):
        logger.info(f"GuiManager received missing UUIDs: {missing_uuids_set if missing_uuids_set else 'None'}")
        self.active_missing_uuids = missing_uuids_set
        for component in self.all_components:
            required_types = component.get_required_data_types()
            if not required_types:
                continue
            relevant_missing_uuids_for_comp = set()
            for data_type in required_types:
                relevant_missing_uuids_for_comp |= self._missing_uuids_for_dtype(data_type)
            try:
                component.handle_missing_uuids(relevant_missing_uuids_for_comp)
            except Exception as e:
                 logger.error(f"Error notifying component {type(component).__name__} about missing UUIDs: {e}", exc_info=True)


    def notify_missing_replay_data(self):
        """
        Checks which components are missing required data types from data_buffers
        during replay mode and notifies them.
        """
        if state != "replay_active": # Only relevant in replay mode
            # If not in replay, ensure all "CSV Not Loaded" overlays are cleared
            for component in self.all_components:
                try:
                    component.handle_missing_replay_data(set())
                except Exception as e:
                    logger.error(f"Error clearing replay_data overlay for {type(component).__name__}: {e}")
            return

        logger.info("GuiManager: Checking for missing replay data for components...")
        missing_globally = 0
        for component in self.all_components:
            required_types = component.get_required_data_types()
            if not required_types:
                component.handle_missing_replay_data(set()) # Ensure overlay is cleared if no types required
                continue

            missing_for_this_component = set()
            for data_type in required_types:
                # After CsvReplaySource.start() and compute_all_derived_data(),
                # data_buffers should contain all available raw and derived types.
                # So, just check if the required data_type (be it raw or derived)
                # is present and has data in data_buffers.
                if data_type not in self.data_buffers_ref or not self.data_buffers_ref[data_type]:
                    missing_for_this_component.add(data_type)
            
            # Determine which required types ARE present for this component
            loaded_for_this_component = required_types - missing_for_this_component

            if missing_for_this_component:
                # Still log a single warning if any data is missing for the component overall, but not per type unless debugging
                logger.debug(f"Component {type(component).__name__} (title: {component.config.get('title', 'N/A')}) is still missing replay data for: {missing_for_this_component}")
                missing_globally += len(missing_for_this_component)
            
            if loaded_for_this_component:
                 logger.info(f"Component {type(component).__name__} (title: {component.config.get('title', 'N/A')}) has loaded replay data for: {loaded_for_this_component}")
            
            try:
                # This call is still crucial for showing/hiding the "Data Not Loaded" overlay
                component.handle_missing_replay_data(missing_for_this_component)
            except Exception as e:
                logger.error(f"Error notifying component {type(component).__name__} about missing replay data: {e}", exc_info=True)

        if missing_globally == 0 and any(comp.get_required_data_types() for comp in self.all_components): # Check if any component actually requires data
            logger.info("GuiManager: All components requiring data have their required data types loaded from CSVs.")
        elif not any(comp.get_required_data_types() for comp in self.all_components):
            logger.info("GuiManager: No components are configured to require data for replay.")
        else:
            logger.warning(f"GuiManager: Some components are still missing data for replay (total missing type instances: {missing_globally}).")

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
    # --- derive / fuse ----------------------------------------------------------
    compute_all_derived_data(relative_time)


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


# --- PyQt6 Main Application Window ---

class LEDWidget(QWidget):
    # Green / orange / red LED indicator for the top bar.
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
        self.setGeometry(100, 100, 1400, 900)

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

        self.current_source: Optional[DataSource] = None

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
        
        self.replay_button = QPushButton("Replay CSV…") # Text will change in replay mode
        self.replay_button.clicked.connect(self.handle_replay_action_button)
        self.button_layout.addWidget(self.replay_button)

        self.load_more_csvs_button = QPushButton("Load More CSVs")
        self.load_more_csvs_button.clicked.connect(self.handle_load_more_csvs_action)
        self.load_more_csvs_button.setVisible(False) # Initially hidden
        self.button_layout.addWidget(self.load_more_csvs_button)

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

        # ---------- Help Button ----------
        self.help_button = QPushButton("Help?")
        self.help_button.clicked.connect(self.show_help_window)
        self.bottom_layout.addWidget(self.help_button)

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
            # REMOVED: The call to self.gui_manager.notify_missing_replay_data()
            # It should only be called when data_buffers actually change during replay.

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


    @qasync.asyncSlot()
    async def handle_replay_action_button(self):
        if self._shutting_down: return

        if state == "idle":
            await self.open_replay_dialog()
        elif state == "replay_active":
            await self.exit_replay_mode()
        else:
            logger.warning(f"Replay action button clicked in unexpected state: {state}")



    @qasync.asyncSlot()
    async def exit_replay_mode(self):
        global plotting_paused, start_time, data_buffers, current_task, state # Moved global state here
        logger.info("Exiting replay mode...")
        
        # If there was a CsvReplaySource task (though it should be done for bulk load),
        # ensure it's handled. For bulk load, current_source might already be None.
        if self.current_source and isinstance(self.current_source, CsvReplaySource):
            logger.info("Stopping CsvReplaySource explicitly on exit replay (should be no-op if bulk load finished).")
            await self.stop_current_source_and_wait() # This also clears current_task
        elif current_task: # If a task exists but source is None
            logger.warning("Found an orphaned current_task during exit_replay_mode. Attempting to cancel.")
            await self.cancel_and_wait_task(current_task)
            current_task = None

        # --- FULL CLEANUP ---
        logger.info("Performing full GUI and data clear after exiting replay mode.")
        
        # 1. Clear data buffers, reset session start_time, and other global data stores.
        # clear_gui_action handles data_buffers, start_time, opt_cum_x/y, and GuiManager.clear_all_components
        self.clear_gui_action(confirm=False) # This already does comprehensive data and GUI clearing
        
        # 2. Ensure GuiManager's active_missing_uuids is reset
        if hasattr(self, 'gui_manager'):
            self.gui_manager.notify_missing_uuids(set()) 
        
        # 3. Reset plotting state for IDLE mode
        plotting_paused = True # Default for idle is paused
        
        # 4. Re-enable and reset flowing mode checkbox to its default (usually checked)
        self.flowing_mode_check.setEnabled(True)
        self.flowing_mode_check.setChecked(True) 
        self.apply_interval()

        # 5. Reset any replay-specific display attributes
        if hasattr(self, 'df_filepath_for_display'):
            del self.df_filepath_for_display

        # 6. Transition to idle state. The handle_state_change("idle") will trigger
        #    the necessary GuiManager calls to clear all overlays because clear_gui_action
        #    calls GuiManager.clear_all_components which now robustly clears both overlay types.
        self.handle_state_change("idle") 
        
        logger.info("Returned to idle state from replay mode. GUI and data fully reset.")


    def handle_state_change(self, new_state: str):
        global state, plotting_paused, start_time
        logger.info(f"GUI received state change: {new_state} (from {state})")
        
        previous_state = state
        state = new_state

        if state != "scanning" and self.scan_throbber_timer.isActive():
            self.scan_throbber_timer.stop()



        # Default visibility and enablement (will be overridden by specific states)
        self.scan_button.setVisible(True); self.scan_button.setEnabled(True)
        self.device_label.setVisible(True) # Assuming this is a label next to combo
        self.device_combo.setVisible(True); self.device_combo.setEnabled(True)
        self.replay_button.setVisible(True); self.replay_button.setEnabled(True)
        self.load_more_csvs_button.setVisible(False) # Hidden by default
        self.pause_resume_button.setVisible(True); self.pause_resume_button.setEnabled(False)
        self.capture_button.setVisible(True); self.capture_button.setEnabled(False)
        self.clear_button.setVisible(True); self.clear_button.setEnabled(True)
        self.flowing_mode_check.setEnabled(True) # Checkbox usually enabled

        if state == "idle":
            self.scan_button.setText("Start Scan")
            self.replay_button.setText("Replay CSV…")

            if not self._shutting_down: # Only set these if not quitting
                self.led_indicator.set_color("red")
                self.status_label.setText("On Standby")
            # else: if _shutting_down, LED is gray and status is "Shutting down..." from closeEvent

            self.pause_resume_button.setText("Pause Plotting")

            self.pause_resume_button.setEnabled(False) # Can't pause/resume if not connected/replaying
            plotting_paused = True 
            
            
            # When transitioning to idle, explicitly clear all component overlays
            if hasattr(self, 'gui_manager'):
                logger.debug("Idle state: Ensuring all component overlays (UUID and Replay) are cleared via GuiManager.")
                self.gui_manager.notify_missing_uuids(set()) # Clears UUID overlays
                self.gui_manager.notify_missing_replay_data() # Clears Replay data overlays

            if previous_state == "scanning": 
                logger.info("State changed to idle from scanning (scan failed/cancelled). Automatically clearing GUI.")
                self.clear_gui_action(confirm=False)
            elif previous_state == "replay_active":
                 logger.info("State changed to idle from replay_active. GUI and data were cleared by exit_replay_mode.")
                 # plotting_paused is already True, GUI is clear. Flowing mode enabled by default.
            
            if self.is_capturing: # Should not happen if logic is correct
                logger.warning("Capture was active when state became idle. Files NOT generated.")
                self.is_capturing = False 
            self.capture_button.setText("Start Capture")
            self.capture_button.setEnabled(False) # Can only capture when connected

        elif state == "scanning":
            self.scan_button.setText("Stop Scan")
            self.replay_button.setEnabled(False) # Cannot start replay during scan
            self.device_combo.setEnabled(False)
            self.led_indicator.set_color("orange")
            self.throbber_index = 0
            if not self.scan_throbber_timer.isActive(): self.scan_throbber_timer.start()
            plotting_paused = True # No plotting during scan phase
            self.pause_resume_button.setText("Pause Plotting")
            self.pause_resume_button.setEnabled(False)

        elif state == "connected":
            self.scan_button.setText("Disconnect")
            self.replay_button.setEnabled(False) 
            self.device_combo.setEnabled(False)
            self.led_indicator.set_color("lightgreen")
            self.status_label.setText(f"Connected to: {device_config.name}")
            
            plotting_paused = False 
            if not self.is_capturing:
                self.pause_resume_button.setEnabled(True)
            self.pause_resume_button.setText("Pause Plotting")
            self.capture_button.setEnabled(True)

        elif state == "replay_active":
            # Hide/disable standard live mode buttons
            self.scan_button.setVisible(False)
            self.device_label.setVisible(False)
            self.device_combo.setVisible(False)
            self.pause_resume_button.setVisible(False)
            self.capture_button.setVisible(False)
            self.clear_button.setVisible(False) # Clear is handled by Exit Replay

            # Configure replay mode buttons
            self.replay_button.setText("Exit Replay")
            self.replay_button.setVisible(True)
            self.replay_button.setEnabled(True) 
            self.load_more_csvs_button.setVisible(True)
            self.load_more_csvs_button.setEnabled(True)
            
            self.led_indicator.set_color("purple")
            self.flowing_mode_check.setEnabled(False) # Flowing mode not used in replay
            self.flowing_mode_check.setChecked(False) 
            
            # plotting_paused is set to True by _load_csvs_for_replay to enable sliders.
            # The actual display of data is then driven by sliders.
            # Status label is updated by _load_csvs_for_replay.
            if not plotting_paused: # Should be true after loading
                 logger.warning("Plotting was not paused after entering replay_active. Forcing pause for sliders.")
                 plotting_paused = True
            self.pause_resume_button.setText("Resume Plotting") # Reflects plotting_paused = True
            
            # Status label will be set by the loading function, e.g.:
            # self.status_label.setText(f"Replay Ready. View: {os.path.basename(getattr(self, 'df_filepath_for_display', 'CSVs'))}. Click 'Exit Replay'.")


        elif state == "disconnecting":
            self.scan_button.setText("Disconnecting...")
            self.scan_button.setEnabled(False)
            self.replay_button.setEnabled(False)
            self.device_combo.setEnabled(False)
            
            if not self._shutting_down: # Only set orange LED and specific status if it's a manual disconnect
                self.led_indicator.set_color("orange")
                self.status_label.setText("Status: Disconnecting...")
            # else: if _shutting_down, LED is already gray and status is "Shutting down..." from closeEvent, so don't change them.

            plotting_paused = True
            self.pause_resume_button.setText("Pause Plotting")
            self.pause_resume_button.setEnabled(False)


    def handle_component_replay_export(self, component_instance: TimeSeriesPlotComponent, start_time_rel: float, end_time_rel: float):
        if self._shutting_down: return
        if state != "replay_active":
            logger.warning("Replay plot export requested but not in replay_active state.")
            QMessageBox.warning(self, "Export Error", "Plot export is only available during active replay.")
            return
            
        logger.info(f"Handling replay export request for component '{component_instance.config.get('title', 'N/A')}' for window [{start_time_rel:.2f}s, {end_time_rel:.2f}s].")

        target_pdf_dir_path = None
        replayed_csv_path_str = getattr(self, 'df_filepath_for_display', None)

        if replayed_csv_path_str:
            replayed_csv_path = Path(replayed_csv_path_str)
            # Try to find a "Logs/SESSION_TIMESTAMP" structure
            parts = replayed_csv_path.parts
            try:
                # Find the index of "Logs". If multiple "Logs" exist, take the last one.
                logs_indices = [i for i, part_name in enumerate(parts) if part_name == "Logs"]
                if logs_indices:
                    logs_idx = logs_indices[-1]
                    if len(parts) > logs_idx + 1: # Ensure there's a part after "Logs" (the session_timestamp_dir)
                        # Path up to and including session_timestamp_dir
                        session_timestamp_dir = Path(*parts[:logs_idx+2]) 
                        target_pdf_dir_path = session_timestamp_dir / "Replay Exports"
                        logger.info(f"Deduced replay export target based on replayed CSV: {target_pdf_dir_path}")
                    else:
                        logger.warning(f"Path '{replayed_csv_path_str}' contains 'Logs' but no subsequent session directory. Using fallback.")
                else: # 'Logs' not in path
                    logger.warning(f"'Logs' directory not found in replayed CSV path '{replayed_csv_path_str}'. Using fallback.")
            except Exception as path_err: # Catch any error during path parsing
                logger.error(f"Error parsing replayed CSV path '{replayed_csv_path_str}' for export: {path_err}. Using fallback.")
        else:
            logger.warning("No replayed CSV path available (df_filepath_for_display not set). Using fallback for export path.")

        if not target_pdf_dir_path: # Fallback logic if path deduction failed or no CSV path
            fallback_base = Path("Logs") / "Replay_Exports_Standalone" # Ensures it's under "Logs"
            timestamp_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            target_pdf_dir_path = fallback_base / f"Export_{timestamp_now}"
            logger.info(f"Using fallback replay export target: {target_pdf_dir_path}")
        
        try:
            os.makedirs(target_pdf_dir_path, exist_ok=True)
            
            # Use the existing method, ensuring it's called correctly for single component export
            self.generate_pdf_plots_from_buffer_for_component(
                str(target_pdf_dir_path), 
                component_instance,
                start_time_rel,
                end_time_rel
            )
            QMessageBox.information(self, "Export Successful", f"Plot exported to:\n{target_pdf_dir_path}")

        except Exception as e:
            logger.error(f"Failed to export replay plot: {e}", exc_info=True)
            QMessageBox.critical(self, "Export Failed", f"Could not export plot:\n{e}")


    def update_scan_status(self, text: str):
         if state == "scanning": self.status_label.setText(text)
    def update_connection_status(self, text: str):
         if state != "connected" and state != "idle": self.status_label.setText(text)
    def show_message_box(self, title: str, message: str):
        QMessageBox.warning(self, title, message)


    async def _load_csvs_for_replay(self, file_paths: List[str], is_initial_load: bool):
        global data_buffers, start_time, plotting_paused # plotting_paused handled by caller at the end

        if is_initial_load:
            logger.info("Initial CSV load: Clearing GUI and data buffers first.")
            self.clear_gui_action(confirm=False) 

        if not file_paths:
            logger.warning("Replay: No file paths provided to _load_csvs_for_replay.")
            if is_initial_load: # If initial load had no files, ensure we are idle.
                 self.handle_state_change("idle")
            return

        successful_load_occurred = False
        for path_idx, path in enumerate(file_paths):
            logger.info(f"Replay: Processing file {path_idx+1}/{len(file_paths)}: {os.path.basename(path)}")
            # --- Overwrite Logic: Clear relevant data_buffers entries before CsvReplaySource loads this file ---
            try:
                # Peek at headers to identify data types in the current CSV
                temp_df_headers = pd.read_csv(path, nrows=0).columns.tolist()
                
                # Identify the actual time column name used in this CSV
                exact_match_time_s = 'Time (s)'
                actual_time_col_in_this_csv = None
                if exact_match_time_s in temp_df_headers:
                    actual_time_col_in_this_csv = exact_match_time_s
                else:
                    for col_header_peek in temp_df_headers:
                        if col_header_peek.lower() == 'time (s)' or col_header_peek.lower() == 'master time (s)':
                            actual_time_col_in_this_csv = col_header_peek
                            break
                
                if not actual_time_col_in_this_csv:
                    logger.warning(f"Skipping CSV {os.path.basename(path)} for overwrite-prep: No recognized time column in headers: {temp_df_headers}")
                else:
                    data_types_in_current_csv = [h for h in temp_df_headers if h != actual_time_col_in_this_csv]
                    logger.debug(f"Data types provided by {os.path.basename(path)}: {data_types_in_current_csv}")
                    for dt_to_overwrite in data_types_in_current_csv:
                        # Normalize the data type key if it's a known variant from CsvReplaySource's renaming
                        # (e.g. if CsvReplaySource renames 'Master Time (s)' for its own use, that's handled internally by it)
                        # For data columns, the names are usually direct.
                        if dt_to_overwrite in data_buffers and data_buffers[dt_to_overwrite]: # Check if buffer has data
                            logger.info(f"Replay: Overwriting data for '{dt_to_overwrite}' using {os.path.basename(path)}.")
                            data_buffers[dt_to_overwrite] = [] 
            except Exception as e:
                logger.error(f"Error peeking at headers for {os.path.basename(path)} to implement overwrite: {e}. Will attempt to load anyway.")

            # --- Load Data using CsvReplaySource ---
            try:
                source = CsvReplaySource(path) 
                self.df_filepath_for_display = path # Update display to show current/last processed file

                await source.start() # This populates global data_buffers and calls compute_all_derived_data
                logger.info(f"Successfully processed data from {os.path.basename(path)}")
                successful_load_occurred = True # Mark that at least one file part loaded
            except ValueError as ve: # Specifically catch ValueError from CsvReplaySource if time col is missing
                logger.error(f"ValueError processing CSV {os.path.basename(path)}: {ve}")
                self.show_message_box("Replay Error", f"Could not load CSV (time column issue?):\n{os.path.basename(path)}\n\nError: {ve}")
                continue # Continue with next file in the batch
            except Exception as e:
                logger.error(f"Failed to process CSV {os.path.basename(path)}: {e}", exc_info=True)
                self.show_message_box("Replay Error", f"Could not load or parse CSV:\n{os.path.basename(path)}\n\nError: {e}")
                continue 

        if not successful_load_occurred and file_paths:
            logger.warning("Replay: No data was successfully loaded into buffers from any of the selected CSVs.")
            if is_initial_load:
                self.handle_state_change("idle") 
            return

        if is_initial_load and successful_load_occurred:
            self.handle_state_change("replay_active")
        

        # Common post-load logic for both initial and "load more"
        global plotting_paused # Ensure we're affecting the global
        logger.info("CSV data batch processed. Setting plotting_paused = True for slider interaction.")



        plotting_paused = True 
        
        # After all files in the batch are processed, first notify components about available data
        # to update/clear overlays based on the final state of data_buffers.




        # First, trigger GUI update for components to process the newly loaded data
        # This allows them to set up sliders, plot data etc., based on data_buffers
        logger.debug("_load_csvs_for_replay: Calling trigger_gui_update for components to process data.")
        self.trigger_gui_update() 

        # Process events to let component updates (like repaints, slider range changes) take effect
        logger.debug("_load_csvs_for_replay: Processing events after component update.")
        QApplication.processEvents() 

        # Now, with data_buffers populated and components potentially initialized from it,
        # notify about missing replay data to set overlay states correctly.
        if hasattr(self, 'gui_manager'):
            logger.debug("_load_csvs_for_replay: Calling notify_missing_replay_data to set overlays.")
            self.gui_manager.notify_missing_replay_data()

        # Process events again to ensure overlay visibility changes are rendered.
        logger.debug("_load_csvs_for_replay: Processing events after overlay update.")
        QApplication.processEvents() 


        
        
        replay_file_path_display = getattr(self, 'df_filepath_for_display', 'Multiple CSVs')
        self.status_label.setText(f"Replay Data Ready. View: {os.path.basename(replay_file_path_display)}. Click 'Exit Replay'.")
        if state == "replay_active":
            self.replay_button.setText("Exit Replay")
            self.replay_button.setEnabled(True)
            self.pause_resume_button.setText("Resume Plotting") # Reflects plotting_paused = True
            self.load_more_csvs_button.setEnabled(True) # Re-enable after loading more


    @qasync.asyncSlot()
    async def handle_load_more_csvs_action(self):
        if self._shutting_down or state != "replay_active":
            logger.warning(f"Load More CSVs action attempted in invalid state: {state} or shutting down.")
            return

        paths, _ = QFileDialog.getOpenFileNames(self, "Select More CSVs to Load", "", "CSV files (*.csv)")
        if not paths: 
            logger.info("Load More CSVs: No files selected.")
            return

        original_status = self.status_label.text()
        self.status_label.setText(f"Loading more CSVs...")
        self.load_more_csvs_button.setEnabled(False)
        self.replay_button.setEnabled(False) # Disable exit while loading more

        await self._load_csvs_for_replay(paths, is_initial_load=False)
        
        # _load_csvs_for_replay now handles final status update and button re-enabling.
        # self.load_more_csvs_button.setEnabled(True) # Done by _load_csvs_for_replay
        self.replay_button.setEnabled(True) # Re-enable exit




    @qasync.asyncSlot()
    async def open_replay_dialog(self):
        global state # current_task is not directly managed here anymore for replay
        if self._shutting_down: return

        if state != "idle":
            self.show_message_box("Replay Error", f"Replay can only be started from 'Idle' state. Current state: {state}.")
            logger.warning(f"Attempted to start replay from non-idle state: {state}")
            return

        paths, _ = QFileDialog.getOpenFileNames(self, "Select Replay CSV(s)", "", "CSV files (*.csv)")
        if not paths:
            logger.info("Replay: No CSV files selected for initial load.")
            return 

        # If somehow a source is active in idle state (should not happen), stop it.
        # This check is mostly for BLE sources, replay source is managed differently now.
        if self.current_source and not isinstance(self.current_source, CsvReplaySource): # Be more specific
            logger.warning("Found an active non-CSV source while in idle state before replay. Stopping it.")
            await self.stop_current_source_and_wait()
        
        
        # Ensure any "Missing UUID" overlays are cleared before entering replay logic
        if hasattr(self, 'gui_manager'):
            self.gui_manager.notify_missing_uuids(set())

        # Set a busy status before starting the potentially long load
        self.status_label.setText("Loading initial CSV(s)...")
        self.replay_button.setEnabled(False) # Disable button during load

        await self._load_csvs_for_replay(paths, is_initial_load=True)
        
        # Re-enable replay button if not already handled by _load_csvs_for_replay or state change
        if state == "idle": # If loading failed and returned to idle
            self.replay_button.setEnabled(True)
        elif state == "replay_active": # If successful
            self.replay_button.setEnabled(True)


        
    def _initiate_source_termination(self):
        """
        Initiates the termination of the current data source and its associated task.
        This method is non-blocking. Cleanup and final state changes are handled
        by the task's done_callback (_ble_source_done_callback).
        """
        global current_task # Keep track of the global current_task

        source_to_terminate = self.current_source
        task_to_cancel = current_task # Use the global current_task

        if source_to_terminate and isinstance(source_to_terminate, BleDataSource):
            logger.info(f"Initiating termination for source: {type(source_to_terminate).__name__}")
            # Call the source's stop() method - this should be quick (setting flags)
            # We run it as a fire-and-forget task because BleDataSource.stop() is async
            asyncio.create_task(source_to_terminate.stop())
            # Note: self.current_source will be set to None by _ble_source_done_callback
        else:
            logger.debug("_initiate_source_termination: No current BleDataSource to stop.")

        if task_to_cancel and not task_to_cancel.done():
            task_name = task_to_cancel.get_name() if hasattr(task_to_cancel, 'get_name') else "UnnamedTask"
            logger.info(f"Requesting cancellation of BleDataSource task: {task_name}")
            task_to_cancel.cancel()
            # Note: global current_task will be set to None by _ble_source_done_callback
        elif task_to_cancel and task_to_cancel.done():
            task_name = task_to_cancel.get_name() if hasattr(task_to_cancel, 'get_name') else "UnnamedTask"
            logger.debug(f"BleDataSource task {task_name} was already done when termination initiated.")
        else:
            logger.debug("_initiate_source_termination: No active task to cancel or task already None.")


    @qasync.asyncSlot()
    async def toggle_scan(self): # Keep async if start logic involves await, otherwise can be sync
        global current_task, state, data_buffers, start_time, device_config # current_task is global
        if self._shutting_down: return

        if state == "idle":
            event_loop = asyncio.get_event_loop()
            if event_loop and event_loop.is_running():
                self.clear_gui_action(confirm=False)
                self.handle_state_change("scanning") # GUI updates to "Scanning..."

                self.current_source = BleDataSource(device_config, gui_emitter)
                # current_task is global and will be set here
                current_task = asyncio.create_task(self.current_source.start())
                current_task.set_name(f"BleDataSourceTask_{id(current_task)}") 
                current_task.add_done_callback(self._ble_source_done_callback)
                logger.info(f"BleDataSource task {current_task.get_name()} created and started.")
            else:
                logger.error("Asyncio loop not running!")
                self.show_message_box("Error", "Asyncio loop is not running.")
        
        elif state == "scanning":
            logger.info(f"Stop Scan requested from state: {state}.")
            self._initiate_source_termination()
            # Immediately update GUI. The _ble_source_done_callback will handle
            # the final cleanup of self.current_source and global current_task.
            self.handle_state_change("idle")

        elif state == "connected":
            logger.info(f"Disconnect requested from state: {state}.")
            # Immediately update GUI to "Disconnecting...".
            # The _ble_source_done_callback will eventually transition to "idle".
            self.handle_state_change("disconnecting")
            self._initiate_source_termination()
        


    def _ble_source_done_callback(self, task_future: asyncio.Future):
        global current_task, state 
        task_name = task_future.get_name() if hasattr(task_future, 'get_name') else "UnknownTask"
        try:
            task_future.result() 
            logger.info(f"BleDataSource task '{task_name}' finished normally (e.g. disconnected or scan phase ended).")
        except asyncio.CancelledError:
            logger.info(f"BleDataSource task '{task_name}' was cancelled.")
        except Exception as e:
            logger.error(f"BleDataSource task '{task_name}' finished with error: {e}", exc_info=False) # exc_info=False for brevity unless debugging
            if not self._shutting_down : 
                 self.show_message_box("BLE Error", f"Live connection task '{task_name}' ended with error:\n{e}")
        finally:
            logger.info(f"Executing done_callback finally block for task '{task_name}'.")
            
            # Clear references if this callback is for the task currently known globally
            if current_task is task_future:
                logger.debug(f"Clearing global current_task in done_callback for task '{task_name}'.")
                current_task = None
                # Also clear self.current_source if it corresponds to this task's source
                if self.current_source and isinstance(self.current_source, BleDataSource): # Simple check
                    logger.debug(f"Clearing self.current_source in done_callback for task '{task_name}'.")
                    self.current_source = None
            else:
                other_task_name = current_task.get_name() if current_task and hasattr(current_task, 'get_name') else str(current_task)
                logger.warning(f"Done_callback for task '{task_name}', but global current_task is '{other_task_name}'. This might indicate rapid task cycling or a logic issue.")
                # If the completed task's source is still referenced, clear it.
                # This is harder to do safely without more context tracking.
                # For now, if current_task is different, we assume another task is primary.

            # Final state transition to idle if not already there and not shutting down
            # This ensures the GUI reflects the end of BLE activity.
            if state != "idle" and not self._shutting_down:
                logger.info(f"Task '{task_name}' done. Current state is '{state}', transitioning to 'idle'.")
                self.handle_state_change("idle")
            elif state == "idle" and not self._shutting_down:
                logger.info(f"Task '{task_name}' done. State is already 'idle'. Ensuring GUI consistency.")
                self.handle_state_change("idle") # Re-assert to fix button states if needed
            elif self._shutting_down:
                 logger.info(f"Task '{task_name}' done during application shutdown. No further state changes from callback.")


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


    def generate_pdf_plots_from_buffer(self, pdf_dir: str, capture_start_relative_time: float, specific_component_instance: Optional[BaseGuiComponent] = None, specific_end_relative_time: Optional[float] = None):
        global data_buffers
        
        if specific_component_instance and not isinstance(specific_component_instance, TimeSeriesPlotComponent):
            logger.warning(f"PDF export called for specific non-TimeSeriesPlot component {type(specific_component_instance)}. This PDF export is for TimeSeriesPlots only.")
            return

        export_target_description = f"component '{specific_component_instance.config.get('title', type(specific_component_instance).__name__)}'" if specific_component_instance else "all applicable plots"
        time_window_description = f"from {capture_start_relative_time:.3f}s"
        if specific_end_relative_time is not None:
            time_window_description += f" to {specific_end_relative_time:.3f}s"
        
        logger.info(f"Generating PDF for {export_target_description} ({time_window_description}). Dir: {pdf_dir}")

        if not data_buffers: logger.warning("Data buffer empty, skipping PDF generation."); return
        
        try:
            plt.style.use('science')
            plt.rcParams.update({'text.usetex': False, 'figure.figsize': [5.5, 3.5], 'legend.fontsize': 9,
                                 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.titlesize': 11})
        except Exception as style_err:
            logger.warning(f"Could not apply 'science' style: {style_err}. Using default.")
            plt.rcParams.update({'figure.figsize': [6.0, 4.0]})

        gen_success = False
        components_to_process = [specific_component_instance] if specific_component_instance else self.gui_manager.all_components

        for component in components_to_process:
            if not isinstance(component, TimeSeriesPlotComponent):
                continue

            plot_config = component.config
            plot_title_base = plot_config.get('title', 'UntitledPlot')
            datasets = plot_config.get('datasets', [])
            if not datasets: continue

            required_uuids_for_plot = set()
            required_types = component.get_required_data_types()
            for dtype in required_types:
                uuid = self.gui_manager.device_config_ref.get_uuid_for_data_type(dtype)
                if uuid: required_uuids_for_plot.add(uuid)
            
            missing_uuids_for_this_plot = required_uuids_for_plot.intersection(self.gui_manager.active_missing_uuids)
            if missing_uuids_for_this_plot and not (state == "replaying" or plotting_paused): # Allow export from replay/pause even if UUIDs were missing live
                 logger.warning(f"Skipping PDF for plot '{plot_title_base}' as it depends on missing UUID(s): {missing_uuids_for_this_plot} and not in replay/paused export mode.")
                 continue

            fig, ax = plt.subplots()
            plot_title_suffix = f" ({capture_start_relative_time:.1f}s - {specific_end_relative_time:.1f}s)" if specific_end_relative_time is not None else ""
            ax.set_title(plot_title_base + plot_title_suffix)
            ax.set_xlabel(plot_config.get('xlabel', 'Time [s]'))
            ax.set_ylabel(plot_config.get('ylabel', 'Value'))
            
            plot_created_for_component = False
            for dataset_conf in datasets:
                data_type = dataset_conf['data_type']
                if data_type in data_buffers and data_buffers[data_type]:
                    full_data = data_buffers[data_type]
                    
                    
                    
                    
                    
                    plot_data_filtered = [
                        (item[0] - capture_start_relative_time, item[1]) 
                        for item in full_data 
                        if item[0] >= capture_start_relative_time and (specific_end_relative_time is None or item[0] <= specific_end_relative_time)
                    ]

                    if plot_data_filtered:
                        try:
                            times_rel_export_start = [p[0] for p in plot_data_filtered]
                            values = [p[1] for p in plot_data_filtered]
                            ax.plot(times_rel_export_start, values, label=dataset_conf.get('label', data_type), color=dataset_conf.get('color', 'k'), linewidth=1.2)
                            plot_created_for_component = True
                        except Exception as plot_err: logger.error(f"Error plotting {data_type} for PDF '{plot_title_base}': {plot_err}")
            
            if plot_created_for_component:
                ax.legend(); ax.grid(True, linestyle='--', alpha=0.6); fig.tight_layout(pad=0.5)
                safe_suffix = component.get_log_filename_suffix()
                prefix = f"{self.capture_timestamp}_" if self.capture_timestamp and not specific_component_instance else "" # No capture timestamp for slider export
                pdf_filename = f"{prefix}{safe_suffix}.pdf"
                pdf_filepath = os.path.join(pdf_dir, pdf_filename)
                try: 
                    fig.savefig(pdf_filepath, bbox_inches='tight')
                    logger.info(f"Saved PDF: {pdf_filename}")
                    gen_success = True
                except Exception as save_err: 
                    logger.error(f"Error saving PDF {pdf_filename}: {save_err}")
                    # raise RuntimeError(f"Save PDF failed: {save_err}") from save_err # Don't raise, just log
            else: logger.info(f"Skipping PDF '{plot_title_base}' (no data in specified time window).")
            plt.close(fig)
        
        if gen_success: logger.info(f"PDF generation finished for {export_target_description}. Dir: {pdf_dir}")
        else: logger.warning(f"PDF generation done for {export_target_description}, but no plots were saved.")


    def generate_pdf_plots_from_buffer_for_component(self, pdf_dir: str, component_to_export: BaseGuiComponent, capture_start_relative_time: float, capture_end_relative_time: float):
        self.generate_pdf_plots_from_buffer(pdf_dir, capture_start_relative_time, specific_component_instance=component_to_export, specific_end_relative_time=capture_end_relative_time)



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
        global data_buffers, start_time, opt_cum_x, opt_cum_y
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

            # reset optical-flow accumulators:
            opt_cum_x = 0
            opt_cum_y = 0
            logger.info(f"Optical‐flow accumulators reset → opt_cum_x={opt_cum_x}, opt_cum_y={opt_cum_y}")

            # Determine clear mode based on current state
            # Live BLE states: "connected", "scanning", "disconnecting"
            is_live_ble_state = state in ["connected", "scanning", "disconnecting"]
            
            if is_live_ble_state:
                logger.info("Clear GUI called in live BLE state. Clearing connected components, preserving UUID overlays for unconnected.")
                self.gui_manager.clear_components(clear_only_connected_data=True)
            else: # Idle, replay_active, or other non-live states
                logger.info("Clear GUI called in non-live state. Clearing all components fully.")
                self.gui_manager.clear_components(clear_only_connected_data=False)


            if self.current_source:
                 logger.info(f"Clear GUI Action: Found active source {type(self.current_source).__name__}. It will be orphaned. Consider stopping it first if it's an ongoing task.")
                 
                 


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
    
    # ==== Help window launcher ====
    def show_help_window(self):
        if not hasattr(self, "_help_window") or self._help_window is None:
            self._help_window = HelpWindow(self)
        self._help_window.show()
        self._help_window.raise_()
        self._help_window.activateWindow()

    # --- Close Event Handling ---

    def closeEvent(self, event):
        global stop_flag
        if self._shutting_down:
            logger.debug("closeEvent: Already shutting down, accepting event.")
            event.accept()
            return

        reply = QMessageBox.question(self, 'Confirm Quit',
                                     "Are you sure you want to quit?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)



        if reply == QMessageBox.StandardButton.Yes:
            logger.info("User confirmed quit. Initiating shutdown sequence.")
            self._shutting_down = True # Set this flag IMPORTANTLY before calling handle_state_change
            stop_flag = True         # Signal all async loops to stop

            # --- Immediate GUI Feedback: Transition to Idle state, then override status ---
            self.handle_state_change("idle") # Let GUI elements go to their idle state
            
            # NOW, specifically set the shutdown message and disable relevant buttons for quitting
            self.status_label.setText("Shutting down...") # Override status set by idle
            self.scan_button.setText("Quitting...")     # scan_button text will be "Start Scan" from idle, override
            self.scan_button.setEnabled(False)          # idle state enables it, so disable again
            self.replay_button.setEnabled(False)        # idle state enables it, so disable again
            self.device_combo.setEnabled(False)         # idle state enables it, so disable again

            # Clear button might be enabled by idle, ensure it's off during pure shutdown
            self.clear_button.setEnabled(False)
            self.led_indicator.set_color("gray") # Explicitly set LED to gray for shutdown visual

            QApplication.processEvents() # Force GUI update with the "Shutting down..." overrides

            # --- Start Background Shutdown Operations ---
            event.ignore()  # We will handle the actual app quit in async_shutdown_operations
            asyncio.create_task(self.async_shutdown_operations())
        else:
            logger.info("User cancelled quit.")
            event.ignore() # Do not close the window


    async def async_shutdown_operations(self):
        global current_task # 'client' is not global, it's managed by BleDataSource
        logger.info("Async shutdown: Starting...")

        # 1. Signal the current source to stop (sets _stop_requested in BleDataSource)
        if self.current_source:
            logger.info(f"Async shutdown: Signalling current source {type(self.current_source).__name__} to stop...")
            try:
                # BleDataSource.stop() is async, so await it.
                # This call should be quick as it mostly sets flags.
                await self.current_source.stop()
                logger.info(f"Async shutdown: Current source {type(self.current_source).__name__} stop signal sent.")
            except Exception as e:
                logger.error(f"Async shutdown: Error signalling current source to stop: {e}")
            # self.current_source will be fully cleared by the task's done_callback or later if task is None

        # 2. Cancel and await the main BLE task (if any)
        # Its 'finally' block in BleDataSource.start() will now execute with _stop_requested=True
        if current_task and not current_task.done():
            logger.info("Async shutdown: Requesting cancellation and awaiting active BLE task...")
            if not current_task.cancelled(): # Check if not already cancelled
                await self.cancel_and_wait_task(current_task)
                logger.info("Async shutdown: Active BLE task cancellation and wait completed.")
            else:
                logger.info("Async shutdown: Active BLE task was already cancelled prior to awaiting.")
        elif current_task and current_task.done():
            logger.info("Async shutdown: Active BLE task was already done.")
        else:
            logger.info("Async shutdown: No active BLE task to cancel or await.")
        
        # 3. Ensure self.current_source is None if its task is handled or was never there.
        # The _ble_source_done_callback should set self.current_source to None if it handled the task.
        # This is a fallback if current_task was None/done but self.current_source somehow persisted.
        if self.current_source:
            logger.warning(f"Async shutdown: self.current_source ({type(self.current_source).__name__}) still exists after task handling. Forcing clear.")
            self.current_source = None
        
        logger.info("Async shutdown: Performing GUI cleanup...")

        # Hide the main window now that BLE operations are complete or timed out.
        logger.info("Async shutdown: Hiding main window.")
        if self.isVisible(): # Check if it's not already hidden for some reason
            self.hide()
        QApplication.processEvents() # Allow hide event to process



        self.plot_update_timer.stop()
        self.scan_throbber_timer.stop()

        # Remove and close the QtLogHandler first
        if self.log_handler:
            logger.info("Async shutdown: Removing GUI log handler...")
            root_logger = logging.getLogger()
            if self.log_handler in root_logger.handlers:
                root_logger.removeHandler(self.log_handler)
            self.log_handler.close() # Calls logging.Handler.close()
            self.log_handler = None
            logger.info("Async shutdown: GUI log handler removed and closed.")

        # Perform standard logging shutdown BEFORE Qt application fully quits.
        # This allows other standard handlers (like StreamHandlers) to flush properly.
        # Our QtLogHandler was already removed from the root logger, but logging.shutdown()
        # will process it if it's still in the global list, so it must be called while Qt is live.
        logger.info("Async shutdown: Calling logging.shutdown()...")
        logging.shutdown()
        logger.info("Async shutdown: logging.shutdown() completed.")
        
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

    # request aliasing
    fmt = QSurfaceFormat()
    fmt.setSamples(4)  # Request MSAA x4
    QSurfaceFormat.setDefaultFormat(fmt)

    # main application setup
    app = QApplication(sys.argv)
    qasync_loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(qasync_loop)
    main_window = MainWindow()
    main_window.show()
    exit_code = 0 
    try:
        logger.info("Starting qasync event loop...")
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
        
        logger.info(f"Application exiting with code {exit_code}.")
        sys.exit(exit_code)

# <<< END OF FILE >>>