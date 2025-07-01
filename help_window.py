# help_window_gui.py

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTabWidget, QWidget, QTextBrowser,
    QSplitter, QTreeWidget, QTreeWidgetItem, QHBoxLayout
)
from PyQt6.QtGui import QFont, QColor, QFontInfo
from PyQt6.QtCore import Qt
from typing import Dict


class HelpWindow(QDialog):
    """
    Modal guide window with enhanced explanations and navigation.
    - GUI Tab: Explains user interface controls.
    - Code Tab: Provides a detailed guide to customizing the application's source code,
                structured as a navigable tree.
    """
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Application Guide")
        self.resize(1200, 700)  # Increased size for better readability

        self.main_layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)

        # --- GUI TAB ---
        self._setup_gui_tab()

        # --- CODE TAB ---
        self._setup_code_tab()

        self.setLayout(self.main_layout)

        # Store help content: maps string content_key to HTML string
        self._help_content_map: Dict[str, str] = {}
        self._populate_code_help_tree()

        # Select the first item in the code help tree by default
        if hasattr(self, 'code_help_tree') and self.code_help_tree.topLevelItemCount() > 0:
            first_item = self.code_help_tree.topLevelItem(0)
            self.code_help_tree.setCurrentItem(first_item)
            # Manually call the selection handler for the first item if setCurrentItem doesn't trigger it
            self._on_code_help_item_selected(first_item, 0)


    def _setup_gui_tab(self):
        gui_tab_widget = QWidget()
        gui_layout = QVBoxLayout(gui_tab_widget)

        gui_text_browser = QTextBrowser() # Using QTextBrowser for rich text
        gui_text_browser.setReadOnly(True)
        gui_text_browser.setOpenExternalLinks(True) # If you add any links

        # Enhanced GUI Guide Content (using HTML for structure)
        # Enhanced GUI Guide Content (using HTML for structure)
        gui_html_content = """
        <html><body>
        <h1>Application GUI Guide</h1>

        <h2>Overview</h2>
        <p>This guide explains the various controls and displays available in the application's graphical user interface (GUI).</p>

        <h2>Top Control Bar</h2>
        <table border="0" cellpadding="5" style="width:100%;">
            <tr><td style="width:25%;"><b>LED (leftmost)</b></td><td>Indicates application status:
                                                <ul><li><font color="red"><b>Red</b></font>: Idle or Disconnected.</li>
                                                    <li><font color="orange"><b>Orange</b></font>: Scanning for or disconnecting from a BLE device.</li>
                                                    <li><font color="lightgreen"><b>Green</b></font>: Connected to a BLE device (Live Mode).</li>
                                                    <li><b style="color:#9B59B6;">Purple</b>: Replay Mode active.</li></ul></td></tr>
            <tr><td><b>"Target" Dropdown</b></td><td>Allows selection of the target BLE device name to connect to. This list is populated from the <code>AVAILABLE_DEVICE_NAMES</code> variable in the code. (Visible in Live Mode only).</td></tr>
            <tr><td><b>"Start Scan" / "Stop Scan" / "Disconnect" Button</b></td><td>Manages the BLE connection lifecycle. (Visible in Live Mode only).</td></tr>
            <tr><td><b>"Replay CSV..." / "Exit Replay" Button</b></td><td>Manages Replay Mode:
                                                <ul><li><b>"Replay CSV..."</b> (when idle): Opens a file dialog to select one or more CSV files to load for data replay.</li>
                                                    <li><b>"Exit Replay"</b> (when in replay mode): Stops the replay session, clears all loaded data, and returns the application to the idle state.</li></ul></td></tr>
            <tr><td><b>"Load More CSVs" Button</b></td><td>Appears only during Replay Mode. Allows you to load additional CSV files. Data from these new files will <b>overwrite</b> data for any matching data types already loaded.</td></tr>
            <tr><td><b>"Pause Plotting" / "Resume Plotting" Button</b></td><td>Toggles the live updating of all plot components. Data collection continues in the background even when plotting is paused. (Visible in Live Mode only).</td></tr>
            <tr><td><b>"Start Capture" / "Stop Capture & Export" Button</b></td><td>Manages data logging sessions during live operation. (Visible in Live Mode only).</td></tr>
            <tr><td><b>"Clear GUI" Button</b></td><td>Wipes all data from internal buffers, resets all plots, and clears any "UUID Missing" overlays. If a capture is active, it will be stopped <b>without</b> exporting files. (Visible in Live Mode only).</td></tr>
            <tr><td><b>Status Label (rightmost)</b></td><td>Displays real-time information about the application's state, such as "Scanning...", "Connected", or "Replay Data Ready".</td></tr>
        </table>

        <h2>Replay Mode & Controls</h2>
        <p>Replay Mode allows you to load and analyze previously captured CSV data without a live BLE connection.</p>
        <ul>
            <li><b>Entering Replay Mode:</b> Click the <b>"Replay CSV..."</b> button when the application is idle. You can select one or multiple CSV files. The application expects CSVs to have a time column named 'Time (s)' or 'Master Time (s)'.</li>
            <li><b>Replay Sliders:</b> Once data is loaded, most visual components will display a time slider.
                <ul>
                    <li><b>Time Series Plots:</b> Show a <b>range slider</b>. Drag the handles to select a time window to view. The plot will zoom to this window.</li>
                    <li><b>Heatmap, Nyquist, 3D IMU Plots:</b> Show a <b>single-handle slider</b>. Drag the handle to "scrub" to a specific point in time. The component will render its state for that exact moment, including reconstructing historical trails for the CoP and Nyquist plots.</li>
                </ul>
            </li>
            <li><b>Exporting from Replay:</b> In Replay Mode, Time Series Plots have an <b>"Export Visible Plot to PDF"</b> button. This exports only the time window currently selected by the range slider to a PDF file.</li>
            <li><b>Loading More Data:</b> The <b>"Load More CSVs"</b> button lets you add data from other CSV files to the current replay session. If a new CSV contains data types that are already loaded, the old data for those types will be replaced by the new data. This is useful for combining data from different individual log files from the same capture session.</li>
            <li><b>Exiting Replay Mode:</b> Click the <b>"Exit Replay"</b> button. This will clear all loaded data and return the application to its normal "idle" state, ready for a live BLE connection or a new replay session.</li>
        </ul>

        <h2>Main Tab Area</h2>
        <p>The central area of the GUI is occupied by tabs. Each tab can contain one or more GUI components arranged in a grid. Component-specific controls (e.g., heatmap sensitivity) are found within the component itself.</p>

        <h2>Bottom Control Bar</h2>
        <table border="0" cellpadding="5" style="width:100%;">
            <tr><td style="width:25%;"><b>"Flowing Mode" Checkbox</b></td><td>(Live Mode only) When checked, the X-axes of time-series plots scroll dynamically, showing a sliding window of data. The duration of this window is set by 'Interval (s)'.</td></tr>
            <tr><td><b>"Interval (s)" Textbox</b></td><td>(Live Mode only) Numeric input for the width (in seconds) of the flowing window.</td></tr>
            <tr><td><b>"Apply Interval" Button</b></td><td>(Live Mode only) Applies the new interval value.</td></tr>
            <tr><td><b>"Log Raw Data to Console" Checkbox</b></td><td>Toggles detailed logging of raw sensor data to the application's console (not the GUI log panel). Useful for debugging data parsing.</td></tr>
            <tr><td><b>"Help?" Button</b></td><td>Opens this guide window.</td></tr>
        </table>

        <h2>Log Panel (Bottom)</h2>
        <p>A text area at the very bottom of the window that displays log messages from the application's main logger, useful for monitoring status, warnings, and errors.</p>

        </body></html>
        """
        
        
        gui_text_browser.setHtml(gui_html_content)
        gui_layout.addWidget(gui_text_browser)
        self.tab_widget.addTab(gui_tab_widget, "GUI Guide")



    def _setup_code_tab(self):
        code_tab_widget = QWidget()
        code_layout = QHBoxLayout(code_tab_widget)

        self.code_help_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.code_help_tree = QTreeWidget()
        self.code_help_tree.setHeaderLabel("Code Customization Topics")
        self.code_help_tree.setMinimumWidth(280)
        self.code_help_tree.setMaximumWidth(450)

        self.code_help_display = QTextBrowser()
        self.code_help_display.setReadOnly(True)

        # Attempt to set a good monospaced font
        font = QFont("Consolas", 10) # Preferred monospaced font
        # Check if the system resolved "Consolas" to something actually monospaced
        # QFontInfo can give more details about the resolved font
        font_info = QFontInfo(font)
        if font_info.fixedPitch():
            self.code_help_display.setFont(font)
        else:
            # Fallback if "Consolas" wasn't fixed pitch (e.g., not installed or bad substitution)
            font.setFamily("Courier New") # Common fallback
            font.setPointSize(10)
            font_info_fallback = QFontInfo(font)
            if font_info_fallback.fixedPitch():
                self.code_help_display.setFont(font)
            else:
                # If Courier New also fails, try a generic monospace hint
                font.setStyleHint(QFont.StyleHint.Monospace)
                font.setFamily("Monospace") # Generic family name
                font.setPointSize(10)
                self.code_help_display.setFont(font)
                # At this point, we've done our best; Qt will pick *something*
        
        self.code_help_display.setOpenExternalLinks(True)

        self.code_help_splitter.addWidget(self.code_help_tree)
        self.code_help_splitter.addWidget(self.code_help_display)
        self.code_help_splitter.setStretchFactor(0, 0) 
        self.code_help_splitter.setStretchFactor(1, 1) 

        code_layout.addWidget(self.code_help_splitter)
        self.tab_widget.addTab(code_tab_widget, "Code Customization")

        self.code_help_tree.itemClicked.connect(self._on_code_help_item_selected)



    def _add_help_topic(self, text: str, parent_item: QTreeWidgetItem | QTreeWidget | None = None, content_key: str = "") -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent_item, [text])
        item.setData(0, Qt.ItemDataRole.UserRole, content_key) # Store the string key with the item
        return item

    def _get_formatted_code_block(self, code_str: str) -> str:
        escaped_code = code_str.replace("&", "&").replace("<", "<").replace(">", ">")
        # Added some line height and slight color adjustments for better readability
        return (f"<pre style='background-color: #f4f4f8; border: 1px solid #ddd; padding: 12px; "
                f"border-radius: 5px; line-height: 1.4em; color: #333; overflow-x: auto;'>"
                f"<code>{escaped_code}</code></pre>")

    def _populate_code_help_tree(self):
        # General Overview
        overview_key = "overview"
        self._add_help_topic("Overview", self.code_help_tree, overview_key)
        overview_text_html = """
        <h2>Overview of Code Structure</h2>
        <p>The application's Python script is broadly divided into two main parts:</p>
        <ol>
            <li><b>Customizable Section:</b> This is where you'll make most of your modifications to adapt the application to new sensors, data processing logic, or GUI layouts. It's found near the top of the script, demarcated by comments like <code># Start of customizable section</code> and <code># End of customizable section</code>.</li>
            <li><b>Backend/Core Logic:</b> This section contains the underlying framework for BLE communication, GUI management, data buffering, and other core functionalities. While powerful, changes here require a deeper understanding of the system and are generally not needed for typical customizations. It's usually marked with a "<i>don’t-touch-unless-you-know-what-you’re-doing</i>" type of warning.</li>
        </ol>
        <p>The customization process usually involves interacting with specific, well-defined parts of the customizable section.</p>
        """
        self._help_content_map[overview_key] = overview_text_html

        # Core Customization Areas
        core_key = "core_customization"
        core_item = self._add_help_topic("Core Customization Areas", self.code_help_tree, core_key)
        self._help_content_map[core_key] = "<h3>Select a sub-topic to see details.</h3><p>This section outlines the primary Python objects and structures you'll modify when customizing the application.</p>"

        data_handlers_key = "data_handlers"
        self._add_help_topic("1. Data Handlers (`handle_*)", core_item, data_handlers_key)
        data_handlers_text_html = """
        <h3>1. Data Handlers (<code>handle_*</code> functions)</h3>
        <p><b>Purpose:</b> Data handlers are Python functions responsible for parsing raw byte arrays received from BLE characteristics into meaningful, structured data.</p>
        <p><b>Location:</b> Defined in the customizable section, under a comment like <code># 1. --- Data Handlers for Different Characteristics ---</code>.</p>
        <p><b>Function Signature:</b></p>
        """ + self._get_formatted_code_block("def handle_sensor_xyz_data(data: bytearray) -> Dict[str, float]:\n    # Your parsing logic here\n    # return {'data_type_1': value1, 'data_type_2': value2}") + """
        <ul>
            <li><b>Input:</b> <code>data (bytearray)</code> - The raw bytes from the BLE notification.</li>
            <li><b>Output:</b> <code>Dict[str, float]</code> - A dictionary where:
                <ul>
                    <li>Keys (strings) are unique identifiers for each piece of data extracted (these become <code>data_type</code>s, e.g., <code>'temperature_celsius'</code>, <code>'pressure_value'</code>).</li>
                    <li>Values (floats, or other scalars like int or bool if appropriate, though float is common for sensor data) are the processed numerical data.</li>
                </ul>
            </li>
        </ul>
        <p><b>Processing Examples:</b></p>
        <ul>
            <li><b>Text Data:</b> <code>text = data.decode("utf-8").strip(); parts = text.split(',')</code> (as seen in <code>handle_orientation_data</code>).</li>
            <li><b>Binary Data (<code>struct</code> module):</b> For fixed-format binary data (e.g., floats, integers). Example: <code>value1, value2 = struct.unpack('<ff', data)</code> for two little-endian floats. Check Python's <code>struct</code> module documentation for format strings.</li>
            <li><b>Custom Binary Parsing:</b> Direct byte manipulation (e.g., <code>int.from_bytes(data[0:2], 'little')</code>).</li>
        </ul>
        <p><b>Logging:</b> Use <code>data_logger.info(f"...")</code> or <code>data_logger.debug(f"...")</code> within handlers for debugging data parsing. This output can be toggled in the GUI via "Log Raw Data to Console".</p>
        <p><b>Example (Conceptual):</b></p>
        """ + self._get_formatted_code_block("""
def handle_temp_humidity_data(data: bytearray) -> dict:
    try:
        # Assuming 2 bytes for temp (scaled by 100), 2 for humidity (scaled by 100)
        # Example: data = b'\\x19\\x00\\x32\\x00' -> temp=25, humidity=50
        raw_temp = int.from_bytes(data[0:2], 'little')
        raw_humidity = int.from_bytes(data[2:4], 'little')
        
        temperature_c = raw_temp / 100.0
        humidity_percent = raw_humidity / 100.0
        
        data_logger.info(f"Temp: {temperature_c}°C, Humidity: {humidity_percent}%")
        return {
            "temperature_celsius": temperature_c,
            "humidity_percentage": humidity_percent
        }
    except Exception as e:
        data_logger.error(f"Error parsing temp/humidity data: {e}")
        return {}
""") + """
        <p><b>Error Handling:</b> Handlers <em>must</em> gracefully handle parsing errors (e.g., using <code>try-except</code> blocks for <code>IndexError</code>, <code>ValueError</code>, <code>struct.error</code>) and return an empty dictionary (<code>{}</code>) on failure. This prevents the entire application from crashing due to malformed BLE data.</p>
        <p><b>Important:</b> The keys in the returned dictionary become the <code>data_type</code> identifiers used throughout the rest of the application (in <code>DeviceConfig</code>, <code>tab_configs</code>, derived data, etc.). Choose descriptive and unique names.</p>
        """
        self._help_content_map[data_handlers_key] = data_handlers_text_html

        derived_data_key = "derived_data"
        self._add_help_topic("2. Derived Data Framework", core_item, derived_data_key)
        derived_data_text_html = """
        <h3>2. Derived Data Framework</h3>
        <p><b>Purpose:</b> Allows you to create new data series by computing them from existing raw or other derived data series. This is useful for sensor fusion, unit conversions, calculations of rates, or any transformation of available data.</p>
        <p><b>Key Components & Workflow:</b></p>
        <ol>
            <li><b>Define a Computation Function:</b>
                <ul>
                    <li><b>Location:</b> Typically in the <code># 2. --- Derived/Fusion Data Handlers ---</code> section.</li>
                    <li><b>Signature:</b> Conventionally <code>def _my_compute_function() -> Optional[float]:</code> (or other scalar type).</li>
                    <li><b>Accessing Dependencies:</b> Reads the latest values of its dependencies from the global <code>data_buffers</code> dictionary. For example, <code>data_buffers.get('raw_sensor_A', [])</code>. Always check if buffers exist and have data.</li>
                    <li><b>Return Value:</b> Returns the computed scalar value. If the value cannot be computed (e.g., a dependency is missing, or not enough data points for a rate calculation), it <em>must</em> return <code>None</code>.</li>
                    <li><b>State (if needed):</b> For calculations requiring history (e.g., derivatives, moving averages), the function can maintain its own state. This is often done using function attributes or closures. See <code>_compute_dZ_dt</code> for an example using a <code>deque</code> as a function attribute to store a history of values.</li>
                </ul>
            </li>
            <li><b>Create a <code>DerivedDataDefinition</code>:</b>
                <ul>
                    <li>This class bundles information about your derived data. An instance is created with:
                        <ul>
                        <li><code>data_type (str)</code>: The unique name for your new (virtual) data series (e.g., <code>'speed_meters_per_second'</code>). This name will be used to access the data in <code>data_buffers</code> and in GUI component configurations.</li>
                        <li><code>dependencies (List[str])</code>: A list of <code>data_type</code> strings that your computation function needs as input. These can be raw types from sensors or other derived types.</li>
                        <li><code>compute_func (Callable)</code>: A reference to the Python computation function you defined in step 1.</li>
                        </ul>
                    </li>
                </ul>
            </li>
            <li><b>Register the Definition:</b>
                <ul>
                    <li><b>Location:</b> In the <code># 3. --- Register the Derived Data Handlers and their Dependencies ---</code> section.</li>
                    <li><b>Action:</b> Call <code>register_derived_data(your_derived_data_definition_instance)</code>. This adds your definition to the global <code>derived_data_definitions</code> registry.</li>
                </ul>
            </li>
        </ol>
        <p><b>Automatic Computation:</b> The backend function <code>compute_all_derived_data(current_relative_time: float)</code> is called automatically whenever new raw data arrives. It iterates through all registered derived data definitions. If all dependencies for a definition are available in <code>data_buffers</code>, its <code>compute_func</code> is called. The result (if not <code>None</code>) is then appended to <code>data_buffers</code> under the derived <code>data_type</code>, using the same timestamp as the triggering raw sample.</p>
        <p><b>UUID Propagation for "Missing UUIDs":</b> The framework intelligently handles "UUID missing" status. If a raw sensor's UUID is not found on the connected device, any derived data that (directly or indirectly) depends on that sensor's data will also be flagged as unavailable, and GUI components relying on it will show the "UUID Missing" overlay.</p>
        <p><b>Example (from your code - <code>_compute_dZ_dt</code>):</b></p>
        """ + self._get_formatted_code_block("""
# Step 1: Computation Function (simplified)
def _compute_dZ_dt(...) -> Optional[float]:
    # ... (initializes history deque if not present) ...
    hist_attr = "_compute_dZ_dt_history"
    if not hasattr(_compute_dZ_dt, hist_attr):
        setattr(_compute_dZ_dt, hist_attr, deque(maxlen=history_maxlen))
    history: deque = getattr(_compute_dZ_dt, hist_attr)

    buf = data_buffers.get('impedance_magnitude_ohm', [])
    if not buf: return None
    t_now, z_now = buf[-1]
    history.append((t_now, z_now))
    # ... (logic to drop old samples, check span, compute slope) ...
    if len(history) < 3 or (history[-1][0] - history[0][0]) < min_span_sec:
        return None
    # ... (least-squares slope calculation) ...
    return float(slope)

# Step 2 & 3: Definition and Registration
register_derived_data(
    DerivedDataDefinition(
        data_type='impedance_change_speed_ohm_per_s',
        dependencies=['impedance_magnitude_ohm'], # Depends on this raw data type
        compute_func=_compute_dZ_dt,
    )
)""") + """
        """
        self._help_content_map[derived_data_key] = derived_data_text_html

        device_config_key = "device_config"
        self._add_help_topic("3. Device Config (`DeviceConfig`)", core_item, device_config_key)
        device_config_text_html = """
        <h3>3. Device Configuration (<code>device_config</code> instance & <code>AVAILABLE_DEVICE_NAMES</code>)</h3>
        <p><b>Purpose:</b> These structures define the BLE device(s) the application can connect to and how their data is interpreted.</p>
        <p><b>Location:</b> Defined in the customizable section, under <code># 4. --- Device Configuration ---</code>.</p>
        
        <h4><code>device_config = DeviceConfig(...)</code></h4>
        <p>This global object (an instance of the <code>DeviceConfig</code> class) holds the configuration for the currently targeted BLE device. Its attributes are:</p>
        <ul>
            <li><code>name (str)</code>: The advertised BLE name of the device (e.g., <code>"Nano33IoT"</code>). This is the <em>initial default name</em>. The GUI's "Target" dropdown (populated by <code>AVAILABLE_DEVICE_NAMES</code>) can update this attribute at runtime via <code>device_config.update_name(selected_name)</code>.</li>
            <li><code>service_uuid (str)</code>: The main BLE Service UUID (128-bit string format) that the application will look for on the target device.</li>
            <li><code>characteristics (List[CharacteristicConfig])</code>: This is a crucial list. Each element is an instance of <code>CharacteristicConfig</code>, defining one BLE characteristic to interact with.</li>
            <li><code>find_timeout (float)</code>: Timeout in seconds for discovering the device during the BLE scanning phase.</li>
            <li><code>data_timeout (float)</code>: Timeout in seconds for receiving data. If connected, but no new data arrives from <em>any</em> subscribed characteristic for this duration, the application may consider the connection lost and attempt to disconnect.</li>
        </ul>

        <h4><code>CharacteristicConfig</code> Class (used within <code>device_config.characteristics</code>)</h4>
        <p>Each <code>CharacteristicConfig</code> instance defines:</p>
        <ul>
            <li><code>uuid (str)</code>: The specific BLE Characteristic UUID (128-bit string format) from which to read data.</li>
            <li><code>handler (Callable)</code>: A direct reference to the data handler function (e.g., <code>handle_orientation_data</code>, <code>handle_my_new_sensor_data</code>) that will be called to process data received from this characteristic.</li>
            <li><code>produces_data_types (List[str])</code>: A list of all <code>data_type</code> string keys that the specified <code>handler</code> function is expected to return in its output dictionary. This mapping is vital for the system to:
                <ul>
                    <li>Know which <code>data_type</code>s originate from which UUID.</li>
                    <li>Correctly handle "UUID Missing" notifications if a characteristic isn't found.</li>
                    <li>Populate the <code>device_config.data_type_to_uuid_map</code> used internally.</li>
                </ul>
            </li>
        </ul>
        <p><b>Example Snippet for <code>device_config</code>:</b></p>
        """ + self._get_formatted_code_block("""
device_config = DeviceConfig(
    name="Nano33IoT",  # Initial default target name
    service_uuid="19B10000-E8F2-537E-4F6C-D104768A1214",
    characteristics=[
        CharacteristicConfig(
            uuid="19B10001-E8F2-537E-4F6C-D104768A1214", # Example: Orientation data
            handler=handle_orientation_data,
            produces_data_types=['orientation_x', 'orientation_y', 'orientation_z']
        ),
        CharacteristicConfig(
            uuid="19B10002-E8F2-537E-4F6C-D104768A1214", # Example: Insole data
            handler=handle_insole_data,
            produces_data_types=HEATMAP_KEYS + ['estimated_weight'] # HEATMAP_KEYS is a list of strings
        ),
        # ... more CharacteristicConfig entries for other sensors/data sources
    ],
    find_timeout=5.0,
    data_timeout=1.0 # Increased to 1.0s in your code for more tolerance
)
""") + """
        <h4><code>AVAILABLE_DEVICE_NAMES = [...]</code></h4>
        <p>This global list of strings (e.g., <code>AVAILABLE_DEVICE_NAMES = ["Nano33IoT", "NanoESP32", "MyOtherDevice"]</code>) populates the "Target" dropdown menu in the GUI's top bar. </p>
        <ul>
            <li>When the user selects a name from this dropdown, the <code>device_config.name</code> attribute is updated to the selected string.</li>
            <li>This allows the application to target different physical devices (which might advertise different names) without needing to change the <code>service_uuid</code> or characteristic UUIDs, assuming they share the same BLE service structure. If they have different service/characteristic UUIDs, you would typically need separate, more complex configurations or a way to switch entire <code>DeviceConfig</code> profiles. For this application's structure, it assumes the selected device name corresponds to a device that will provide the configured service and characteristics.</li>
        </ul>
        <p><b>To Add Support for a New Device Name:</b> Simply add the new advertised name as a string to this list. Example: <code>AVAILABLE_DEVICE_NAMES.append("MyNewBLEPeripheral")</code>.</p>
        """
        self._help_content_map[device_config_key] = device_config_text_html

        gui_components_key = "gui_components"
        self._add_help_topic("4. GUI Component Classes", core_item, gui_components_key)
        gui_components_text_html = """
        <h3>4. GUI Component Classes (Subclasses of <code>BaseGuiComponent</code>)</h3>
        <p><b>Purpose:</b> GUI components are modular PyQt6 widgets responsible for displaying specific types of data visually (e.g., time-series plots, heatmaps, single value readouts, 3D models).</p>
        <p><b>Location:</b> Defined in the customizable section, under <code># 5. --- GUI Component Classes ---</code>.</p>
        
        <h4>Base Class: <code>BaseGuiComponent(QWidget)</code></h4>
        <p>All custom GUI components <em>must</em> inherit from <code>BaseGuiComponent</code>. This base class provides common functionality, an interface for the <code>GuiManager</code>, and features like the "UUID Missing" overlay.</p>
        <p><b>Key Methods to Implement/Override in Your Subclasses:</b></p>
        <ul>
            <li><code>__init__(self, config: Dict[str, Any], data_buffers_ref: Dict, device_config_ref: DeviceConfig, parent: Optional[QWidget] = None)</code>:
                <ul>
                    <li><strong>Crucial:</strong> Call <code>super().__init__(config, data_buffers_ref, device_config_ref, parent)</code>.</li>
                    <li><code>config</code>: A dictionary passed from this component's definition in <code>tab_configs</code>. Use it for per-instance customization (e.g., plot titles, data types to plot, colors, initial settings like heatmap sensitivity). Example: <code>self.my_setting = config.get('my_custom_setting', default_value)</code>.</li>
                    <li><code>data_buffers_ref</code>: A reference to the global <code>data_buffers</code> dictionary. Store it (e.g., <code>self.data_buffers_ref = data_buffers_ref</code>) to access sensor data in <code>update_component</code>.</li>
                    <li><code>device_config_ref</code>: A reference to the global <code>device_config</code> instance. Store it if needed, e.g., to look up UUIDs for data types.</li>
                    <li>Set up the component's UI elements (e.g., <code>pg.PlotWidget()</code>, <code>QLabel()</code>, <code>gl.GLViewWidget()</code>). Add them to the component's layout.</li>
                </ul>
            </li>
            <li><code>get_widget(self) -> QWidget</code>: Returns the primary QWidget that this component manages. Often, this is just <code>self</code> if the component itself is the main widget. For complex components, it might be a specific child widget. (Default in <code>BaseGuiComponent</code> is <code>return self</code>).</li>
            <li><code>get_required_data_types(self) -> Set[str]</code>: <strong>Essential.</strong> Must return a Python <code>set</code> of <code>data_type</code> strings that this component absolutely needs to function and display its information (e.g., <code>{'temperature_celsius', 'humidity_percent'}</code>). This set is used by the <code>GuiManager</code> to determine if any required BLE characteristics (UUIDs) are missing for this component.</li>
            <li><code>update_component(self, current_relative_time: float, is_flowing: bool)</code>: <strong>Essential.</strong> Called periodically by <code>GuiManager</code> (via a timer) to refresh the component's display.
                <ul>
                    <li>Check the global <code>plotting_paused</code> variable; if <code>True</code>, usually just return without updating visuals.</li>
                    <li>Fetch the latest relevant data from <code>self.data_buffers_ref</code> using the <code>data_type</code>s it depends on. Remember <code>data_buffers</code> stores lists of <code>(timestamp, value)</code> tuples.</li>
                    <li>Update the component's visual representation (e.g., update plot lines, redraw heatmap, change label text).</li>
                    <li><code>is_flowing (bool)</code>: Indicates if time-series plots should use a sliding time window (relevant for <code>TimeSeriesPlotComponent</code>).</li>
                </ul>
            </li>
            <li><code>clear_component(self)</code>: <strong>Essential.</strong> Resets the component's display and any internal state to an initial/empty condition. Called by <code>GuiManager</code> when the "Clear GUI" button is pressed or during some state transitions. It should also refresh/clear any "UUID Missing" overlay by calling <code>super().handle_missing_uuids(set())</code>.</li>
        </ul>
        <p><b>Optional Logging-Related Methods (from <code>BaseGuiComponent</code>, can be relied upon or overridden):</b></p>
        <ul>
            <li><code>self.is_loggable (bool)</code>: Automatically set in <code>BaseGuiComponent.__init__</code> based on <code>config.get('enable_logging', False)</code>. Indicates if this component instance should be considered for CSV logging.</li>
            <li><code>get_loggable_data_types(self) -> Set[str]</code>: If <code>self.is_loggable</code> is <code>True</code>, this method returns a set of <code>data_type</code> strings this component wants to log. The default implementation in <code>BaseGuiComponent</code> returns the same set as <code>get_required_data_types()</code>. Override this in your subclass if you want to log a different set of data types than those strictly required for display (e.g., log raw inputs to a calculation component, or log fewer types).</li>
            <li><code>get_log_filename_suffix(self) -> str</code>: If <code>self.is_loggable</code> is <code>True</code>, this returns a string suffix used to create a unique CSV filename for this specific component instance (e.g., <code>"plot_temperature_log"</code>). The base implementation generates a suffix from the class name or the <code>'title'</code> in its <code>config</code>. Override for more control over the individual CSV filename.</li>
        </ul>
        <p><b>Handling Missing UUIDs (<code>handle_missing_uuids</code>):</b></p>
        <ul>
            <li><code>BaseGuiComponent</code> provides a default <code>handle_missing_uuids</code> method that shows/hides a standard overlay message on the component if its required UUIDs are missing.</li>
            <li>You can override this method in your subclass for custom behavior (e.g., disabling specific controls within your component, showing a different visual indication). If you override it, you might still want to call <code>super().handle_missing_uuids(...)</code> if you want the base overlay functionality too, or manage your own visual cues entirely.</li>
        </ul>
        <p><b>Existing Component Examples in Your Code:</b></p>
        <ul>
            <li><code>TimeSeriesPlotComponent</code>: For line graphs of data over time.</li>
            <li><code>PressureHeatmapComponent</code>: Specialized for displaying 2D pressure distribution with extensive controls.</li>
            <li><code>NyquistPlotComponent</code>: For impedance Nyquist plots with trail and snapshot features.</li>
            <li><code>IMUVisualizerComponent</code>: For 3D orientation visualization using OpenGL.</li>
            <li><code>SingleValueDisplayComponent</code>: For showing the latest value of a single data type as text.</li>
        </ul>
        <p>Study these existing components to see practical implementations of the required methods and how they interact with Qt/PyQtGraph widgets and the data buffering system.</p>
        """
        self._help_content_map[gui_components_key] = gui_components_text_html

        tab_layout_key = "tab_configs"
        self._add_help_topic("5. Tab Layout (`tab_configs`)", core_item, tab_layout_key)
        tab_layout_text_html = """
        <h3>5. Tab Layout (<code>tab_configs</code> global variable)</h3>
        <p><b>Purpose:</b> The <code>tab_configs</code> variable is a Python list of dictionaries that defines the entire structure of the GUI's tabbed area. It specifies how many tabs there are, what they are named, which GUI components appear on each tab, their arrangement in a grid, and their individual configurations.</p>
        <p><b>Location:</b> Defined in the customizable section, under <code># 6. --- Tab Layout Configuration ---</code>.</p>
        
        <h4>Overall Structure of <code>tab_configs</code>:</h4>
        <p><code>tab_configs</code> is a <code>List[Dict]</code>. Each dictionary in this list represents one tab in the GUI.</p>
        """ + self._get_formatted_code_block("""
tab_configs = [
    { # Definition for Tab 1
        'tab_title': 'Insole View',
        'layout': [
            # ... list of component definition dictionaries for Tab 1 ...
        ]
    },
    { # Definition for Tab 2
        'tab_title': 'IMU 3D View',
        'layout': [
            # ... list of component definition dictionaries for Tab 2 ...
        ]
    },
    # ... more tab definitions ...
]
""") + """
        <h4>Structure of a Single Tab Definition Dictionary:</h4>
        <p>Each dictionary within <code>tab_configs</code> must contain:</p>
        <ul>
            <li><code>'tab_title' (str)</code>: The text that will be displayed on the tab itself (e.g., <code>'Insole View'</code>).</li>
            <li><code>'layout' (List[Dict])</code>: A list where each element is another dictionary. Each of these inner dictionaries defines a single GUI component to be placed on this tab.</li>
        </ul>

        <h4>Structure of a Component Definition Dictionary (within a tab's <code>'layout'</code> list):</h4>
        <p>This dictionary specifies how a single component instance is created and configured:</p>
        """ + self._get_formatted_code_block("""
# Example of one component definition within a tab's 'layout' list:
{
    'component_class': TimeSeriesPlotComponent,  # REQUIRED: The Python class of the component
    'row': 0,                                  # REQUIRED: Grid row position (0-indexed)
    'col': 0,                                  # REQUIRED: Grid column position (0-indexed)
    'rowspan': 1,                              # Optional: How many rows it spans (default is 1)
    'colspan': 2,                              # Optional: How many columns it spans (default is 1)
    'config': {                                # REQUIRED (can be empty {}): Configuration dictionary passed to the component
        # --- Component-specific config keys below ---
        'title': 'Temperature Data',             # Example for TimeSeriesPlotComponent
        'datasets': [
            {'data_type': 'temp_celsius', 'label': 'Temp °C', 'color': 'red'}
        ],
        'plot_height': 300,                      # Example custom config for some components
        'enable_logging': True                   # Standard config: enable CSV logging for this component instance
        # ... any other key-value pairs expected by TimeSeriesPlotComponent's __init__ ...
    }
}
""") + """
        <ul>
            <li><code>'component_class' (Type[BaseGuiComponent])</code>: <strong>Required.</strong> A direct reference to the component's Python class (e.g., <code>TimeSeriesPlotComponent</code>, <code>PressureHeatmapComponent</code>, <code>MyCustomDataViewer</code>). Do not use a string here.</li>
            <li><code>'row' (int)</code>, <code>'col' (int)</code>: <strong>Required.</strong> The starting cell (0-indexed) in the tab's grid layout where this component will be placed.</li>
            <li><code>'rowspan' (int)</code>, <code>'colspan' (int)</code>: (Optional) Integers specifying how many grid cells the component should occupy vertically and horizontally, respectively. If omitted, they default to <code>1</code>.</li>
            <li><code>'config' (Dict[str, Any])</code>: <strong>Required</strong> (though it can be an empty dictionary <code>{}</code> if the component needs no specific configuration). This dictionary is passed <em>directly</em> to the <code>__init__</code> method of the <code>component_class</code> as its <code>config</code> argument.
                <ul>
                    <li>This is the primary mechanism for customizing individual instances of components.</li>
                    <li>The specific keys and values accepted/expected within this <code>'config'</code> dictionary depend entirely on the implementation of the chosen <code>component_class</code>. For example, <code>TimeSeriesPlotComponent</code> expects keys like <code>'title'</code>, <code>'datasets'</code>, <code>'xlabel'</code>, etc. <code>PressureHeatmapComponent</code> expects keys like <code>'initial_sensitivity'</code>, <code>'image_path'</code>, etc.</li>
                    <li>A standard key understood by <code>BaseGuiComponent</code> is <code>'enable_logging': True/False</code>, which controls whether this specific component instance participates in CSV data logging.</li>
                </ul>
            </li>
        </ul>
        <p>The <code>GuiManager</code> parses <code>tab_configs</code> at startup to dynamically build the entire tabbed interface.</p>
        """
        self._help_content_map[tab_layout_key] = tab_layout_text_html


        # Adding New Features (Scenarios)
        scenarios_key = "scenarios"
        scenarios_item = self._add_help_topic("Adding New Features (Scenarios)", self.code_help_tree, scenarios_key)
        self._help_content_map[scenarios_key] = "<h3>Select a specific scenario to see detailed steps.</h3><p>This section provides step-by-step guides for common customization tasks, referencing the core areas described above.</p>"

        add_sensor_key = "add_sensor"
        self._add_help_topic("Adding a New Sensor/Characteristic", scenarios_item, add_sensor_key)
        add_sensor_text_html = """
        <h3>Scenario: Adding a New Sensor (BLE Characteristic)</h3>
        <p>This guide walks you through integrating data from a new BLE characteristic into the application. This assumes the new characteristic provides data that can be parsed into one or more numerical values (<code>data_type</code>s).</p>

        <h4>Step 1: Implement a Data Handler Function</h4>
        <p><b>Action:</b> In the <code># 1. --- Data Handlers for Different Characteristics ---</code> section of your code, write a new Python function to parse the data from your new sensor.</p>
        <p><b>Details:</b></p>
        <ul>
            <li>Function signature must be: <code>def handle_your_new_sensor_data(data: bytearray) -> Dict[str, float]:</code> (or other scalar types like <code>int</code> if more appropriate for the value).</li>
            <li>Input <code>data</code>: The raw <code>bytearray</code> received from the BLE notification for this characteristic.</li>
            <li>Processing: Decode this <code>bytearray</code> according to your sensor's specific data format. This might involve using <code>struct.unpack()</code> for binary data, <code>data.decode('utf-8')</code> for text-based data, or direct byte manipulation for custom formats.</li>
            <li>Output: Must return a Python dictionary.
                <ul>
                    <li>The keys of this dictionary should be unique string identifiers for each piece of data extracted (e.g., <code>'my_sensor_temperature'</code>, <code>'my_sensor_pressure_pa'</code>). These keys become the <code>data_type</code>s.</li>
                    <li>The values should be the processed numerical data (typically floats or integers).</li>
                </ul>
            </li>
            <li>Error Handling: Crucially, your handler <em>must</em> include robust error handling (e.g., a <code>try-except Exception as e:</code> block). If parsing fails for any reason (e.g., unexpected data length, invalid format), it should log an error using <code>data_logger.error(...)</code> and return an empty dictionary (<code>{}</code>). This prevents a single malformed packet from crashing the application.</li>
            <li>Logging for Debugging: Use <code>data_logger.info(f"Parsed data: ...")</code> or <code>data_logger.debug(...)</code> inside your handler to verify that your parsing logic is correct. This output can be viewed in the console if "Log Raw Data to Console" is checked in the GUI.</li>
        </ul>
        <p><b>Example (parsing two little-endian floats):</b></p>
        """ + self._get_formatted_code_block("""
# In section: # 1. --- Data Handlers for Different Characteristics ---
import struct # Make sure struct is imported

def handle_my_new_bme_sensor(data: bytearray) -> dict:
    try:
        # Assuming the sensor sends:
        # - 2 bytes: temperature (int16, scaled by 100, little-endian)
        # - 2 bytes: pressure (uint16, in Pascals, little-endian)
        # - 1 byte:  humidity (uint8, percentage)
        # Total: 5 bytes expected
        if len(data) < 5:
            data_logger.warning(f"MyNewBMESensor: Data too short ({len(data)} bytes), expected 5.")
            return {}

        # '<' for little-endian, 'h' for short (int16), 'H' for unsigned short (uint16), 'B' for unsigned char (uint8)
        raw_temp, raw_pressure, raw_humidity = struct.unpack("<hHB", data[:5])
        
        temperature = raw_temp / 100.0  # Degrees Celsius
        pressure_pa = float(raw_pressure)   # Pascals
        humidity_pct = float(raw_humidity)  # Percentage

        data_logger.info(f"MyNewBMESensor: Temp={temperature:.2f}°C, Press={pressure_pa:.0f}Pa, Hum={humidity_pct:.0f}%")
        return {
            "bme_temperature_celsius": temperature,
            "bme_pressure_pascal": pressure_pa,
            "bme_humidity_percent": humidity_pct
        }
    except struct.error as e:
        data_logger.error(f"MyNewBMESensor: struct.unpack error: {e} - Data: {data.hex()}")
        return {}
    except Exception as e:
        data_logger.error(f"MyNewBMESensor: General error parsing data: {e} - Data: {data.hex()}")
        return {}
""") + """

        <h4>Step 2: Register the Characteristic in <code>DeviceConfig</code></h4>
        <p><b>Action:</b> Locate the <code>device_config = DeviceConfig(...)</code> instantiation in the <code># 4. --- Device Configuration ---</code> section.</p>
        <p><b>Details:</b> Add a new <code>CharacteristicConfig</code> object to the <code>characteristics</code> list within the <code>DeviceConfig</code> constructor arguments.</p>
        This <code>CharacteristicConfig</code> entry links your new sensor's BLE Characteristic UUID to the data handler function you just wrote and declares all the <code>data_type</code> keys that handler produces.
        <p><b>Example (continuing with <code>MyNewBMESensor</code>):</b></p>
        """ + self._get_formatted_code_block("""
# In section: # 4. --- Device Configuration ---
# Ensure handle_my_new_bme_sensor is defined before this point

device_config = DeviceConfig(
    name="YourDeviceName", # Or use existing if it's the same physical device
    service_uuid="YOUR-MAIN-SERVICE-UUID", # Your device's primary service UUID
    characteristics=[
        # ... other existing CharacteristicConfig entries ...
        CharacteristicConfig(
            uuid="YOUR-NEW-BME-CHARACTERISTIC-UUID",  # The 128-bit UUID string for this new characteristic
            handler=handle_my_new_bme_sensor,         # Reference to the handler function from Step 1
            produces_data_types=[                     # List ALL data_type keys your handler returns
                'bme_temperature_celsius',
                'bme_pressure_pascal',
                'bme_humidity_percent'
            ]
        ),
    ],
    # ... other DeviceConfig parameters (find_timeout, data_timeout) ...
)
""") + """
        <p><strong>Crucial:</strong> The <code>uuid</code> must be the correct 128-bit string for your BLE characteristic. The <code>produces_data_types</code> list must <em>exactly</em> match all the keys in the dictionary returned by your <code>handle_my_new_bme_sensor</code> function.</p>

        <h4>Step 3: Utilize the New `data_type`(s) in a GUI Component</h4>
        <p><b>Action:</b> Go to the <code># 6. --- Tab Layout Configuration ---</code> section (where the global <code>tab_configs</code> list is defined).</p>
        <p><b>Details:</b> Decide where and how you want to display this new sensor data. You can:</p>
        <ul>
            <li>Add a new component (like <code>TimeSeriesPlotComponent</code> or <code>SingleValueDisplayComponent</code>) to an existing tab's <code>'layout'</code> list.</li>
            <li>Create an entirely new tab and add components to it (see "Adding a New Tab" scenario).</li>
        </ul>
        <p>In the component's <code>'config'</code> dictionary, reference your new <code>data_type</code>(s).</p>
        <p><b>Example (adding a plot for the BME temperature and a display for humidity):</b></p>
        """ + self._get_formatted_code_block("""
# In tab_configs, within a chosen tab's 'layout' list:
# (Assume this tab already exists or you're adding a new one)
{
    'component_class': TimeSeriesPlotComponent,
    'row': 0, 'col': 1, # Adjust grid position as needed within the tab
    'config': {
        'title': 'BME Sensor Temperature',
        'xlabel': 'Time [s]',
        'ylabel': 'Temperature (°C)',
        'datasets': [
            {'data_type': 'bme_temperature_celsius', 'label': 'Temperature', 'color': 'orange'}
        ],
        'enable_logging': True # Set to True to include 'bme_temperature_celsius' in CSV exports
    }
},
{
    'component_class': SingleValueDisplayComponent,
    'row': 1, 'col': 1, # Adjust grid position
    'config': {
        'label': 'Current Humidity:', 
        'data_type': 'bme_humidity_percent', # Use one of your new data_types
        'format': '{:.1f}',                 # Format string for the value
        'units': '%',                       # Units to display
        'enable_logging': True            # Also log humidity if desired
    }
}
""") + """

        <h4>Step 4: (Optional) Update <code>AVAILABLE_DEVICE_NAMES</code></h4>
        <p><b>Action:</b> If this new sensor is part of a BLE device that advertises under a new name not already listed in <code>AVAILABLE_DEVICE_NAMES</code>.</p>
        <p><b>Details:</b> Add the new advertised device name string to the <code>AVAILABLE_DEVICE_NAMES</code> list (this list is usually defined near the <code>DeviceConfig</code> definition).</p>
        <p><b>Example:</b> <code>AVAILABLE_DEVICE_NAMES = ["Nano33IoT", "NanoESP32", "MyNewBMEDevice"]</code></p>
        <p>This allows users to select "MyNewBMEDevice" from the "Target" dropdown in the GUI, which will then update <code>device_config.name</code>. The rest of the <code>device_config</code> (service UUID, characteristic UUIDs) will still be used as defined.</p>
        <p>After these changes, restart the application for them to take effect.</p>
        """
        self._help_content_map[add_sensor_key] = add_sensor_text_html

        add_derived_key = "add_derived"
        self._add_help_topic("Adding a New Derived Data Series", scenarios_item, add_derived_key)
        add_derived_text_html = """
        <h3>Scenario: Adding a New Derived Data Series</h3>
        <p>This explains how to create a new data series that is calculated from one or more existing (raw sensor or other derived) data series. This is useful for calculations like sensor fusion, unit conversions, rates of change, etc.</p>

        <h4>Step 1: Write a Computation Function</h4>
        <p><b>Action:</b> In the <code># 2. --- Derived/Fusion Data Handlers ---</code> section of the customizable code, define a new Python function that will perform your desired calculation.</p>
        <p><b>Details:</b></p>
        <ul>
            <li><b>Function Signature:</b> Conventionally, these functions are named with a leading underscore (e.g., <code>def _compute_my_derived_value() -> Optional[float]:</code>). The return type should be <code>Optional[float]</code> or another scalar type (e.g., <code>Optional[int]</code>) if appropriate.</li>
            <li><b>Accessing Input Data (Dependencies):</b> The function should read its input data from the global <code>data_buffers</code> dictionary. Use <code>data_buffers.get('input_data_type_name', [])</code> to safely access the list of <code>(timestamp, value)</code> tuples for each dependency. Always check if the buffer exists and is not empty before trying to access its elements (e.g., <code>if not voltage_buffer or not current_buffer: return None</code>).</li>
            <li><b>Return Value:</b> The function must return the single, calculated scalar value. If the calculation cannot be performed (e.g., due to missing input data in <code>data_buffers</code>, or insufficient data points for a rate calculation), it <em>must</em> return <code>None</code>. The framework will skip adding a <code>None</code> value to the buffers.</li>
            <li><b>Purity and State:</b> Generally, these functions should be "pure" with respect to reading their inputs from <code>data_buffers</code>; they should not modify the input buffers. The framework handles appending the computed result. For calculations requiring a history of values (e.g., derivatives, moving averages), the function can maintain its own state across calls. This is typically done using function attributes (as seen in <code>_compute_dZ_dt</code> which uses a <code>deque</code> attached as an attribute to itself) or closures if preferred.</li>
            <li><b>Debugging:</b> You can use <code>data_logger.debug(...)</code> within the computation function to log intermediate values or results for debugging, visible in the console when "Log Raw Data to Console" is enabled.</li>
        </ul>
        <p><b>Example:</b> Suppose you have raw data types <code>'raw_sensor_A_volts'</code> and <code>'raw_sensor_B_milliamps'</code>, and you want to derive <code>'combined_power_watts'</code>.</p>
        """ + self._get_formatted_code_block("""
# In section: # 2. --- Derived/Fusion Data Handlers ---
from collections import deque # If using history, like in _compute_dZ_dt

def _compute_combined_power() -> Optional[float]:
    # Get the latest voltage
    voltage_data = data_buffers.get('raw_sensor_A_volts', [])
    if not voltage_data:
        # data_logger.debug("Power calculation: Missing voltage data.") # Optional
        return None
    latest_voltage = voltage_data[-1][1]  # Value is the second element of the tuple

    # Get the latest current
    current_data_ma = data_buffers.get('raw_sensor_B_milliamps', [])
    if not current_data_ma:
        # data_logger.debug("Power calculation: Missing current data.") # Optional
        return None
    latest_current_ma = current_data_ma[-1][1]
    
    # Perform calculation (e.g., P = V * I, converting mA to A)
    power_watts = latest_voltage * (latest_current_ma / 1000.0)
    
    # data_logger.debug(f"Computed combined power: {power_watts:.3f} W") # Optional
    return power_watts
""") + """
        <h4>Step 2: Define and Register the <code>DerivedDataDefinition</code></h4>
        <p><b>Action:</b> In the <code># 3. --- Register the Derived Data Handlers and their Dependencies ---</code> section, you need to create an instance of <code>DerivedDataDefinition</code> and then register it.</p>
        <p><b>Details for <code>DerivedDataDefinition</code>:</b></p>
        <ul>
            <li><code>data_type (str)</code>: A new, unique string name for your derived data series (e.g., <code>'combined_power_watts'</code>). This name will be used to store and access this data in <code>data_buffers</code> and in GUI component configurations.</li>
            <li><code>dependencies (List[str])</code>: A list of <code>data_type</code> strings that your computation function (from Step 1) requires as input. These <em>must</em> be existing <code>data_type</code> keys (either from raw sensor handlers or other, previously registered, derived data).</li>
            <li><code>compute_func (Callable)</code>: A direct reference to your computation function (e.g., <code>_compute_combined_power</code>).</li>
        </ul>
        <p><b>Action (Registration):</b> Call <code>register_derived_data(your_definition_instance)</code>.</p>
        <p><b>Example (for the power calculation):</b></p>
        """ + self._get_formatted_code_block("""
# In section: # 3. --- Register the Derived Data Handlers ... ---
# Ensure _compute_combined_power is defined before this point

power_definition = DerivedDataDefinition(
    data_type='combined_power_watts',        # The new data_type name
    dependencies=[
        'raw_sensor_A_volts',                # Dependency 1
        'raw_sensor_B_milliamps'             # Dependency 2
    ],
    compute_func=_compute_combined_power     # Your computation function from Step 1
)
register_derived_data(power_definition)
""") + """
        <h4>Step 3: Utilize the New Derived `data_type` in the GUI</h4>
        <p><b>Action:</b> Now that your derived data (e.g., <code>'combined_power_watts'</code>) is defined and will be automatically computed, you can use its <code>data_type</code> string in any GUI component configuration within <code>tab_configs</code> (section <code># 6. --- Tab Layout Configuration ---</code>), just as you would use a raw sensor <code>data_type</code>.</p>
        <p><b>Example (plotting the derived power on a <code>TimeSeriesPlotComponent</code>):</b></p>
        """ + self._get_formatted_code_block("""
# In tab_configs, within a component's 'config' dictionary:
# This would typically be part of a TimeSeriesPlotComponent definition
'datasets': [
    {'data_type': 'combined_power_watts', 'label': 'Combined Power (W)', 'color': 'green'}
    # ... other datasets for the same plot ...
],
'enable_logging': True # If you want to log this derived data to CSV
""") + """
        <p>The system will automatically call <code>_compute_combined_power</code> whenever new <code>'raw_sensor_A_volts'</code> or <code>'raw_sensor_B_milliamps'</code> data arrives. The calculated result will be available in <code>data_buffers['combined_power_watts']</code> and will be picked up by any GUI component configured to display it.</p>
        <p>Restart the application for these changes to take effect.</p>
        """
        self._help_content_map[add_derived_key] = add_derived_text_html

        add_component_type_key = "add_component_type"
        self._add_help_topic("Adding a New GUI Component Type", scenarios_item, add_component_type_key)
        add_component_type_text_html = """
        <h3>Scenario: Adding a New GUI Component Type</h3>
        <p>If the existing component types (<code>TimeSeriesPlotComponent</code>, <code>PressureHeatmapComponent</code>, <code>SingleValueDisplayComponent</code>, <code>IMUVisualizerComponent</code>, <code>NyquistPlotComponent</code>) don't meet your specific visualization needs, you can create an entirely new type of GUI component.</p>

        <h4>Step 1: Define a New Class Subclassing <code>BaseGuiComponent</code></h4>
        <p><b>Action:</b> In the <code># 5. --- GUI Component Classes ---</code> section of the customizable code, define your new component class. It <em>must</em> inherit from <code>BaseGuiComponent</code>.</p>
        <p><b>Key Methods to Implement in Your New Class:</b></p>
        <ul>
            <li><code><b>__init__(self, config: Dict[str, Any], data_buffers_ref: Dict, device_config_ref: DeviceConfig, parent: Optional[QWidget] = None)</b></code>:
                <ul>
                    <li><strong>Call <code>super().__init__(...)</code> first:</strong> <code>super().__init__(config, data_buffers_ref, device_config_ref, parent)</code>. This initializes base class features like <code>self.is_loggable</code> and the missing UUID overlay system.</li>
                    <li>Access custom parameters from the <code>config</code> dictionary (passed from <code>tab_configs</code>) using <code>self.config.get("my_param_key", default_value)</code>. Store them as instance attributes if needed.</li>
                    <li>Store references: <code>self.data_buffers_ref = data_buffers_ref</code> and <code>self.device_config_ref = device_config_ref</code>.</li>
                    <li>Create and arrange the Qt widgets that make up your component's user interface (e.g., <code>QLabel</code>s, custom drawing widgets like a <code>QWidget</code> subclass with a reimplemented <code>paintEvent</code>, or more complex Qt widgets). Use Qt layouts (<code>QVBoxLayout</code>, <code>QHBoxLayout</code>, <code>QGridLayout</code>) to manage these child widgets within your component. Set the main layout for your component using <code>self.setLayout(...)</code>.</li>
                </ul>
            </li>
            <li><code><b>get_required_data_types(self) -> Set[str]</b></code>:
                <ul>
                    <li><strong>Essential.</strong> Must return a Python <code>set</code> of all <code>data_type</code> strings that your component needs from <code>data_buffers</code> to display its information. This is critical for the "UUID Missing" notification system to work correctly for your component.</li>
                </ul>
            </li>
            <li><code><b>update_component(self, current_relative_time: float, is_flowing: bool)</b></code>:
                <ul>
                    <li><strong>Essential.</strong> This method is called frequently by the <code>GuiManager</code>'s update timer.</li>
                    <li>First, check the global <code>plotting_paused</code> variable. If <code>True</code>, you should generally return immediately without updating the visual display (unless your component needs to show a "Paused" state).</li>
                    <li>Fetch the latest relevant data from <code>self.data_buffers_ref</code> using the <code>data_type</code>s your component depends on (as declared in <code>get_required_data_types</code>). Remember that data in buffers is a list of <code>(timestamp, value)</code> tuples.</li>
                    <li>Update your component's visual elements based on the new data (e.g., change text on a <code>QLabel</code>, trigger a repaint of a custom widget, update data in a chart).</li>
                </ul>
            </li>
            <li><code><b>clear_component(self)</b></code>:
                <ul>
                    <li><strong>Essential.</strong> This method is called when the "Clear GUI" button is pressed or during certain state transitions.</li>
                    <li>Reset your component's visual state to its default or empty state (e.g., clear text in labels, reset graphics, clear internal data stores specific to the component's display).</li>
                    <li><strong>Important:</strong> You should also ensure any "UUID Missing" overlay is cleared. The easiest way is to call <code>super().handle_missing_uuids(set())</code> at the end of your <code>clear_component</code> implementation.</li>
                </ul>
            </li>
        </ul>
        <p><b>Optional Methods to Override (if default behavior of <code>BaseGuiComponent</code> is not sufficient):</b></p>
        <ul>
            <li><code>get_widget(self) -> QWidget</code>: Default returns <code>self</code>. Override if your component's main display widget is a child, not the component itself.</li>
            <li><code>get_loggable_data_types(self) -> Set[str]</code>: If <code>self.is_loggable</code> is true, default logs <code>get_required_data_types()</code>. Override if you want to log a different set of data.</li>
            <li><code>get_log_filename_suffix(self) -> str</code>: If <code>self.is_loggable</code> is true, default creates a suffix from class name or <code>config['title']</code>. Override for custom CSV filenames for this component type.</li>
            <li><code>handle_missing_uuids(self, missing_uuids_for_component: Set[str])</code>: The base class shows/hides a standard text overlay. Override for fully custom visual feedback when required data is missing (e.g., disabling parts of your component's UI, showing different icons). If you override, decide if you also want to call <code>super().handle_missing_uuids(...)</code>.</li>
        </ul>
        <p><b>Example Skeleton for a Custom Component:</b></p>
        """ + self._get_formatted_code_block("""
# In section: # 5. --- GUI Component Classes ---
class MyCustomGaugeWidget(BaseGuiComponent):
    def __init__(self, config: Dict[str, Any], data_buffers_ref, device_config_ref, parent=None):
        super().__init__(config, data_buffers_ref, device_config_ref, parent)
        
        self.data_to_display = self.config.get("data_key_for_gauge", "some_default_data_type")
        self.gauge_title = self.config.get("gauge_title", "Live Value")
        self.min_val = self.config.get("min_value", 0.0)
        self.max_val = self.config.get("max_value", 100.0)

        # --- UI Setup ---
        main_layout = QVBoxLayout(self)
        self.title_label = QLabel(self.gauge_title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # For a real gauge, you'd use a custom QWidget with paintEvent, or a library
        self.value_display_label = QLabel("--.--") 
        self.value_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.value_display_label.font()
        font.setPointSize(18)
        font.setBold(True)
        self.value_display_label.setFont(font)

        main_layout.addWidget(self.title_label)
        main_layout.addWidget(self.value_display_label)
        # Add more complex UI elements for a gauge here
        self.setLayout(main_layout)
        self.setFixedHeight(150) # Example fixed size

    def get_required_data_types(self) -> Set[str]:
        return {self.data_to_display}

    def update_component(self, current_relative_time: float, is_flowing: bool):
        if plotting_paused:
            # Could set text to "(Paused)" or similar if desired
            return

        data_series = self.data_buffers_ref.get(self.data_to_display, [])
        if data_series:
            latest_timestamp, latest_value = data_series[-1]
            # Here you would update your gauge visuals. For this label example:
            self.value_display_label.setText(f"{latest_value:.2f}")
            # Add logic to change color or appearance based on value vs min/max
        else:
            # Only show "--.--" if not paused and no UUID missing overlay is active
            if not (self.uuid_missing_overlay and self.uuid_missing_overlay.isVisible()):
                self.value_display_label.setText("--.--")

    def clear_component(self):
        self.value_display_label.setText("--.--")
        # Reset any gauge-specific visuals
        super().handle_missing_uuids(set()) # Important to clear base overlay

    # Optional: Override for custom log filename if this component is loggable
    # def get_log_filename_suffix(self) -> str:
    #     if self.is_loggable:
    #         return f"gauge_{self.data_to_display}"
    #     return ""
""") + """
        <h4>Step 2: Use Your New Component Type in `tab_configs`</h4>
        <p><b>Action:</b> In the <code># 6. --- Tab Layout Configuration ---</code> section (where the <code>tab_configs</code> global list is defined), you can now use your new component class just like the built-in ones.</p>
        <p><b>Details:</b> Add an entry to a tab's <code>'layout'</code> list, specifying <code>'component_class': MyCustomGaugeWidget</code> (or whatever your new class is named). Provide any necessary parameters that your component's <code>__init__</code> method expects within its <code>'config'</code> dictionary.</p>
        <p><b>Example using <code>MyCustomGaugeWidget</code>:</b></p>
        """ + self._get_formatted_code_block("""
# In tab_configs, within a tab's 'layout' list:
{
    'component_class': MyCustomGaugeWidget, # Your new component class
    'row': 2, 
    'col': 0,
    'colspan': 1, # Example: make it span 1 column
    'config': {
        'gauge_title': 'System Pressure',
        'data_key_for_gauge': 'system_pressure_psi', # data_type it should display
        'min_value': 0,
        'max_value': 200,
        'enable_logging': True # If you want its data logged
    }
}
""") + """
        <p>The <code>GuiManager</code> will automatically instantiate and manage your custom component as part of the GUI. Remember to restart the application to see your changes.</p>
        """
        self._help_content_map[add_component_type_key] = add_component_type_text_html

        add_tab_key = "add_tab"
        self._add_help_topic("Adding/Modifying Tabs & Layouts", scenarios_item, add_tab_key)
        add_tab_text_html = """
        <h3>Scenario: Adding a New Tab or Modifying an Existing Tab's Layout</h3>
        <p>The global <code>tab_configs</code> list (defined in section <code># 6. --- Tab Layout Configuration ---</code>) is the central place for defining the structure of all tabs and the arrangement of components within them.</p>

        <h4>To Add an Entirely New Tab:</h4>
        <p><b>Action:</b> Append a new dictionary to the <code>tab_configs</code> list. This new dictionary will define your new tab.</p>
        <p><b>Details:</b> This new tab definition dictionary must have at least two keys:</p>
        <ul>
            <li><code>'tab_title' (str)</code>: The string that will appear as the name on the tab itself in the GUI (e.g., <code>'Advanced Diagnostics'</code>).</li>
            <li><code>'layout' (List[Dict])</code>: A list of component definition dictionaries. Each dictionary in this list defines one GUI component that will appear on this new tab. For the structure of a component definition dictionary, refer to the "Tab Layout (<code>tab_configs</code>)" help topic or other scenarios.</li>
        </ul>
        <p><b>Example (adding a new tab named "Environment" with one plot and one value display):</b></p>
        """ + self._get_formatted_code_block("""
# In section # 6. --- Tab Layout Configuration ---
# Append this entire dictionary to the existing tab_configs = [...] list:
{
    'tab_title': 'Environment Sensors',
    'layout': [
        {   # First component on this new tab
            'component_class': TimeSeriesPlotComponent,
            'row': 0, 'col': 0, 'rowspan': 1, 'colspan': 2, # Span 2 columns
            'config': {
                'title': 'Ambient Temperature & Humidity',
                'datasets': [
                    {'data_type': 'ambient_temp_c', 'label': 'Temperature (°C)', 'color': 'red'},
                    {'data_type': 'ambient_humidity_pct', 'label': 'Humidity (%)', 'color': 'blue'}
                ],
                'enable_logging': True
            }
        },
        {   # Second component on this new tab
            'component_class': SingleValueDisplayComponent,
            'row': 1, 'col': 0, # Position below the plot, in the first column
            'config': {
                'label': 'Current Temp:', 
                'data_type': 'ambient_temp_c',
                'format': '{:.1f}', 'units': '°C'
                # 'enable_logging' defaults to False if not specified here
            }
        },
        # ... you can add more components to this new tab's 'layout' list ...
    ]
}
""") + """

        <h4>To Add a New Component to an Existing Tab:</h4>
        <p><b>Action:</b> First, locate the dictionary for the existing tab within the <code>tab_configs</code> list (you can identify it by its <code>'tab_title'</code> value).</p>
        <p><b>Details:</b> Once you've found the target tab's dictionary, append a new component definition dictionary to its <code>'layout'</code> list.
        Make sure to specify <code>'row'</code> and <code>'col'</code> coordinates for the new component that place it appropriately within that tab's grid. Consider if it needs to span multiple rows/columns (<code>'rowspan'</code>, <code>'colspan'</code>) and if this affects the layout of other components on that tab.
        </p>
        <p><b>Example (adding a new <code>SingleValueDisplayComponent</code> to an existing tab that already has a plot at row 0, col 0):</b></p>
        """ + self._get_formatted_code_block("""
# Find your existing tab definition in tab_configs, e.g.:
# {
# 'tab_title': 'IMU Basic',
# 'layout': [
# { 'component_class': TimeSeriesPlotComponent, 'row':0, 'col':0, ...config...},
#         # <<< ADD YOUR NEW COMPONENT DEFINITION DICTIONARY HERE >>>
#         {
#             'component_class': SingleValueDisplayComponent,
#             'row': 1,  # Place it in the next row
#             'col': 0,  # In the same column as the plot above, or a different one
#             'config': { 
#                 'label': 'Max Roll:', 
#                 'data_type': 'orientation_x_max_hold', # Assuming this data_type exists
#                 'format': '{:.1f}', 'units': '°'
#             }
#         }
# ]
# },
""") + """
        <h4>To Modify an Existing Component's Configuration, Position, or Size on a Tab:</h4>
        <p><b>Action:</b> Navigate to the <code>tab_configs</code> list. Find the specific tab dictionary by its <code>'tab_title'</code>. Then, within that tab's <code>'layout'</code> list, find the component definition dictionary for the component you wish to change (you might identify it by its <code>'component_class'</code> and current <code>'config'</code> values like <code>'title'</code>).</p>
        <p><b>Details for Modification:</b></p>
        <ul>
            <li><b>Position/Size:</b> Change its <code>'row'</code>, <code>'col'</code>, <code>'rowspan'</code>, or <code>'colspan'</code> integer values.</li>
            <li><b>Configuration:</b> Modify any key-value pairs within its <code>'config'</code> dictionary. This could be changing a plot's title (<code>config['title']</code>), adding/removing/modifying datasets in a <code>TimeSeriesPlotComponent</code> (<code>config['datasets']</code>), changing the <code>data_type</code> a <code>SingleValueDisplayComponent</code> shows (<code>config['data_type']</code>), or toggling CSV logging for that instance (<code>config['enable_logging']</code>). The acceptable keys in <code>'config'</code> depend on the specific <code>component_class</code>.</li>
        </ul>
        <p>Remember that the application needs to be restarted for any changes made to <code>tab_configs</code> to be reflected in the GUI.</p>
        """
        self._help_content_map[add_tab_key] = add_tab_text_html

        config_logging_key = "config_logging"
        self._add_help_topic("Configuring Data Logging", scenarios_item, config_logging_key)
        config_logging_text_html = """
        <h3>Scenario: Configuring Data Logging for a Component</h3>
        <p>The application supports logging data to CSV files during a "Capture" session. This logging is configured on a per-component-instance basis.</p>

        <h4>Step 1: Enable Logging for a Specific Component Instance</h4>
        <p><b>Action:</b> In the <code>tab_configs</code> global list (section <code># 6. --- Tab Layout Configuration ---</code>), locate the definition dictionary for the specific component instance you want to enable logging for.</p>
        <p><b>Details:</b> Within that component's <code>'config'</code> sub-dictionary, add or set the key <code>'enable_logging'</code> to <code>True</code>. If this key is set to <code>False</code> or is absent, logging will be disabled for that particular instance of the component.</p>
        <p><b>Example (enabling logging for a <code>TimeSeriesPlotComponent</code> instance):</b></p>
        """ + self._get_formatted_code_block("""
# In tab_configs, for a specific TimeSeriesPlotComponent instance:
{
    'component_class': TimeSeriesPlotComponent,
    'row': 0, 'col': 0, # Its position on the tab
    'config': {
        'title': 'Primary Sensor Data',
        'datasets': [{'data_type': 'sensor_alpha_value', 'label': 'Alpha', 'color': 'blue'}],
        'enable_logging': True  # <<--- THIS ENABLES LOGGING FOR THIS PLOT
    }
}
""") + """
        <p>Setting <code>'enable_logging': True</code> makes this component instance "loggable". This means two things:</p>
        <ol>
            <li>It may generate its own individual CSV file containing its selected data.</li>
            <li>Its selected data will be included in the "master" CSV file generated for the tab it resides on.</li>
        </ol>

        <h4>Step 2: Determine Which `data_type`s are Logged (<code>get_loggable_data_types</code>)</h4>
        <p><b>Context:</b> When logging is enabled for a component instance (via <code>'enable_logging': True</code> in its config), the system needs to know precisely which <code>data_type</code>(s) associated with that component should be written to the CSV file(s).</p>
        <p><b>Default Behavior (from <code>BaseGuiComponent</code>):</b> By default, if a component instance is loggable, it will attempt to log all the <code>data_type</code>s that are returned by its <code>get_required_data_types()</code> method. This is often convenient as it logs what the component displays.</p>
        <p><b>Customization (Optional, by overriding in the component's class):</b> If you need a loggable component type to log a <em>different</em> set of <code>data_type</code>s than what it strictly requires for its visual display, you must override the <code>get_loggable_data_types(self) -> Set[str]</code> method within that component's class definition (in section <code># 5. --- GUI Component Classes ---</code>).</p>
        <p><b>Example (in a custom component class that calculates a result but might also want to log its inputs):</b></p>
        """ + self._get_formatted_code_block("""
# In section: # 5. --- GUI Component Classes ---
class MyComplexProcessor(BaseGuiComponent):
    # ... __init__ and other methods ...

    def get_required_data_types(self) -> Set[str]:
        # For display, it might only need to show the final result
        return {'processed_output_value'} 

    def get_loggable_data_types(self) -> Set[str]:
        # If self.is_loggable is True (due to 'enable_logging': True in config),
        # we want to log not just the output, but also the raw inputs it used.
        # Note: self.is_loggable is checked by the framework before calling this.
        return {'raw_input_X', 'raw_input_Y', 'processed_output_value'}
""") + """

        <h4>Step 3: Define the Individual CSV Filename Suffix (<code>get_log_filename_suffix</code>)</h4>
        <p><b>Context:</b> Each loggable component instance can generate its own individual CSV file (in addition to contributing to the master tab CSV). The filename for this individual CSV is partly determined by a suffix string provided by the component.</p>
        <p><b>Default Behavior (from <code>BaseGuiComponent</code>):</b> The <code>BaseGuiComponent</code> class provides a default suffix. It tries to use the <code>'title'</code> field from the component's <code>config</code> dictionary if available; otherwise, it falls back to using the component's class name and a unique ID. This usually results in reasonably descriptive filenames.</p>
        <p><b>Customization (Optional, by overriding in the component's class):</b> To have more specific or controlled naming for an individual component's CSV log file, you can override the <code>get_log_filename_suffix(self) -> str</code> method in that component's class definition.</p>
        <p><b>Example (in a component class, making the suffix include a configured ID):</b></p>
        """ + self._get_formatted_code_block("""
# In section: # 5. --- GUI Component Classes ---
class SpecificSensorPlot(BaseGuiComponent):
    # ... __init__ where self.config might contain 'sensor_channel_id' ...

    def get_log_filename_suffix(self) -> str:
        # This method is only called by the framework if self.is_loggable is True.
        channel_id = self.config.get('sensor_channel_id', 'unknown_channel')
        base_title = self.config.get('title', self.__class__.__name__)
        # Clean up the title and ID for use in a filename
        safe_title = re.sub(r'[^\\w\\-]+', '_', base_title).strip('_')
        safe_channel_id = re.sub(r'[^\\w\\-]+', '_', str(channel_id)).strip('_')
        return f"plot_{safe_title}_channel_{safe_channel_id}"
        # Return "" if not loggable (though framework usually checks is_loggable first)
""") + """
        <p>The final individual CSV filename will be constructed by the system using a timestamp prefix, the capture directory structure, and this suffix. For example: <code>Logs/YYYYMMDD_HHMMSS/csv_files/TIMESTAMP_your_suffix.csv</code>.</p>
        <p><b>Master Tab CSV Files:</b></p>
        <p>In addition to any individual CSV files generated by components, a "master" CSV file is created for <em>each tab</em> in the GUI. This master CSV for a tab includes columns for all <code>data_type</code>s that are designated as loggable by <em>any</em> loggable component instance present on that specific tab. The data is merged and time-aligned (outer join, forward-filled where necessary, based on pandas behavior in your code) to provide a comprehensive view of all logged data on that tab. The filename for this master tab CSV is based on the tab's title.</p>
        <p>Logging occurs when a "Capture" session is started and then stopped via the GUI buttons. Files are generated upon stopping the capture.</p>
        """
        self._help_content_map[config_logging_key] = config_logging_text_html


        # --- Backend / Core Logic (Modification Not Recommended) ---
        backend_key = "backend_logic"
        backend_item = self._add_help_topic("Backend & Core Logic (Advanced)", self.code_help_tree, backend_key)
        # Set a special flag or style for "not recommended" items if desired
        font = backend_item.font(0)
        font.setItalic(True)
        backend_item.setFont(0, font)
        backend_item.setForeground(0, QColor("slateGray")) # Example: Dim the text color

        backend_intro_text = """
        <h3>Backend & Core Logic (Advanced - Modification Not Recommended)</h3>
        <p><font color="red"><b>Warning:</b> Modifications to the code described in this section can easily break core application functionality. Proceed with extreme caution and only if you have a deep understanding of the application's architecture, asynchronous programming (<code>asyncio</code>, <code>qasync</code>), and BLE (<code>bleak</code>).</font></p>
        <p>This section provides a high-level overview of the non-customizable parts of the application. It's intended for understanding, not direct modification for typical use cases.</p>
        """
        self._help_content_map[backend_key] = backend_intro_text

        # Helper function to apply "not recommended" style to children
        def _style_backend_sub_item(item: QTreeWidgetItem):
            child_font = item.font(0)
            child_font.setItalic(True)
            item.setFont(0, child_font)
            item.setForeground(0, QColor("slateGray"))

        # Sub-item: BLE Handling
        ble_handling_key = "backend_ble"
        ble_item = self._add_help_topic("1. BLE Communication & State Machine", backend_item, ble_handling_key)
        _style_backend_sub_item(ble_item)
        ble_handling_text_html = """
        <h4>1. BLE Communication & State Machine (<code>connection_task</code>, etc.)</h4>
        <p><b>Core Function:</b> <code>connection_task()</code> (defined globally).</p>
        <p>This asynchronous function is the heart of the BLE interaction. It manages the application's connection lifecycle through a state machine, controlled by the global <code>state</code> variable and <code>stop_flag</code>.</p>
        <p><b>Key States and Transitions:</b></p>
        <ul>
            <li><b><code>"idle"</code></b>: The initial state. The application is waiting for user interaction. Pressing "Start Scan" transitions to <code>"scanning"</code>.</li>
            <li><b><code>"scanning"</code></b>:
                <ul>
                    <li>Calls <code>find_device()</code> which uses <code>BleakScanner</code> to search for the target device (specified by <code>device_config.name</code> and <code>device_config.service_uuid</code>).</li>
                    <li>Uses a <code>detection_callback</code> to identify the correct device based on its advertised name and service UUIDs.</li>
                    <li>If the device is found, it attempts to connect using <code>BleakClient(target_device).connect()</code>.</li>
                    <li>If connection is successful, transitions to <code>"connected"</code>.</li>
                    <li>If scan times out or connection fails after retries, it may briefly show a status and return to scanning or, if "Stop Scan" is pressed, transition to <code>"idle"</code>.</li>
                </ul>
            </li>
            <li><b><code>"connected"</code></b>:
                <ul>
                    <li>Once connected, it verifies the presence of the main service (<code>device_config.service_uuid</code>).</li>
                    <li>It then iterates through all characteristics defined in <code>device_config.characteristics</code>. For each:
                        <ul>
                            <li>It checks if the characteristic UUID exists on the connected device's service.</li>
                            <li>If found, it checks if the characteristic supports "notify" or "indicate" properties.</li>
                            <li>If both conditions are met, it calls <code>client.start_notify(char_uuid, handler_func)</code>, where <code>handler_func</code> is a partial application of <code>notification_handler</code> bound to the specific <code>CharacteristicConfig</code>.</li>
                            <li>If a required characteristic UUID is not found or doesn't support notifications, it's added to a <code>missing_uuids</code> set, and a signal (<code>gui_emitter.missing_uuids_signal</code>) is emitted to update the GUI.</li>
                        </ul>
                    </li>
                    <li>If no usable characteristics are found, or if starting notifications fails critically, it transitions to <code>"disconnecting"</code>.</li>
                    <li>Otherwise, it enters a listening loop, periodically checking:
                        <ul>
                            <li>The <code>disconnected_event</code> (set by <code>bleak</code>'s <code>disconnected_callback</code> or manually).</li>
                            <li>A data timeout (<code>device_config.data_timeout</code>): if no data is received via <em>any</em> notification for this duration.</li>
                            <li><code>client.is_connected</code> status.</li>
                            <li>The global <code>stop_flag</code>.</li>
                        </ul>
                        Any of these conditions (or user pressing "Disconnect") will trigger a transition to <code>"disconnecting"</code>.
                    </li>
                </ul>
            </li>
            <li><b><code>"disconnecting"</code></b>:
                <ul>
                    <li>This state ensures cleanup. It attempts to stop notifications for all subscribed characteristics (<code>client.stop_notify()</code>).</li>
                    <li>Then, it calls <code>client.disconnect()</code>.</li>
                    <li>Finally, it transitions to <code>"idle"</code>.</li>
                </ul>
            </li>
        </ul>
        <p><b><code>notification_handler(char_config, sender, data)</code>:</b></p>
        <ul>
            <li>This asynchronous callback is invoked by <code>bleak</code> whenever data arrives on a subscribed characteristic.</li>
            <li>It calls the specific <code>handler</code> function (e.g., <code>handle_orientation_data</code>) associated with the <code>char_config</code> for that UUID.</li>
            <li>The parsed values (dictionary) from the user-defined handler are then added to the global <code>data_buffers</code> with a relative timestamp.</li>
            <li>Crucially, after new raw data is added, it calls <code>compute_all_derived_data()</code> to update any dependent derived data series.</li>
        </ul>
        <p><b><code>GuiSignalEmitter</code>:</b> An instance (<code>gui_emitter</code>) is used to send signals from the asynchronous BLE logic (running in <code>asyncio</code>'s event loop) to the PyQt GUI thread. This is essential for updating GUI elements like the LED, status labels, and scan button text in a thread-safe manner.</p>
        <p><b>Error Handling:</b> The <code>connection_task</code> includes <code>try-except</code> blocks to handle <code>BleakError</code>, <code>asyncio.TimeoutError</code>, <code>asyncio.CancelledError</code>, and general exceptions during scanning, connection, and notification phases to prevent crashes and attempt to recover or transition to a safe state.</p>
        <p><b>Concurrency:</b> The entire BLE operation runs as an <code>asyncio.Task</code> (<code>current_task</code>), managed by the <code>qasync</code> event loop which integrates <code>asyncio</code> with Qt's event loop.</p>
        """
        self._help_content_map[ble_handling_key] = ble_handling_text_html

        # Sub-item: Core GUI Logic
        core_gui_key = "backend_gui"
        core_gui_item = self._add_help_topic("2. Core GUI Management & Event Loop", backend_item, core_gui_key)
        _style_backend_sub_item(core_gui_item)
        core_gui_text_html = """
        <h4>2. Core GUI Management & Event Loop (<code>MainWindow</code>, <code>GuiManager</code>, <code>qasync</code>)</h4>
        <p><b><code>MainWindow</code> Class:</b></p>
        <ul>
            <li><b>Primary Window:</b> The main application window, inheriting from <code>QMainWindow</code>.</li>
            <li><b>UI Setup:</b> Initializes the main layout, top button bar (LED, scan button, etc.), bottom control bar (flowing mode, interval, help), and the log text box.</li>
            <li><b><code>GuiManager</code> Instantiation:</b> Creates an instance of <code>GuiManager</code>, passing it the <code>QTabWidget</code>, <code>tab_configs</code>, <code>data_buffers</code>, and <code>device_config</code>. The <code>GuiManager</code> is then responsible for populating the tabs.</li>
            <li><b>Signal Connections:</b> Connects GUI element signals (button clicks, checkbox changes) to their respective handler methods (slots). It also connects signals from <code>gui_emitter</code> (originating from BLE logic) to methods that update the GUI state (e.g., <code>handle_state_change</code>, <code>update_scan_status</code>).</li>
            <li><b>State Management (via <code>handle_state_change</code>):</b> This crucial method updates the enabled/disabled state and text of GUI controls (like the scan button, pause button, LED color) based on the global <code>state</code> variable (<code>"idle"</code>, <code>"scanning"</code>, <code>"connected"</code>, <code>"disconnecting"</code>).</li>
            <li><b>Timers:</b>
                <ul>
                    <li><code>plot_update_timer</code>: Periodically calls <code>trigger_gui_update</code>, which emits <code>request_plot_update_signal</code>. This signal is connected to <code>_update_gui_now</code>, which then calls <code>gui_manager.update_all_components()</code> to refresh all visual components if not paused.</li>
                    <li><code>scan_throbber_timer</code>: Animates the "Scanning..." throbber in the status label when <code>state == "scanning"</code>.</li>
                </ul>
            </li>
            <li><b>Asynchronous Operations Control:</b> Methods like <code>toggle_scan</code> are <code>@qasync.asyncSlot()</code>, allowing them to <code>await</code> asynchronous BLE operations (like <code>connection_task</code> or its cancellation) without freezing the GUI.</li>
            <li><b>Capture Logic:</b> Manages the <code>is_capturing</code> state and calls <code>generate_pdf_plots_from_buffer</code> and <code>generate_csv_files_from_buffer</code> when a capture is stopped.</li>
            <li><b>Shutdown Handling (<code>closeEvent</code>, <code>async_shutdown_operations</code>):</b> Implements a graceful shutdown sequence. When the window is closed, it sets <code>stop_flag</code>, cancels any ongoing <code>current_task</code> (BLE operations), disconnects the client if connected, stops timers, and then quits the application. This is done asynchronously to ensure BLE operations are properly terminated.</li>
        </ul>
        <p><b><code>GuiManager</code> Class:</b></p>
        <ul>
            <li><b>Dynamic GUI Creation:</b> Parses the <code>tab_configs</code> list at startup. For each tab definition, it creates a new tab. For each component definition within a tab's layout, it instantiates the specified <code>component_class</code>, passing it its <code>config</code> dictionary and references to <code>data_buffers</code> and <code>device_config</code>.</li>
            <li><b>Component Management:</b> Stores all created component instances in <code>self.all_components</code>.</li>
            <li><b><code>update_all_components()</code>:</b> Iterates through <code>self.all_components</code> and calls each component's <code>update_component()</code> method.</li>
            <li><b><code>clear_all_components()</code>:</b> Calls each component's <code>clear_component()</code> method.</li>
            <li><b><code>notify_missing_uuids()</code>:</b> Receives a set of missing UUIDs from the BLE logic (via <code>gui_emitter</code>). For each component, it determines if any of its <code>get_required_data_types()</code> depend on these missing UUIDs (recursively checking derived data dependencies) and then calls the component's <code>handle_missing_uuids()</code> method with the relevant subset of missing UUIDs for that component.</li>
        </ul>
        <p><b><code>qasync</code> and <code>asyncio</code> Integration:</b></p>
        <ul>
            <li>The application uses <code>qasync.QEventLoop(app)</code> to integrate Python's <code>asyncio</code> event loop with Qt's event loop. This allows asynchronous Python code (primarily for BLE communication which involves waiting for I/O) to run concurrently with the responsive Qt GUI.</li>
            <li><code>asyncio.set_event_loop(qasync_loop)</code> makes this integrated loop the default for <code>asyncio</code>.</li>
            <li>The main application execution is <code>with qasync_loop: qasync_loop.run_forever()</code>.</li>
        </ul>
        <p><b>Data Flow (High Level):</b></p>
        <p>BLE Device → <code>bleak</code> → <code>notification_handler</code> (async) → User's <code>handle_*</code> function → <code>data_buffers</code> (global) & <code>compute_all_derived_data</code> → <code>GuiManager</code> (via timer) → Individual <code>BaseGuiComponent</code> subclasses' <code>update_component()</code> methods → Visual update on screen.</p>
        <p>State changes and status updates from BLE logic flow via <code>gui_emitter</code> signals to <code>MainWindow</code> slots, which then update GUI elements.</p>
        """
        self._help_content_map[core_gui_key] = core_gui_text_html

        # Sub-item: Data Buffering & Export
        data_handling_key = "backend_data"
        data_item = self._add_help_topic("3. Data Buffering & Export", backend_item, data_handling_key)
        _style_backend_sub_item(data_item)

        data_handling_text_html = """
        <h4>3. Data Buffering & Export Mechanisms</h4>
        <p><b><code>data_buffers</code> (Global Dictionary):</b></p>
        <ul>
            <li><b>Structure:</b> <code>Dict[str, List[Tuple[float, float]]]</code>.</li>
            <li><b>Keys:</b> <code>data_type</code> strings (e.g., <code>'orientation_x'</code>, <code>'estimated_weight'</code>, <code>'impedance_magnitude_ohm'</code>, <code>'power_watts'</code>). These come from the <code>produces_data_types</code> in <code>CharacteristicConfig</code> or the <code>data_type</code> in <code>DerivedDataDefinition</code>.</li>
            <li><b>Values:</b> Each value is a list of tuples. Each tuple is <code>(relative_time_seconds, value_float)</code>.
                <ul><li><code>relative_time_seconds</code>: Time in seconds since the <code>start_time</code> of the current connection/session.</li></ul>
            </li>
            <li><b>Population:</b>
                <ul>
                    <li>Raw sensor data is added by <code>notification_handler</code> after being processed by a user-defined <code>handle_*</code> function.</li>
                    <li>Derived data is added by <code>compute_all_derived_data</code> after being calculated by a user-defined computation function.</li>
                </ul>
            </li>
            <li><b>Clearing:</b> <code>data_buffers.clear()</code> is called by <code>MainWindow.clear_gui_action()</code> and during some state transitions to reset all stored data.</li>
            <li><b>Replay Loading:</b> In Replay Mode, the <code>CsvReplaySource</code> class performs a bulk load of data from one or more CSV files directly into <code>data_buffers</code>, populating it with historical data.</li>
            <li><b>Replay Reading:</b> GUI components in Replay Mode use the <code>get_value_at_time()</code> utility function to efficiently find the correct historical data point from these buffers to display for a given time selected on a slider.</li>
        </ul>
        <p><b><code>start_time</code> (Global <code>datetime.datetime</code>):</b></p>
        <ul>
            <li>Set to <code>datetime.datetime.now()</code> when the <em>first piece of data</em> is received by <code>notification_handler</code> after a connection is established or after a "Clear GUI" action.</li>
            <li>Used to calculate the <code>relative_time_seconds</code> for all entries in <code>data_buffers</code>.</li>
            <li>Reset to <code>None</code> by <code>MainWindow.clear_gui_action()</code>.</li>
        </ul>

        <p><b>Data Capture and Export (<code>MainWindow</code> methods):</b></p>
        <ul>
            <li><b>Initiation:</b> <code>toggle_capture()</code> method in <code>MainWindow</code>, triggered by "Start Capture" button.
                <ul>
                    <li>Sets <code>self.is_capturing = True</code>.</li>
                    <li>Records <code>self.capture_t0_absolute</code> (absolute <code>datetime</code>) and <code>self.capture_start_relative_time</code> (relative to session <code>start_time</code>).</li>
                    <li>Creates a timestamped output directory under "Logs/".</li>
                </ul>
            </li>
            <li><b>Termination & Generation:</b> <code>stop_and_generate_files()</code> method, called when "Stop Capture & Export" is pressed or during shutdown if capture was active.
                <ul>
                    <li>Sets <code>self.is_capturing = False</code>.</li>
                    <li>Calls <code>generate_pdf_plots_from_buffer()</code> and <code>generate_csv_files_from_buffer()</code>.</li>
                </ul>
            </li>
            <li><b><code>generate_pdf_plots_from_buffer()</code>:</b>
                <ul>
                    <li>Iterates through all components in <code>GuiManager.all_components</code>.</li>
                    <li>For each component that is an instance of <code>TimeSeriesPlotComponent</code> and doesn't have missing UUIDs for its data:
                        <ul>
                            <li>Creates a Matplotlib figure and axes.</li>
                            <li>Retrieves data from <code>data_buffers</code> for the <code>data_type</code>s specified in the plot's <code>config['datasets']</code>.</li>
                            <li>Filters this data to include only samples whose original <code>relative_time_seconds</code> is <em>at or after</em> <code>self.capture_start_relative_time</code>.</li>
                            <li>The X-axis for the PDF plot is adjusted so that t=0 corresponds to <code>self.capture_start_relative_time</code>.</li>
                            <li>Uses Matplotlib to plot the data, apply labels, title, and legend based on the component's config.</li>
                            <li>Saves the plot as a PDF file in the "pdf_plots" subdirectory of the capture folder. The filename is based on the component's <code>get_log_filename_suffix()</code>.</li>
                        </ul>
                    </li>
                </ul>
            </li>
            <li><b><code>generate_csv_files_from_buffer()</code>:</b>
                <ul>
                    <li><b>Individual Component CSVs:</b>
                        <ul>
                            <li>Iterates through all components in <code>GuiManager.all_components</code>.</li>
                            <li>If <code>component.is_loggable</code> is <code>True</code> (set by <code>'enable_logging': True</code> in its config):
                                <ul>
                                    <li>Gets the set of <code>data_type</code>s to log from <code>component.get_loggable_data_types()</code>.</li>
                                    <li>For each such <code>data_type</code>, retrieves data from <code>data_buffers</code>, filtering for the capture window (between <code>capture_start_relative_time</code> and <code>capture_end_relative_time</code>). UUID-missing data types are skipped.</li>
                                    <li>Converts these series into a Pandas DataFrame.</li>
                                    <li>Adds a "Time (s)" column where t=0 is the <code>capture_start_relative_time</code>.</li>
                                    <li>Saves this DataFrame to a CSV file in the "csv_files" subdirectory. Filename uses <code>component.get_log_filename_suffix()</code>.</li>
                                </ul>
                            </li>
                        </ul>
                    </li>
                    <li><b>Master Tab CSVs:</b>
                        <ul>
                            <li>Iterates through each tab defined in <code>tab_configs</code>.</li>
                            <li>Collects all unique <code>data_type</code>s that are designated as loggable by <em>any</em> loggable component instance present on that specific tab.</li>
                            <li>Retrieves, filters (for capture window and missing UUIDs), and converts these data types into Pandas Series, similar to individual CSVs.</li>
                            <li>Concatenates these Series into a single "master" DataFrame for the tab (using an outer join to align by timestamp).</li>
                            <li>Adds a "Master Time (s)" column (t=0 at capture start).</li>
                            <li>Saves this master DataFrame to a CSV file in the "csv_files" subdirectory, with a filename based on the tab's title.</li>
                        </ul>
                    </li>
                </ul>
            </li>
        </ul>
        <p><b>Data Filtering:</b> The export functions filter data from <code>data_buffers</code> to include only the portion recorded between the "Start Capture" and "Stop Capture" events, based on the <code>relative_time_seconds</code> timestamps.</p>
        """
        
        self._help_content_map[data_handling_key] = data_handling_text_html


        add_replay_key = "add_replay_compat"
        self._add_help_topic("Making a Component Replay-Compatible", scenarios_item, add_replay_key)
        add_replay_text_html = """
        <h3>Scenario: Making a Custom Component Replay-Compatible</h3>
        <p>To make a custom GUI component work with the data replay sliders, you need to implement a specific pattern. The goal is to switch between live data updates and slider-driven historical data rendering.</p>

        <h4>Step 1: Add Replay Controls to the Component's UI</h4>
        <p><b>Action:</b> In your component's <code>__init__</code> method, create the necessary replay controls (e.g., a <code>QSlider</code> for single-point scrubbing, or a <code>superqt.QRangeSlider</code> for windowed views). Add them to your component's layout.</p>
        <p><b>Details:</b></p>
        <ul>
            <li>These controls should be hidden by default: <code>self.my_slider.setVisible(False)</code>. They will be made visible only in replay mode.</li>
            <li>Connect the slider's <code>valueChanged</code> signal to a handler method in your component (e.g., <code>self.my_slider.valueChanged.connect(self.on_slider_moved)</code>).</li>
        </ul>

        <h4>Step 2: Modify `update_component` to Handle Replay State</h4>
        <p><b>Action:</b> In your component's <code>update_component(self, current_relative_time: float, is_flowing: bool)</code> method, use the <code>is_flowing</code> argument to distinguish between live mode and replay mode.</p>
        <p><b>Details:</b></p>
        <ul>
            <li><b>Live Mode (<code>if is_flowing:</code>):</b> This block contains your existing logic for updating from live data. It should also ensure your replay controls are hidden (<code>self.my_slider.setVisible(False)</code>).</li>
            <li><b>Replay Mode (<code>else:</code>):</b> This block is executed when replay is active.
                <ul>
                    <li>Make your replay slider visible: <code>self.my_slider.setVisible(True)</code>.</li>
                    <li>Set the slider's range (min/max values) based on the total time span of the relevant data in <code>self.data_buffers_ref</code>.</li>
                    <li><b>Important:</b> The rendering logic for a specific time point should <em>not</em> be in this block. This block only manages the visibility and range of the slider. The actual rendering is triggered by the slider's signal.</li>
                </ul>
            </li>
        </ul>

        <h4>Step 3: Implement the Slider's Handler Method</h4>
        <p><b>Action:</b> Create the method that was connected to your slider's <code>valueChanged</code> signal (e.g., <code>def on_slider_moved(self, slider_value):</code>).</p>
        <p><b>Details:</b></p>
        <ul>
            <li>This method is the entry point for replay rendering.</li>
            <li>It should first convert the integer <code>slider_value</code> to a float representing time in seconds.</li>
            <li>It then calls your core rendering logic function (see Step 4), passing this time value.</li>
            <li>It should also handle pausing the global plot updates if the user interacts with the slider, to prevent live updates from interfering.</li>
        </ul>

        <h4>Step 4: Implement a `render_for_time(time_sec)` Method</h4>
        <p><b>Action:</b> This is the core replay rendering logic for your component. It takes a specific time point and updates the component's visuals to match the data at that instant.</p>
        <p><b>Details:</b></p>
        <ul>
            <li>Use the utility function <code>get_value_at_time(data_type, time_sec, self.data_buffers_ref)</code> to fetch the historical value for each of your component's required data types at the given <code>time_sec</code>.</li>
            <li>If your component displays a "trail" of data (like the Nyquist plot or Heatmap CoP), you must reconstruct this trail by iterating through historical timestamps up to <code>time_sec</code> and calculating the state at each step.</li>
            <li>Once all historical data is retrieved, update your component's visual elements (e.g., redraw plots, update 3D model orientation, repaint a gauge).</li>
        </ul>
        <p><b>Example Pattern (from `IMUVisualizerComponent`):</b></p>
        """ + self._get_formatted_code_block("""
        # In update_component:
        def update_component(self, current_relative_time: float, is_flowing: bool):
            if plotting_paused and is_flowing: return
            
            if not is_flowing: # Replay Mode
                self.scrub_time_widget_imu.setVisible(True)
                # ... logic to set slider range based on data_buffers ...
                if plotting_paused: # If paused (slider mode), render for current slider value
                    self.render_imu_for_time(float(self.time_slider_imu.value()) / self.SLIDER_FLOAT_PRECISION_FACTOR_IMU)
            else: # Live Mode
                self.scrub_time_widget_imu.setVisible(False)
                # ... logic to update from latest data in data_buffers ...

        # Slider's connected slot:
        def on_time_slider_imu(self, slider_value_int: int):
            # ... logic to set global plotting_paused = True ...
            time_sec_float = float(slider_value_int) / self.SLIDER_FLOAT_PRECISION_FACTOR_IMU
            self.render_imu_for_time(time_sec_float)

        # Core replay rendering logic:
        def render_imu_for_time(self, time_sec: float):
            # Use get_value_at_time to get quaternion values at 'time_sec'
            q_w = get_value_at_time("quat_w", time_sec, self.data_buffers_ref)
            q_x = get_value_at_time("quat_x", time_sec, self.data_buffers_ref)
            # ... etc for q_y, q_z ...
            
            # Create the historical quaternion from these values
            historical_quaternion = QQuaternion(float(q_w), float(q_x), float(q_y), float(q_z))
            
            # Apply transformation to the 3D model
            transform = pg.Transform3D()
            transform.rotate(historical_quaternion)
            if self.object_mesh:
                self.object_mesh.setTransform(transform)
        """) + """
                <h4>Step 5: Update `clear_component`</h4>
                <p><b>Action:</b> Ensure your component's <code>clear_component</code> method also resets and hides your new replay controls.</p>
                <p><b>Example:</b></p>
                """ + self._get_formatted_code_block("""
        def clear_component(self):
            # ... clear your component's main display ...
            if hasattr(self, 'time_slider_imu'):
                self.time_slider_imu.setVisible(False)
                self.time_slider_imu.setRange(0, 0)
            super().handle_missing_replay_data(set()) # Clear any "CSV Data Missing" overlay
        """)
        self._help_content_map[add_replay_key] = add_replay_text_html


        # Expand Tree by default for better discoverability
        self.code_help_tree.expandAll()


    def _on_code_help_item_selected(self, item: QTreeWidgetItem, column: int = 0):
        content_key = item.data(0, Qt.ItemDataRole.UserRole) # Retrieve the string key
        if content_key:
            content_html = self._help_content_map.get(content_key, 
                "<p><b>Content not found for this topic.</b> This is an unexpected error.</p>")
            self.code_help_display.setHtml(f"<html><body>{content_html}</body></html>")
            self.code_help_display.scrollToAnchor(None) # Scroll to top
            self.code_help_display.verticalScrollBar().setValue(0)
        else:
            # Fallback if somehow an item has no content_key (e.g. if a header item was made selectable)
            self.code_help_display.setHtml(f"<html><body><p>No specific details available for this item. It might be a category header.</p></body></html>")
