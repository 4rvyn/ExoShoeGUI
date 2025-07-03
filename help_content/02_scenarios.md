# Adding New Features (Scenarios)
This section provides practical, step-by-step guides for common customization tasks. Each scenario includes code examples that are consistent with the application's framework.

<br>

### Scenario: Adding a New Sensor (BLE Characteristic)
This guide walks through the complete process of integrating data from a new BLE characteristic, from parsing raw bytes to displaying the data in the GUI.

#### Step 1: Implement a Data Handler Function
In the `# 1. --- Data Handlers ---` section, write a new Python function. This function's role is to parse the raw `bytearray` received from your sensor.

-   **Input:** It must accept a single `bytearray` argument.
-   **Output:** It must return a dictionary mapping string keys (`data_type` names) to their corresponding numeric values (e.g., `{'my_temp': 25.5, 'my_humidity': 45.1}`). <span style="color:#b71c1c;">These keys must be unique across the entire application.</span>
-   **Error Handling:** It must handle potential errors (e.g., incorrect data length, parsing issues) gracefully by logging a warning via `data_logger` and returning an empty dictionary `{}`.

<div style="background-color: #f0f0f0; border-left: 5px solid #666; padding: 10px; margin: 10px 0; font-family: Consolas, 'Courier New', monospace; font-size: 14px; color: #333;">
<pre><code><span style="color: #008080;"># In section: # 1. --- Data Handlers for Different Characteristics ---</span>
<span style="color: #0000ff;">import</span> struct <span style="color: #008080;"># Make sure 'struct' is imported for binary data parsing</span>

<span style="color: #0000ff;">def</span> <span style="color: #006400;">handle_my_new_bme_sensor</span>(data: <span style="color: #a31515;">bytearray</span>) -> <span style="color: #a31515;">dict</span>:
    <span style="color: #008080;">"""
    Parses a 5-byte payload for a BME-style sensor.
    Payload format: Temperature (int16_t, little-endian), 
                      Pressure (uint16_t, little-endian),
                      Humidity (uint8_t).
    """</span>
    <span style="color: #0000ff;">try</span>:
        <span style="color: #008080;"># Always validate the incoming data length first.</span>
        <span style="color: #0000ff;">if</span> len(data) &lt; <span style="color: #098658;">5</span>:
            data_logger.warning(<span style="color: #a31515;">f"MyNewBMESensor: Data too short ({</span>len(data)<span style="color: #a31515;">} bytes), expected 5."</span>)
            <span style="color: #0000ff;">return</span> {}
        
        <span style="color: #008080;"># Unpack the bytes using a format string. '&lt;' denotes little-endian.</span>
        raw_temp, raw_pressure, raw_humidity = struct.unpack(<span style="color: #a31515;">"&lt;hHB"</span>, data[:<span style="color: #098658;">5</span>])
        
        <span style="color: #008080;"># Return a dictionary with unique, descriptive keys and scaled values.</span>
        <span style="color: #0000ff;">return</span> {
            <span style="color: #a31515;">"bme_temperature_celsius"</span>: raw_temp / <span style="color: #098658;">100.0</span>,
            <span style="color: #a31515;">"bme_pressure_pascal"</span>: <span style="color: #a31515;">float</span>(raw_pressure),
            <span style="color: #a31515;">"bme_humidity_percent"</span>: <span style="color: #a31515;">float</span>(raw_humidity)
        }
    <span style="color: #0000ff;">except</span> <span style="color: #a31515;">Exception</span> <span style="color: #0000ff;">as</span> e:
        data_logger.error(<span style="color: #a31515;">f"MyNewBMESensor: Error parsing data: {e}"</span>)
        <span style="color: #0000ff;">return</span> {}</code></pre>
</div>

#### Step 2: Register the Characteristic in `DeviceConfig`
In the `# 4. --- Device Configuration ---` section, add a new `CharacteristicConfig` object to the `characteristics` list within your `device_config` instance. This critical step links your sensor's unique UUID to the new handler function and declares which `data_type` keys it is responsible for producing.

<div style="background-color: #f0f0f0; border-left: 5px solid #666; padding: 10px; margin: 10px 0; font-family: Consolas, 'Courier New', monospace; font-size: 14px; color: #333;">
<pre><code><span style="color: #008080;"># In section: # 4. --- Device Configuration ---</span>
device_config = DeviceConfig(
    <span style="color: #008080;"># ...</span>
    characteristics=[
        <span style="color: #008080;"># ... other existing CharacteristicConfig entries ...</span>
        
        <span style="color: #008080;"># Add your new characteristic here</span>
        CharacteristicConfig(
            uuid=<span style="color: #d16969;">"19B100XX-E8F2-537E-4F6C-D104768A1214"</span>, <span style="color: #008080;"># &lt;-- REPLACE WITH YOUR ACTUAL UUID</span>
            handler=handle_my_new_bme_sensor,
            produces_data_types=[
                <span style="color: #a31515;">'bme_temperature_celsius'</span>,
                <span style="color: #a31515;">'bme_pressure_pascal'</span>,
                <span style="color: #a31515;">'bme_humidity_percent'</span>
            ]
        ),
    ],
    <span style="color: #008080;"># ...</span>
)</code></pre>
</div>

#### Step 3: (Optional) Update `AVAILABLE_DEVICE_NAMES`
If this new characteristic is part of a new device profile with a different advertised name, add its name to the `AVAILABLE_DEVICE_NAMES` list to make it selectable in the GUI's `Target` dropdown menu.

<br>

### Scenario: Adding a New Derived Data Series
Create a "virtual" sensor by calculating a new data stream from one or more existing raw or derived data streams.

#### Step 1: Write a Computation Function
In the `# 2. --- Derived/Fusion Data Handlers ---` section, define a function that performs your calculation.

-   It must read its own input data from the global `data_buffers` dictionary.
-   It should handle cases where dependency data is not yet available by returning `None`.
-   For calculations requiring historical data (e.g., rate of change), use a `deque` as a persistent state variable attached to the function itself.

<div style="background-color: #f0f0f0; border-left: 5px solid #666; padding: 10px; margin: 10px 0; font-family: Consolas, 'Courier New', monospace; font-size: 14px; color: #333;">
<pre><code><span style="color: #008080;"># In section: # 2. --- Derived/Fusion Data Handlers ---</span>
<span style="color: #0000ff;">from</span> collections <span style="color: #0000ff;">import</span> deque

<span style="color: #0000ff;">def</span> <span style="color: #006400;">_compute_temperature_rate_of_change</span>() -> <span style="color: #a31515;">Optional</span>[<span style="color: #a31515;">float</span>]:
    <span style="color: #008080;"># Use a function attribute for persistent history</span>
    <span style="color: #0000ff;">if</span> <span style="color: #0000ff;">not</span> hasattr(_compute_temperature_rate_of_change, <span style="color: #a31515;">"_history"</span>):
        _compute_temperature_rate_of_change._history = deque(maxlen=<span style="color: #098658;">10</span>)
    
    history: deque = _compute_temperature_rate_of_change._history
    
    <span style="color: #008080;"># Get the latest data point from the dependency</span>
    temp_buffer = data_buffers.get(<span style="color: #a31515;">'bme_temperature_celsius'</span>, [])
    <span style="color: #0000ff;">if</span> <span style="color: #0000ff;">not</span> temp_buffer:
        <span style="color: #0000ff;">return</span> <span style="color: #0000ff;">None</span>
        
    latest_time, latest_temp = temp_buffer[-<span style="color: #098658;">1</span>]
    history.append((latest_time, latest_temp))
    
    <span style="color: #008080;"># Need at least two points to calculate a rate</span>
    <span style="color: #0000ff;">if</span> len(history) &lt; <span style="color: #098658;">2</span>:
        <span style="color: #0000ff;">return</span> <span style="color: #0000ff;">None</span>
        
    oldest_time, oldest_temp = history[<span style="color: #098658;">0</span>]
    delta_time = latest_time - oldest_time
    
    <span style="color: #0000ff;">if</span> delta_time &lt; <span style="color: #098658;">0.1</span>: <span style="color: #008080;"># Avoid division by zero or noisy small intervals</span>
        <span style="color: #0000ff;">return</span> <span style="color: #0000ff;">None</span>
        
    delta_temp = latest_temp - oldest_temp
    <span style="color: #0000ff;">return</span> delta_temp / delta_time <span style="color: #008080;"># °C / second</span></code></pre>
</div>

#### Step 2: Define and Register the `DerivedDataDefinition`
In the `# 3. --- Register the Derived Data Handlers ---` section, create a `DerivedDataDefinition` instance for your new calculation and register it using `register_derived_data`. The framework will automatically handle the dependency resolution and execution order.

<div style="background-color: #f0f0f0; border-left: 5px solid #666; padding: 10px; margin: 10px 0; font-family: Consolas, 'Courier New', monospace; font-size: 14px; color: #333;">
<pre><code><span style="color: #008080;"># In section: # 3. --- Register the Derived Data Handlers ---</span>
register_derived_data(
    DerivedDataDefinition(
        data_type=<span style="color: #a31515;">'temp_rate_of_change'</span>, <span style="color: #008080;"># The new key for your derived data</span>
        dependencies=[<span style="color: #a31515;">'bme_temperature_celsius'</span>], <span style="color: #008080;"># List of data_types it needs</span>
        compute_func=_compute_temperature_rate_of_change
    )
)</code></pre>
</div>

<br>

### Scenario: Creating a New Custom GUI Component
Build a new, replay-compatible visualization widget beyond the existing plots and views.

#### Step 1: Define the Component Class and UI
In the `# 5. --- GUI Component Classes ---` section, create a new class that inherits from `BaseGuiComponent`. In its `__init__` method, set up your UI elements and any replay-specific controls.

-   <b style="color:#00579c;">Standard UI:</b> Create your primary visualization widgets (e.g., `QLabel`, `QProgressBar`, custom-painted widgets).
-   <b style="color:#00579c;">Replay UI:</b> Create replay controls (e.g., a `QSlider`) and place them in a container widget. Hide this container by default: `self.replay_controls.setVisible(False)`.
-   <b style="color:#00579c;">Configuration:</b> Read component-specific settings (like titles, data keys, min/max values) from the `config` dictionary argument.

<div style="background-color: #f0f0f0; border-left: 5px solid #666; padding: 10px; margin: 10px 0; font-family: Consolas, 'Courier New', monospace; font-size: 14px; color: #333;">
<pre><code><span style="color: #008080;"># In section: # 5. --- GUI Component Classes ---</span>
<span style="color: #0000ff;">from</span> PyQt6.QtWidgets <span style="color: #0000ff;">import</span> QProgressBar, QVBoxLayout, QSlider, QHBoxLayout

<span style="color: #0000ff;">class</span> <span style="color: #267f99;">MyCustomGaugeWidget</span>(BaseGuiComponent):
    SLIDER_PRECISION = 100 <span style="color: #008080;"># For replay slider resolution</span>

    <span style="color: #0000ff;">def</span> <span style="color: #006400;">__init__</span>(self, config, data_buffers_ref, device_config_ref, parent=<span style="color: #0000ff;">None</span>):
        <span style="color: #0000ff;">super</span>().__init__(config, data_buffers_ref, device_config_ref, parent)
        
        self.data_key = self.config.get(<span style="color: #a31515;">"data_key"</span>)
        layout = QVBoxLayout(self)

        <span style="color: #008080;"># --- Standard UI ---</span>
        self.title_label = QLabel(self.config.get(<span style="color: #a31515;">'title'</span>, <span style="color: #a31515;">'Gauge'</span>))
        self.gauge = QProgressBar()
        self.gauge.setRange(self.config.get(<span style="color: #a31515;">"min"</span>, <span style="color: #098658;">0</span>), self.config.get(<span style="color: #a31515;">"max"</span>, <span style="color: #098658;">100</span>))
        layout.addWidget(self.title_label); layout.addWidget(self.gauge)

        <span style="color: #008080;"># --- Replay UI (initially hidden) ---</span>
        self.replay_controls = QWidget()
        replay_layout = QHBoxLayout(self.replay_controls)
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_label = QLabel(<span style="color: #a31515;">"0.00s"</span>)
        replay_layout.addWidget(QLabel(<span style="color: #a31515;">"Time:"</span>)); replay_layout.addWidget(self.time_slider); replay_layout.addWidget(self.time_label)
        self.replay_controls.setVisible(<span style="color: #0000ff;">False</span>)
        layout.addWidget(self.replay_controls)
        
        <span style="color: #008080;"># Connect signals</span>
        self.time_slider.valueChanged.connect(self._on_slider_scrub)
        
    <span style="color: #0000ff;">def</span> <span style="color: #006400;">get_required_data_types</span>(self) -> <span style="color: #a31515;">Set</span>[<span style="color: #a31515;">str</span>]:
        <span style="color: #0000ff;">return</span> {self.data_key} <span style="color: #0000ff;">if</span> self.data_key <span style="color: #0000ff;">else</span> set()</code></pre>
</div>

#### Step 2: Implement the Core Component Logic
Implement the required methods from `BaseGuiComponent` to handle data updates, clearing, and replay.

-   `update_component(current_relative_time, is_flowing)`: This is the main update entry point. The `is_flowing` boolean is your key to switch between Live and Replay logic.
-   `render_for_time(time_sec)`: This new method will contain the actual rendering logic for a specific point in time. It will be called by both live mode (with the latest time) and replay mode (with the slider's time).
-   `clear_component()`: This must reset your UI to its default state and hide the replay controls.

<div style="background-color: #f0f0f0; border-left: 5px solid #666; padding: 10px; margin: 10px 0; font-family: Consolas, 'Courier New', monospace; font-size: 14px; color: #333;">
<pre><code><span style="color: #008080;"># In MyCustomGaugeWidget class...</span>
<span style="color: #0000ff;">def</span> <span style="color: #006400;">update_component</span>(self, current_relative_time: <span style="color: #a31515;">float</span>, is_flowing: <span style="color: #a31515;">bool</span>):
    <span style="color: #0000ff;">if</span> is_flowing:
        <span style="color: #008080;"># LIVE MODE: Hide slider, render latest data</span>
        self.replay_controls.setVisible(<span style="color: #0000ff;">False</span>)
        self.render_for_time(current_relative_time)
    <span style="color: #0000ff;">else</span>:
        <span style="color: #008080;"># REPLAY MODE: Show slider, update its range</span>
        self.replay_controls.setVisible(<span style="color: #0000ff;">True</span>)
        buffer = self.data_buffers_ref.get(self.data_key, [])
        min_t = buffer[<span style="color: #098658;">0</span>][<span style="color: #098658;">0</span>] <span style="color: #0000ff;">if</span> buffer <span style="color: #0000ff;">else</span> <span style="color: #098658;">0</span>
        max_t = buffer[-<span style="color: #098658;">1</span>][<span style="color: #098658;">0</span>] <span style="color: #0000ff;">if</span> buffer <span style="color: #0000ff;">else</span> <span style="color: #098658;">1</span>
        
        self.time_slider.blockSignals(<span style="color: #0000ff;">True</span>)
        self.time_slider.setRange(<span style="color: #a31515;">int</span>(min_t * self.SLIDER_PRECISION), <span style="color: #a31515;">int</span>(max_t * self.SLIDER_PRECISION))
        self.time_slider.blockSignals(<span style="color: #0000ff;">False</span>)

<span style="color: #0000ff;">def</span> <span style="color: #006400;">_on_slider_scrub</span>(self, value: <span style="color: #a31515;">int</span>):
    time_sec = value / self.SLIDER_PRECISION
    self.time_label.setText(<span style="color: #a31515;">f"{</span>time_sec<span style="color: #a31515;">:.2f}s"</span>)
    self.render_for_time(time_sec)

<span style="color: #0000ff;">def</span> <span style="color: #006400;">render_for_time</span>(self, time_sec: <span style="color: #a31515;">float</span>):
    <span style="color: #008080;"># Central rendering logic uses get_value_at_time</span>
    value = get_value_at_time(self.data_key, time_sec, self.data_buffers_ref)
    <span style="color: #0000ff;">if</span> value <span style="color: #0000ff;">is not</span> <span style="color: #0000ff;">None</span>:
        self.gauge.setValue(<span style="color: #a31515;">int</span>(value))
    <span style="color: #0000ff;">else</span>:
        self.gauge.reset()

<span style="color: #0000ff;">def</span> <span style="color: #006400;">clear_component</span>(self):
    self.gauge.reset()
    self.replay_controls.setVisible(<span style="color: #0000ff;">False</span>)
    self.time_slider.setValue(<span style="color: #098658;">0</span>)
    self.time_label.setText(<span style="color: #a31515;">"0.00s"</span>)
    <span style="color: #0000ff;">super</span>().handle_missing_uuids(set())
    <span style="color: #0000ff;">super</span>().handle_missing_replay_data(set())</code></pre>
</div>

#### Step 3: Use Your New Component in `tab_configs`
Add an entry for your new component class to the `tab_configs` list, providing its required `config` dictionary.

<br>

### Scenario: Modifying Tabs and Layouts
All GUI layout changes are managed declaratively within the `tab_configs` list in section `# 6`. This provides a centralized and readable way to organize the interface.

#### To Add a New Tab
Append a new dictionary to the end of the `tab_configs` list. Each tab needs a unique `tab_title` and a `layout` list, which can start empty or be populated with components.

<div style="background-color: #f0f0f0; border-left: 5px solid #666; padding: 10px; margin: 10px 0; font-family: Consolas, 'Courier New', monospace; font-size: 14px; color: #333;">
<pre><code><span style="color: #008080;"># In tab_configs list:</span>
{
    <span style="color: #a31515;">'tab_title'</span>: <span style="color: #a31515;">'My New Analysis Tab'</span>,
    <span style="color: #a31515;">'layout'</span>: [
        <span style="color: #008080;"># ... add component definitions here ...</span>
    ]
},</code></pre>
</div>

#### To Add or Rearrange Components
Find the target tab's dictionary in `tab_configs` and modify its `layout` list. The layout is a grid, so each component's position is defined by its `row` and `col`.
- **`rowspan` / `colspan`**: You can make a component span multiple cells, for instance, to place a large heatmap next to several smaller plots. `rowspan: 2` makes it two rows tall.
- **Fixed Sizing**: You can suggest a fixed size for a component via its `config`, using `'plot_width'` and `'plot_height'` for `TimeSeriesPlotComponent`, or `'component_width'` and `'component_height'` for others like the `PressureHeatmapComponent`.

<div style="background-color: #f0f0f0; border-left: 5px solid #666; padding: 10px; margin: 10px 0; font-family: Consolas, 'Courier New', monospace; font-size: 14px; color: #333;">
<pre><code><span style="color: #008080;"># Example of a complex layout in a tab's 'layout' list</span>
[
    { <span style="color: #008080;"># Large component on the left, spanning two rows</span>
        <span style="color: #a31515;">'component_class'</span>: PressureHeatmapComponent,
        <span style="color: #a31515;">'row'</span>: <span style="color: #098658;">0</span>, <span style="color: #a31515;">'col'</span>: <span style="color: #098658;">0</span>, <span style="color: #a31515;">'rowspan'</span>: <span style="color: #098658;">2</span>, 
        <span style="color: #a31515;">'config'</span>: { <span style="color: #008080;">...</span> }
    },
    { <span style="color: #008080;"># First small component on the right</span>
        <span style="color: #a31515;">'component_class'</span>: TimeSeriesPlotComponent,
        <span style="color: #a31515;">'row'</span>: <span style="color: #098658;">0</span>, <span style="color: #a31515;">'col'</span>: <span style="color: #098658;">1</span>,
        <span style="color: #a31515;">'config'</span>: { <span style="color: #a31515;">'plot_height'</span>: <span style="color: #098658;">350</span> }
    },
    { <span style="color: #008080;"># Second small component below the first one</span>
        <span style="color: #a31515;">'component_class'</span>: NyquistPlotComponent,
        <span style="color: #a31515;">'row'</span>: <span style="color: #098658;">1</span>, <span style="color: #a31515;">'col'</span>: <span style="color: #098658;">1</span>,
        <span style="color: #a31515;">'config'</span>: { <span style="color: #a31515;">'plot_height'</span>: <span style="color: #098658;">350</span> }
    }
]</code></pre>
</div>

<br>

### Scenario: Configuring Data Logging
Data logging is configured on a per-component-instance basis within `tab_configs`. When a capture session is active, the framework generates two types of CSVs:
1.  **Individual CSVs:** One for each component instance where logging is enabled.
2.  **Master Tab CSVs:** One for each tab, containing an aggregated, time-aligned collection of all loggable data from that tab.

#### Step 1: Enable Logging for a Component Instance
Find the component you want to log data from. In its `'config'` dictionary, add the key-value pair: `'enable_logging': True`.

#### Step 2: (Optional) Customize the Log Filename
The filename for a component's individual CSV is determined by the `get_log_filename_suffix()` method, which defaults to using the component's title from its `config`. You can override this method in the component's class for more specific naming.

<div style="background-color: #f0f0f0; border-left: 5px solid #666; padding: 10px; margin: 10px 0; font-family: Consolas, 'Courier New', monospace; font-size: 14px; color: #333;">
<pre><code><span style="color: #008080;"># In your component's class definition:</span>
<span style="color: #0000ff;">def</span> <span style="color: #006400;">get_log_filename_suffix</span>(self) -> <span style="color: #a31515;">str</span>:
    <span style="color: #0000ff;">if</span> self.is_loggable:
        <span style="color: #008080;"># Return a file-safe string for the CSV filename</span>
        <span style="color: #0000ff;">return</span> <span style="color: #a31515;">"log_primary_temperature_sensor"</span>
    <span style="color: #0000ff;">return</span> <span style="color: #a31515;">""</span></code></pre>
</div>

#### Step 3: (Optional) Customize Logged Data Types
By default, a loggable component saves all data types returned by its `get_required_data_types()` method. To log a different or more specific set of data, you can override the `get_loggable_data_types()` method in the component's class definition. This is useful for complex components that visualize many things but you only want to log a subset.

<div style="background-color: #f0f0f0; border-left: 5px solid #666; padding: 10px; margin: 10px 0; font-family: Consolas, 'Courier New', monospace; font-size: 14px; color: #333;">
<pre><code><span style="color: #008080;"># In your component's class definition:</span>
<span style="color: #0000ff;">def</span> <span style="color: #006400;">get_loggable_data_types</span>(self) -> <span style="color: #a31515;">Set</span>[<span style="color: #a31515;">str</span>]:
    <span style="color: #008080;"># Override to log only a specific subset of its required data</span>
    <span style="color: #0000ff;">if</span> self.is_loggable:
        <span style="color: #0000ff;">return</span> {<span style="color: #a31515;">'bme_temperature_celsius'</span>, <span style="color: #a31515;">'bme_pressure_pascal'</span>}
    <span style="color: #0000ff;">return</span> set()</code></pre>
</div>