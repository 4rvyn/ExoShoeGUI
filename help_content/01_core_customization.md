# Core Customization Areas
This section outlines the primary Python objects and structures you will modify when customizing the application. Understanding these five areas is key to extending the software's functionality.

<br>

### 1. Data Handlers (`handle_*` functions)
<b style="color:#00579c;">Purpose:</b> Data handlers are the entry point for sensor data into the application. They are Python functions responsible for parsing the raw `bytearray` received from a BLE characteristic into a meaningful, structured dictionary.

<b style="color:#00579c;">Location:</b> Defined in the customizable section, under the comment `# 1. --- Data Handlers for Different Characteristics ---`.

#### Function Signature & Responsibilities
A handler must adhere to a specific signature and perform several key tasks:
- **Input:** Accepts a single `data: bytearray` argument.
- **Output:** Returns a `Dict[str, float]`. The keys of this dictionary become the unique `data_type` identifiers used throughout the application (e.g., `'temperature_celsius'`, `'quat_w'`). <span style="color:#b71c1c; font-weight:bold;">These keys must be unique across the entire application.</span>
- **Parsing:** Contains the logic to convert raw bytes into numbers. This can involve decoding text (`data.decode()`), unpacking binary structures (`struct.unpack`), or custom byte manipulation.
- **Error Handling:** Must be robust. Use a `try...except` block to catch any parsing errors and return an empty dictionary `{}` on failure. This prevents malformed data from crashing the application.
- **Logging:** Use `data_logger.info(...)` or `data_logger.debug(...)` to print parsed values for debugging. This output is controlled by the "Log Raw Data to Console" checkbox in the GUI.

<div style="background-color: #f0f0f0; border-left: 5px solid #666; padding: 10px; margin: 10px 0; font-family: Consolas, 'Courier New', monospace; font-size: 14px; color: #333;">
<pre><code><span style="color: #0000ff;">def</span> <span style="color: #006400;">handle_temp_humidity_data</span>(data: <span style="color: #a31515;">bytearray</span>) -> <span style="color: #a31515;">dict</span>:
    <span style="color: #008080;"># Example: 4-byte payload. Temp (int16), Humidity (uint16)</span>
    <span style="color: #0000ff;">try</span>:
        <span style="color: #0000ff;">if</span> len(data) &lt; <span style="color: #098658;">4</span>:
            data_logger.warning(<span style="color: #a31515;">"Temp/Humidity handler received short payload"</span>)
            <span style="color: #0000ff;">return</span> {}

        <span style="color: #008080;"># Unpack two little-endian integers (signed short, unsigned short)</span>
        raw_temp, raw_humidity = struct.unpack(<span style="color: #a31515;">"&lt;hH"</span>, data)
        
        temperature_c = raw_temp / <span style="color: #098658;">100.0</span>
        humidity_percent = raw_humidity / <span style="color: #098658;">100.0</span>
        
        data_logger.info(<span style="color: #a31515;">f"Parsed Temp: {</span>temperature_c<span style="color: #a31515;">:.2f}°C, Humidity: {</span>humidity_percent<span style="color: #a31515;">:.2f}%"</span>)
        <span style="color: #0000ff;">return</span> {
            <span style="color: #a31515;">"ambient_temperature_celsius"</span>: temperature_c,
            <span style="color: #a31515;">"ambient_humidity_percent"</span>: humidity_percent
        }
    <span style="color: #0000ff;">except</span> <span style="color: #a31515;">Exception</span> <span style="color: #0000ff;">as</span> e:
        data_logger.error(<span style="color: #a31515;">f"Error parsing temp/humidity data: {e}"</span>)
        <span style="color: #0000ff;">return</span> {}</code></pre>
</div>

<br>

### 2. Derived Data Framework
<b style="color:#00579c;">Purpose:</b> This powerful framework allows you to create new, "virtual" data series by performing calculations on existing raw or other derived data. This is ideal for sensor fusion, calculating rates of change, or combining values.

<b style="color:#00579c;">Location:</b> Logic is defined in sections `# 2` and `# 3`.

#### Workflow
The process involves three simple steps. The framework handles the execution automatically whenever new data arrives.

1.  **Define a Computation Function:** In section `# 2`, write a Python function that performs your desired calculation. It must read its inputs from the global `data_buffers` and should return `None` if the necessary data isn't available. For history-dependent calculations (like a derivative), use a `deque` attached as a function attribute to maintain state between calls.

2.  **Create a `DerivedDataDefinition`:** This class instance bundles your computation's metadata:
    - `data_type (str)`: The unique name for your new derived data series.
    - `dependencies (List[str])`: A list of the `data_type` strings it needs as input. The framework ensures the function only runs if this data is present.
    - `compute_func (Callable)`: A reference to your computation function.

3.  **Register the Definition:** In section `# 3`, pass your `DerivedDataDefinition` instance to the `register_derived_data()` function.

<div style="background-color: #f0f0f0; border-left: 5px solid #666; padding: 10px; margin: 10px 0; font-family: Consolas, 'Courier New', monospace; font-size: 14px; color: #333;">
<pre><code><span style="color: #008080;"># 1. Computation Function (from the application source)</span>
<span style="color: #0000ff;">def</span> <span style="color: #006400;">_compute_dZ_dt</span>(min_span_sec: <span style="color: #a31515;">float</span> = <span style="color: #098658;">0.08</span>, ...) -> <span style="color: #a31515;">Optional</span>[<span style="color: #a31515;">float</span>]:
    <span style="color: #008080;"># ... (initializes a history deque on the function itself) ...</span>
    
    buf = data_buffers.get(<span style="color: #a31515;">'impedance_magnitude_ohm'</span>, [])
    <span style="color: #0000ff;">if not</span> buf: <span style="color: #0000ff;">return None</span>
    
    <span style="color: #008080;"># ... (appends to history, performs least-squares slope calculation) ...</span>
    <span style="color: #0000ff;">return</span> <span style="color: #a31515;">float</span>(slope)

<span style="color: #008080;"># 2 & 3. Definition and Registration</span>
register_derived_data(
    DerivedDataDefinition(
        data_type=<span style="color: #a31515;">'impedance_change_speed_ohm_per_s'</span>,
        dependencies=[<span style="color: #a31515;">'impedance_magnitude_ohm'</span>], <span style="color: #008080;"># Depends on this raw data type</span>
        compute_func=_compute_dZ_dt,
    )
)</code></pre>
</div>
<br>

### 3. Device & Characteristic Configuration
<b style="color:#00579c;">Purpose:</b> These structures form the bridge between the BLE world (UUIDs) and the application's data processing logic (handlers and `data_type`s).

<b style="color:#00579c;">Location:</b> Defined in section `# 4`.

#### The `DeviceConfig` Object
This global object holds the configuration for a target BLE device profile.
- `name (str)`: The advertised BLE name of the device. This is used for discovery.
- `service_uuid (str)`: The main BLE Service UUID that contains the characteristics of interest.
- `characteristics (List[CharacteristicConfig])`: A list defining each data-producing characteristic.

#### The `CharacteristicConfig` Class
Each instance in the `characteristics` list links a UUID to a handler.
- `uuid (str)`: The specific 128-bit BLE Characteristic UUID.
- `handler (Callable)`: A reference to the data handler function (from Area 1) for this UUID.
- `produces_data_types (List[str])`: **Crucial.** A list of all `data_type` keys that the specified `handler` function is expected to return. This allows the application to know which data comes from which UUID.

<div style="background-color: #f0f0f0; border-left: 5px solid #666; padding: 10px; margin: 10px 0; font-family: Consolas, 'Courier New', monospace; font-size: 14px; color: #333;">
<pre><code><span style="color: #008080;"># Example from the application source</span>
device_config = DeviceConfig(
    name=<span style="color: #a31515;">"Nano33IoT"</span>,
    service_uuid=<span style="color: #a31515;">"19B10000-E8F2-537E-4F6C-D104768A1214"</span>,
    characteristics=[
        CharacteristicConfig(
            uuid=<span style="color: #a31515;">"19B10001-E8F2-537E-4F6C-D104768A1214"</span>,
            handler=handle_orientation_data,
            produces_data_types=[<span style="color: #a31515;">'orientation_x'</span>, <span style="color: #a31515;">'orientation_y'</span>, <span style="color: #a31515;">'orientation_z'</span>]
        ),
        <span style="color: #008080;"># ... more CharacteristicConfig entries ...</span>
    ],
    <span style="color: #008080;"># ...</span>
)</code></pre>
</div>
<br>

### 4. Creating Custom GUI Components
<b style="color:#00579c;">Purpose:</b> GUI components are modular PyQt6 widgets responsible for visualizing data. You can create entirely new types of visualizations by subclassing `BaseGuiComponent`.

<b style="color:#00579c;">Location:</b> Defined in section `# 5`.

#### The `BaseGuiComponent` Contract
Any custom component **must** inherit from `BaseGuiComponent` and implement its core methods. This ensures it integrates correctly with the application's update loop, data system, and replay functionality.

**Essential Methods to Implement:**
- `__init__(self, config, data_buffers_ref, device_config_ref, parent)`: The constructor.
  - Always call `super().__init__(...)`.
  - Use the `config` dictionary for per-instance settings (titles, data keys, etc.).
  - Set up your UI elements (e.g., `QLabel`, `QProgressBar`, a custom-painted widget).
  - Set up and hide any replay-specific controls (like sliders).
- `get_required_data_types(self) -> Set[str]`: Must return a `set` of all `data_type` strings this component needs to function.
- `update_component(self, current_relative_time, is_flowing)`: Called periodically to refresh the display. This is the entry point for updates. Use the `is_flowing` boolean to determine if you are in Live Mode (`True`) or Replay Mode (`False`).
- `clear_component(self)`: Must reset the component's UI and internal state to a clean, default appearance. This should also hide replay controls.

#### Implementing Replay Functionality
To make a component replay-compatible, follow this pattern within your class:
1.  **In `update_component`:**
    - If `is_flowing` is `True` (Live): Hide your replay controls. Call a central rendering method (e.g., `self.render_for_time()`) with the `current_relative_time`.
    - If `is_flowing` is `False` (Replay): Show your replay controls. Check the `data_buffers_ref` for the full time range of your required data and set your slider's min/max values.
2.  **Create a slider handler (`_on_slider_scrub`):** This method is connected to the slider's `valueChanged` signal. It converts the slider's integer value to a time in seconds and then calls your central rendering method with that time.
3.  **Create a central rendering method (`render_for_time`):** This method takes a `time_sec` argument. It uses the `get_value_at_time(data_type, time_sec, self.data_buffers_ref)` utility to fetch the historical state of its data and updates the UI accordingly.

<div style="background-color: #f0f0f0; border-left: 5px solid #666; padding: 10px; margin: 10px 0; font-family: Consolas, 'Courier New', monospace; font-size: 14px; color: #333;">
<pre><code><span style="color: #008080;"># Abridged example of a replay-compatible component</span>
<span style="color: #0000ff;">class</span> <span style="color: #267f99;">MyCustomGaugeWidget</span>(BaseGuiComponent):
    <span style="color: #0000ff;">def</span> <span style="color: #006400;">update_component</span>(self, current_relative_time, is_flowing):
        <span style="color: #0000ff;">if</span> is_flowing:
            self.replay_controls.setVisible(<span style="color: #0000ff;">False</span>)
            self.render_for_time(current_relative_time)
        <span style="color: #0000ff;">else</span>:
            self.replay_controls.setVisible(<span style="color: #0000ff;">True</span>)
            <span style="color: #008080;"># ... logic to set slider range from data_buffers ...</span>

    <span style="color: #0000ff;">def</span> <span style="color: #006400;">_on_slider_scrub</span>(self, value):
        time_sec = value / self.SLIDER_PRECISION
        self.render_for_time(time_sec)

    <span style="color: #0000ff;">def</span> <span style="color: #006400;">render_for_time</span>(self, time_sec):
        value = get_value_at_time(self.data_key, time_sec, self.data_buffers_ref)
        <span style="color: #008080;"># ... update self.gauge with the historical value ...</span>

    <span style="color: #0000ff;">def</span> <span style="color: #006400;">clear_component</span>(self):
        self.gauge.reset()
        self.replay_controls.setVisible(<span style="color: #0000ff;">False</span>)
        <span style="color: #008080;"># ...</span></code></pre>
</div>
<br>

### 5. Tab Layout Configuration (`tab_configs`)
<b style="color:#00579c;">Purpose:</b> `tab_configs` is a single list of dictionaries that defines the entire GUI layout, including all tabs and the components within them. It provides a centralized, human-readable way to organize the interface without writing complex UI code.

<b style="color:#00579c;">Location:</b> Defined in section `# 6`.

#### `tab_configs` Structure
`tab_configs` is a `List[Dict]`. Each dictionary in the list defines one tab from left to right.

```python
tab_configs = [
    { # Definition for the first tab
        'tab_title': 'Insole View',
        'layout': [ ... list of component definitions for this tab ... ]
    },
    { # Definition for the second tab
        'tab_title': 'IMU 3D View',
        'layout': [ ... list of component definitions for this tab ... ]
    },
]
```

#### Component Definition Structure
Each dictionary inside a tab's `'layout'` list defines a single component instance and its properties.
- `'component_class'`: A direct reference to the component's Python class (e.g., `TimeSeriesPlotComponent`).
- `'row'`, `'col'`: The component's top-left starting position in the tab's grid.
- `'rowspan'`, `'colspan'` (Optional): How many rows or columns the component should span. Defaults to 1.
- `'config'`: A dictionary of settings passed directly to the component's `__init__` method. This is where you customize titles, axis labels, data sources, colors, and behavior for that specific instance.
- `'enable_logging'` (in `config`): A standard key. Setting `'enable_logging': True` marks this component's data for inclusion in CSV exports during a capture session.

<div style="background-color: #f0f0f0; border-left: 5px solid #666; padding: 10px; margin: 10px 0; font-family: Consolas, 'Courier New', monospace; font-size: 14px; color: #333;">
<pre><code><span style="color: #008080;"># Example of one component definition within a tab's 'layout' list:</span>
{
    <span style="color: #a31515;">'component_class'</span>: TimeSeriesPlotComponent,
    <span style="color: #a31515;">'row'</span>: <span style="color: #098658;">0</span>, <span style="color: #a31515;">'col'</span>: <span style="color: #098658;">0</span>,
    <span style="color: #a31515;">'rowspan'</span>: <span style="color: #098658;">1</span>, <span style="color: #a31515;">'colspan'</span>: <span style="color: #098658;">2</span>, <span style="color: #008080;"># Make this plot span two columns</span>
    <span style="color: #a31515;">'config'</span>: {
        <span style="color: #a31515;">'title'</span>: <span style="color: #a31515;">'Orientation Data'</span>,
        <span style="color: #a31515;">'ylabel'</span>: <span style="color: #a31515;">'Degrees'</span>,
        <span style="color: #a31515;">'datasets'</span>: [
            {<span style="color: #a31515;">'data_type'</span>: <span style="color: #a31515;">'orientation_x'</span>, <span style="color: #a31515;">'label'</span>: <span style="color: #a31515;">'Roll'</span>, <span style="color: #a31515;">'color'</span>: <span style="color: #a31515;">'r'</span>},
            {<span style="color: #a31515;">'data_type'</span>: <span style="color: #a31515;">'orientation_y'</span>, <span style="color: #a31515;">'label'</span>: <span style="color: #a31515;">'Pitch'</span>, <span style="color: #a31515;">'color'</span>: <span style="color: #a31515;">'g'</span>}
        ],
        <span style="color: #a31515;">'enable_logging'</span>: <span style="color: #0000ff;">True</span>
    }
}</code></pre>
</div>