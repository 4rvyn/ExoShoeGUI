# Backend & Core Logic (Advanced)
<p><span style="color:red"><b>Warning:</b> Modifications to the code described in this section can easily break core application functionality. Proceed with extreme caution and only if you have a deep understanding of the application's architecture, asynchronous programming (<code>asyncio</code>, <code>qasync</code>), and BLE (<code>bleak</code>).</span></p>

This section provides a detailed technical breakdown of the application's non-customizable backend. It is intended to offer a deep architectural understanding for developers, not for modification during typical use. The system is designed around a set of communicating state machines and a central data repository.

<br>

### The Grand Orchestrator: GUI and Application State Machine
The entire application's operational mode is governed by a high-level Finite State Machine (FSM) managed by the `MainWindow` class. This FSM dictates the user's available actions and the overall behavior of the GUI.

#### Core Application States
The global `state` variable, controlled by `MainWindow`, can be in one of the following states:
*   `"idle"`: The default state. The application is waiting for user action. The user can start a BLE scan or load a CSV file for replay.
*   `"scanning"`: The application is actively searching for a BLE device. User interaction is locked until the scan completes, is stopped, or times out.
*   `"connected"`: A connection to a BLE device is active. Data is being received, and the GUI is live. The user can pause plotting, start/stop data capture, or disconnect.
*   `"disconnecting"`: A transient state where the application is cleaning up the BLE connection. GUI interaction is disabled.
*   `"replay_active"`: The application is not connected to a live device. Instead, it has loaded data from one or more CSV files. In this mode, live controls are hidden, and replay-specific controls (like time-scrubbing sliders) are enabled.

#### State Transitions and Actions
The `MainWindow.handle_state_change()` method is the heart of this FSM. When the state changes, it orchestrates the GUI by enabling/disabling buttons, changing text labels, and controlling timers. This ensures the UI is always consistent with the application's backend status.

<div style="background-color: #f0f0f0; border-left: 5px solid #666; padding: 10px; margin: 10px 0; font-family: Consolas, 'Courier New', monospace; font-size: 14px; color: #333;">
<pre><code><span style="color: #008080;"># In MainWindow.handle_state_change(self, new_state: str):</span>

<span style="color: #0000ff;">def</span> <span style="color: #006400;">handle_state_change</span>(self, new_state: <span style="color: #a31515;">str</span>):
    <span style="color: #0000ff;">global</span> state, plotting_paused, start_time
    logger.info(<span style="color: #a31515;">f"GUI received state change: {</span>new_state<span style="color: #a31515;">}"</span>)
    state = new_state

    <span style="color: #008080;"># --- Example actions for different states ---</span>

    <span style="color: #0000ff;">if</span> state == <span style="color: #a31515;">"idle"</span>:
        self.scan_button.setText(<span style="color: #a31515;">"Start Scan"</span>)
        self.led_indicator.set_color(<span style="color: #a31515;">"red"</span>)
        self.status_label.setText(<span style="color: #a31515;">"On Standby"</span>)
        self.capture_button.setEnabled(<span style="color: #0000ff;">False</span>)
        <span style="color: #008080;"># ... etc.</span>
    
    <span style="color: #0000ff;">elif</span> state == <span style="color: #a31515;">"connected"</span>:
        self.scan_button.setText(<span style="color: #a31515;">"Disconnect"</span>)
        self.led_indicator.set_color(<span style="color: #a31515;">"lightgreen"</span>)
        self.status_label.setText(<span style="color: #a31515;">f"Connected to: {</span>device_config.name<span style="color: #a31515;">}"</span>)
        self.capture_button.setEnabled(<span style="color: #0000ff;">True</span>)
        plotting_paused = <span style="color: #0000ff;">False</span>
        <span style="color: #008080;"># ... etc.</span>

    <span style="color: #0000ff;">elif</span> state == <span style="color: #a31515;">"replay_active"</span>:
        self.scan_button.setVisible(<span style="color: #0000ff;">False</span>)
        self.replay_button.setText(<span style="color: #a31515;">"Exit Replay"</span>)
        self.led_indicator.set_color(<span style="color: #a31515;">"purple"</span>)
        self.flowing_mode_check.setEnabled(<span style="color: #0000ff;">False</span>)
        <span style="color: #008080;"># ... etc.</span>
</code></pre>
</div>

<br>

### The Engine Room: Data Sources & BLE Logic
The actual acquisition of data is abstracted away behind a `DataSource` interface. This polymorphic design allows the `MainWindow` to start and stop data flow without needing to know if the source is a live BLE device or a replay from a CSV file.

#### The `BleDataSource` State Machine
The `BleDataSource` class encapsulates all BLE-related logic and manages its own intricate, asynchronous state machine within its `start()` method. This is where the core `asyncio` and `bleak` operations occur.

**Key `BleDataSource` States & Operations:**

1.  **Scanning:**
    *   **Action:** The `start()` method first calls `find_device()`, which instantiates a `BleakScanner`.
    *   **Mechanism:** A `detection_callback` is registered with the scanner. This callback checks every discovered device's name and advertised services against the active `device_config`.
    *   **Transition:** Upon finding a match, an `asyncio.Event` is set, which `find_device()` was waiting for. It then returns the `BleakDevice` object. On timeout or cancellation, it returns `None`.

2.  **Connecting:**
    *   **Action:** If a device is found, `start()` attempts to connect using `await self._client.connect()`. This is wrapped in a retry loop to handle transient connection failures.
    *   **Mechanism:** `BleakClient` handles the low-level GATT connection process. A `disconnected_callback` is registered during client instantiation, which will be invoked by `bleak`'s backend if the connection drops unexpectedly.
    *   **Transition:** On success, the `MainWindow` state is set to `"connected"`. On failure after all retries, the state reverts to `"idle"`.

3.  **Subscribing & Listening (The "Connected" state):**
    *   **Action:** Once connected, the `start()` method verifies that the configured service UUID exists. It then iterates through all characteristics in `device_config`, checks if they exist on the device and support notifications, and calls `await self._client.start_notify()` for each valid one.
    *   **Mechanism:** The `start_notify` call registers the application's `notification_handler` callback with `bleak`. From this point on, `bleak`'s event loop will automatically invoke `notification_handler` whenever the device sends data.
    *   **The Listening Loop:** The function then enters its main `while` loop, which is the core of the active connection. It does not actively poll for data. Instead, it waits asynchronously for one of two things to happen:
        1.  `disconnected_event.wait()`: It waits on an `asyncio.Event`. This event is set by the `disconnected_callback` (if the device disconnects) or by the `stop()` method (if the user clicks "Disconnect"). This is the primary mechanism for exiting the loop cleanly.
        2.  **Data Timeout:** If the `wait()` times out (e.g., after 0.2s), it checks if `time.time() - last_received_time` has exceeded the configured `data_timeout`. If so, it assumes the connection has been lost and breaks the loop.

4.  **Disconnecting:**
    *   **Action:** This logic resides in the `finally` block of the `start()` method, ensuring it *always* runs, regardless of how the listening loop was exited (clean disconnect, error, timeout, or cancellation).
    *   **Mechanism:** It systematically calls `await self._client.stop_notify()` for all active characteristics and finally `await self._client.disconnect()`. This graceful cleanup is critical for releasing BLE resources on both the computer and the peripheral device.

<div style="background-color: #f0f0f0; border-left: 5px solid #666; padding: 10px; margin: 10px 0; font-family: Consolas, 'Courier New', monospace; font-size: 14px; color: #333;">
<pre><code><span style="color: #008080;"># Simplified structure of BleDataSource.start()</span>
<span style="color: #0000ff;">async</span> <span style="color: #0000ff;">def</span> <span style="color: #006400;">start</span>(self):
    <span style="color: #0000ff;">try</span>:
        <span style="color: #008080;"># --- Scanning State ---</span>
        target_device = <span style="color: #0000ff;">await</span> find_device(...)
        <span style="color: #0000ff;">if</span> <span style="color: #0000ff;">not</span> target_device: <span style="color: #0000ff;">return</span>

        <span style="color: #008080;"># --- Connecting State ---</span>
        self._client = BleakClient(target_device, ...)
        <span style="color: #0000ff;">await</span> self._client.connect(...)
        self._gui_emitter.emit_state_change(<span style="color: #a31515;">"connected"</span>)

        <span style="color: #008080;"># --- Subscribing State ---</span>
        <span style="color: #0000ff;">await</span> self._client.start_notify(uuid, notification_handler_partial)
        <span style="color: #008080;"># ... (for all characteristics)</span>
        
        <span style="color: #008080;"># --- Listening State ---</span>
        <span style="color: #0000ff;">while</span> <span style="color: #0000ff;">not</span> self._stop_requested <span style="color: #0000ff;">and</span> self._client.is_connected:
            <span style="color: #0000ff;">try</span>:
                <span style="color: #0000ff;">await</span> asyncio.wait_for(disconnected_event.wait(), timeout=<span style="color: #098658;">0.2</span>)
                <span style="color: #0000ff;">break</span> <span style="color: #008080;"># Event was set, break loop</span>
            <span style="color: #0000ff;">except</span> asyncio.TimeoutError:
                <span style="color: #008080;"># Check for data timeout, then continue waiting</span>
                <span style="color: #0000ff;">if</span> time.time() - last_received_time > self._device_config.data_timeout:
                    <span style="color: #0000ff;">break</span>
    <span style="color: #0000ff;">finally</span>:
        <span style="color: #008080;"># --- Disconnecting State (Cleanup) ---</span>
        <span style="color: #0000ff;">if</span> local_client_ref <span style="color: #0000ff;">and</span> local_client_ref.is_connected:
            <span style="color: #0000ff;">await</span> local_client_ref.stop_notify(...)
            <span style="color: #0000ff;">await</span> local_client_ref.disconnect()
</code></pre>
</div>

<br>

### The Dynamic Builder: GUI Management and Rendering
The user interface is not hard-coded. Instead, it is constructed dynamically at startup by the `GuiManager` class based on the `tab_configs` list.

#### Key Responsibilities of `GuiManager`
*   **Dynamic Instantiation:** In its `create_gui_layout` method, it iterates through the `tab_configs`. For each entry, it creates the specified `component_class`, passing it the relevant `config` dictionary, and adds the component's widget to a `QGridLayout`. This makes the entire UI structure declarative and easy to modify.
*   **Component Aggregation:** It holds a master list of all component instances (`self.all_components`). This allows `MainWindow` to issue a single command to the `GuiManager` (e.g., "update all"), which then iterates through the list and calls the appropriate method on each component.
*   **Update and Clear Propagation:** The `MainWindow`'s update timer triggers `GuiManager.update_all_components()`, which in turn calls `component.update_component()` on every active GUI element. The same pattern applies to `clear_components()`.
*   **Targeted Notification:** When the `BleDataSource` reports a set of missing characteristic UUIDs, the `GuiManager` receives this set. It then intelligently notifies only the relevant components. It does this by querying each component for its `get_required_data_types()` and cross-referencing those types with the missing UUIDs, ensuring that a "Missing UUID" overlay only appears on the specific plots or widgets affected.

#### The `qasync` and `asyncio` Integration
A crucial piece of the architecture is the use of `qasync`. It seamlessly merges Python's `asyncio` event loop (which `bleak` requires for non-blocking network I/O) with Qt's event loop (which handles UI events like button clicks and repaints). Without this, any `await` call for a BLE operation would freeze the entire GUI. `qasync` allows the `BleDataSource.start()` method to run concurrently with the UI, ensuring a responsive user experience while long-running background tasks are active.

<br>

### The Central Repository: Data Buffering & Export
All sensor data, whether from a live BLE stream or a replayed CSV, is stored in a single, globally accessible dictionary named `data_buffers`.

#### Data Buffer Structure
This dictionary serves as the "single source of truth" for the application's data.

*   **Structure:** `Dict[str, List[Tuple[float, float]]]`
*   **Key:** The `data_type` string (e.g., `'orientation_x'`, `'estimated_weight'`). This key must be unique application-wide.
*   **Value:** A list of tuples, where each tuple is `(relative_time_seconds, value)`. `relative_time_seconds` is the time elapsed since the session began.

#### Data Flow
1.  **Population (Write):**
    *   **Live Data:** The `notification_handler` is the primary writer. When new data arrives from a characteristic, it calls the appropriate parser function, gets the `data_type: value` dictionary, calculates the relative time, and appends the new `(time, value)` tuple to the correct list in `data_buffers`.
    *   **Derived Data:** Immediately after raw data is added, `compute_all_derived_data` is called. It checks all `DerivedDataDefinition`s to see if their dependencies are met and, if so, computes the new value and appends it to `data_buffers` using the *same timestamp* as the triggering raw sample.
    *   **Replay Data:** In replay mode, the `CsvReplaySource` reads a CSV into a pandas DataFrame and then iterates through it, populating `data_buffers` in a single bulk operation.

2.  **Consumption (Read):**
    *   **GUI Components:** During each update cycle, every GUI component reads directly from `data_buffers` to get the latest values it needs for rendering.
    *   **Data Export:** When a capture is stopped, the `generate_csv_files_from_buffer` and `generate_pdf_plots_from_buffer` functions read from `data_buffers`, filter the data to the captured time window, and write the output files.

#### Data Export Mechanism
The export process, managed by `MainWindow`, is designed to be comprehensive.
*   **Capture Window:** When "Start Capture" is pressed, the current relative time is stored. When "Stop Capture" is pressed, a new relative time is recorded, defining the precise time window for the export.
*   **CSV Generation:**
    *   **Individual CSVs:** The framework iterates through all loggable components (those with `'enable_logging': True`). For each, it creates a dedicated CSV file containing only the data types that component requires (`get_loggable_data_types()`).
    *   **Master Tab CSVs:** For each tab in the GUI, it gathers all unique loggable data types from all loggable components on that tab. It then creates a single, time-aligned CSV containing all of these data series, providing a comprehensive view of the tab's activity.
*   **PDF Generation:** The framework iterates through all `TimeSeriesPlotComponent` instances and generates a high-quality Matplotlib-based PDF of the data captured within the defined window.