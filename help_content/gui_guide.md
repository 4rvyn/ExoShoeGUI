# 🔵 Application GUI Guide

Welcome! This guide explains the application's interface, features, and controls to help you collect and analyze sensor data.

---

## ✨ Core Concepts

The application has two main modes: **Live Mode** for real-time data from a connected device, and **Replay Mode** for analyzing data from saved files. The color of the **Status LED** in the top-left corner always shows the current mode and state.

### Application Status (LED)

The LED provides at-a-glance information about the application's state.

<table>
  <thead>
    <tr>
      <th style="padding: 6px 15px; text-align: left;">Color</th>
      <th style="padding: 6px 15px; text-align: left;">Status</th>
      <th style="padding: 6px 15px; text-align: left;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 6px 15px;"><span style="color:#D32F2F"><b>Red</b></span></td>
      <td style="padding: 6px 15px;"><b>Idle / Disconnected</b></td>
      <td style="padding: 6px 15px;">The application is waiting for user action. No device is connected.</td>
    </tr>
    <tr>
      <td style="padding: 6px 15px;"><span style="color:#F57C00"><b>Orange</b></span></td>
      <td style="padding: 6px 15px;"><b>Working...</b></td>
      <td style="padding: 6px 15px;">Scanning for, connecting to, or disconnecting from a device.</td>
    </tr>
    <tr>
      <td style="padding: 6px 15px;"><span style="color:#388E3C"><b>Green</b></span></td>
      <td style="padding: 6px 15px;"><b>Connected (Live Mode)</b></td>
      <td style="padding: 6px 15px;">Successfully connected. Receiving and displaying live data.</td>
    </tr>
    <tr>
      <td style="padding: 6px 15px;"><span style="color:#7B1FA2"><b>Purple</b></span></td>
      <td style="padding: 6px 15px;"><b>Replay Mode</b></td>
      <td style="padding: 6px 15px;">Analyzing data loaded from CSV files. No live connection is active.</td>
    </tr>
  </tbody>
</table>

---

## 🗺️ The Main Interface

The GUI is organized into four primary sections:

1.  **Top Control Bar:** Your main command center for managing device connections, data capture, and operating modes.
2.  **Main Tab Area:** The central workspace where data visualizations like plots, heatmaps, and 3D models are displayed in tabs.
3.  **Bottom Control Bar:** Contains controls for plot behavior, advanced logging, and help access.
4.  **Log Panel:** A text area at the very bottom that displays application status messages, warnings, and errors.

---

## 🧭 Top Control Bar

This is your primary hub for interacting with the application. The available controls adapt to the current mode.

<table>
  <thead>
    <tr>
      <th style="padding: 6px 15px; text-align: left;">Control</th>
      <th style="padding: 6px 15px; text-align: left;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 6px 15px; vertical-align: top;"><code>Target</code> Dropdown</td>
      <td style="padding: 6px 15px;"><span style="color:#388E3C"><b>(Live Mode Only)</b></span> Select a device to connect to from the dropdown list.</td>
    </tr>
    <tr>
      <td style="padding: 6px 15px; vertical-align: top;"><code>Start Scan</code> / <code>Stop Scan</code> / <code>Disconnect</code></td>
      <td style="padding: 6px 15px;"><span style="color:#388E3C"><b>(Live Mode Only)</b></span> Manages the connection to a device. The button's label and function change with the connection state.</td>
    </tr>
    <tr>
      <td style="padding: 6px 15px; vertical-align: top;"><code>Replay CSV...</code> / <code>Exit Replay</code></td>
      <td style="padding: 6px 15px;">Manages <b>Replay Mode</b>. <br>• <b><code>Replay CSV...</code></b>: Opens a dialog to select CSV files for analysis. <br>• <b><code>Exit Replay</code></b>: Exits Replay Mode, clears all loaded data, and returns to Idle.</td>
    </tr>
    <tr>
      <td style="padding: 6px 15px; vertical-align: top;"><code>Load More CSVs</code></td>
      <td style="padding: 6px 15px;"><span style="color:#7B1FA2"><b>(Replay Mode Only)</b></span> Allows you to load additional CSV files into the current replay session.<br>> ⚠️ <b>Note:</b> If a new CSV contains data that is already loaded, the old data for those specific streams will be <b>overwritten</b>.</td>
    </tr>
    <tr>
      <td style="padding: 6px 15px; vertical-align: top;"><code>Pause Plotting</code> / <code>Resume Plotting</code></td>
      <td style="padding: 6px 15px;"><span style="color:#388E3C"><b>(Live Mode Only)</b></span> Toggles real-time updating of all plots. Data collection continues in the background while paused.</td>
    </tr>
    <tr>
      <td style="padding: 6px 15px; vertical-align: top;"><code>Start Capture</code> / <code>Stop Capture & Export</code></td>
      <td style="padding: 6px 15px;"><span style="color:#388E3C"><b>(Live Mode Only)</b></span> Manages data recording.<br>• <b><code>Start Capture</code></b>: Begins recording all incoming data streams.<br>• <b><code>Stop Capture & Export</code></b>: Stops recording and saves the data to CSV and PDF files in a timestamped folder inside the `Logs` directory.</td>
    </tr>
    <tr>
      <td style="padding: 6px 15px; vertical-align: top;"><code>Clear GUI</code></td>
      <td style="padding: 6px 15px;">Wipes all data from memory, resets all plots, and clears "missing data" overlays.<br>> ⚠️ <b>Warning:</b> If a data capture is active, it will be stopped <b>without saving</b>.</td>
    </tr>
    <tr>
      <td style="padding: 6px 15px; vertical-align: top;">Status Label</td>
      <td style="padding: 6px 15px;">Displays real-time information, such as "Scanning...", "Connected", "Replay Data Ready", or error messages.</td>
    </tr>
  </tbody>
</table>

---

## 🔄 Understanding Replay Mode

Replay Mode is a powerful feature for detailed analysis of recorded data without a live device.

#### **Entering and Exiting**

*   **Enter:** Click the <span style="color:#0277BD">`Replay CSV...`</span> button when the application is idle (🔴 LED is Red). You can select one or multiple CSV files.
*   **Exit:** Click the <span style="color:#D32F2F">`Exit Replay`</span> button. This clears all data and returns the application to the Idle state.

> 💡 **Tip:** For the application to work correctly, the CSV files must contain a time column named either `Time (s)` or `Master Time (s)`.

### Navigating Replayed Data

Once data is loaded, most visual components will gain a time slider, allowing you to navigate through the recorded session.

<table>
  <thead>
    <tr>
      <th style="padding: 6px 15px; text-align: left;">Slider Type</th>
      <th style="padding: 6px 15px; text-align: left;">Component</th>
      <th style="padding: 6px 15px; text-align: left;">How to Use</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 6px 15px; vertical-align: top;">🎞️ <b>Time Window Slider</b></td>
      <td style="padding: 6px 15px; vertical-align: top;">Time-Series Plots</td>
      <td style="padding: 6px 15px;">This is a <b>range slider</b> with two handles. Drag the handles to select a specific time window. The plot will automatically zoom to display only the data within that range.</td>
    </tr>
    <tr>
      <td style="padding: 6px 15px; vertical-align: top;">🖱️ <b>Time Scrub Slider</b></td>
      <td style="padding: 6px 15px; vertical-align: top;">Heatmap, Nyquist Plot, 3D IMU View</td>
      <td style="padding: 6px 15px;">This is a <b>single-handle slider</b>. Drag the handle to a specific point in time. The component will update to show its state at that exact moment, allowing you to "scrub" through the data.</td>
    </tr>
  </tbody>
</table>

### Exporting from Replay Mode

*   **Time-Series Plots:** An <span style="color:#0277BD">`Export Visible Plot to PDF`</span> button will appear. Clicking it saves a high-quality PDF of **only the time window currently selected** by the Time Window Slider.
*   **Other Components (Heatmap, Nyquist, 3D IMU):** A `Take Snapshot` button allows you to save the current view (at the scrubbed time point) as a PNG, JPG, or PDF file.

---

## 📊 Visual Components Overview

The main tab area hosts various components to visualize data. Here's a quick look at the primary types.

#### Time-Series Plot

This is the most common component, used for plotting any data stream against time.

*   **Live Mode:** In **Flowing Mode** (see Bottom Control Bar), the plot scrolls automatically to show the most recent data. When not flowing, it displays all data received since the last `Clear GUI` action.
*   **Replay Mode:** A **Time Window Slider** (🎞️) appears, allowing you to zoom into specific segments of the recorded data for detailed inspection. An `Export Visible Plot to PDF` button also appears for saving the selected view.
*   **Features:**
    *   **Legend:** Clickable legend to show/hide individual data series.
    *   **Pan/Zoom:** Use the mouse to pan (left-click and drag) and zoom (scroll wheel). Right-click for more view options.

#### Other Components

*   **Pressure Heatmap:** Visualizes pressure distribution from an array of sensors, typically overlaid on an image (e.g., an insole). It includes controls for sensitivity and visualization style. In replay, it uses a **Time Scrub Slider** (🖱️).
*   **3D IMU Visualizer:** Shows the real-time orientation of an Inertial Measurement Unit (IMU) as a 3D model. It can be reset to a baseline orientation and uses a **Time Scrub Slider** (🖱️) in replay.
*   **Nyquist Plot:** Displays the impedance of a sensor by plotting its real vs. imaginary components. It features a data trail to show recent history and uses a **Time Scrub Slider** (🖱️) in replay.

---

## ⚙️ Bottom Control Bar

These controls fine-tune the display and provide access to utilities.

<table>
  <thead>
    <tr>
      <th style="padding: 6px 15px; text-align: left;">Control</th>
      <th style="padding: 6px 15px; text-align: left;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 6px 15px; vertical-align: top;"><code>Flowing Mode</code> Checkbox</td>
      <td style="padding: 6px 15px;"><span style="color:#388E3C"><b>(Live Mode Only)</b></span> When checked, time-series plots scroll automatically, showing a moving window of the most recent data.</td>
    </tr>
    <tr>
      <td style="padding: 6px 15px; vertical-align: top;"><code>Interval (s)</code> Textbox</td>
      <td style="padding: 6px 15px;"><span style="color:#388E3C"><b>(Live Mode Only)</b></span> Sets the duration (in seconds) of the time window displayed when `Flowing Mode` is active.</td>
    </tr>
    <tr>
      <td style="padding: 6px 15px; vertical-align: top;"><code>Apply Interval</code> Button</td>
      <td style="padding: 6px 15px;"><span style="color:#388E3C"><b>(Live Mode Only)</b></span> Applies the value from the `Interval (s)` textbox.</td>
    </tr>
    <tr>
      <td style="padding: 6px 15px; vertical-align: top;"><code>Log Raw Data to Console</code> Checkbox</td>
      <td style="padding: 6px 15px;">An advanced feature for troubleshooting. Toggles the printing of raw data values in the system console (the terminal where the app was launched).</td>
    </tr>
    <tr>
      <td style="padding: 6px 15px; vertical-align: top;"><code>Help?</code> Button</td>
      <td style="padding: 6px 15px;">Opens this guide window.</td>
    </tr>
  </tbody>
</table>