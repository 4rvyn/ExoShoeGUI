# BLE Sensor Data Acquisition GUI

A modular and customizable PyQt6-based graphical interface for acquiring, visualizing, and logging data from Bluetooth Low Energy (BLE) sensors in real-time.

![Main Demo GIF](<INSERT_DEMO_GIF_PATH_HERE.gif>)

### About This Project

This application was developed between November 2024 and May 2025 as a component of my Bachelor's Thesis, titled **"Development and Testing of a multisensory Shoe for an Exoskeleton"** at RWTH Aachen University.

The primary goal of this GUI was to provide a robust, real-time interface for acquiring, processing, and visualizing the fused data streams from the custom-built shoe, which includes pressure, inertia, ground impedance, time-of-flight, and optical flow sensors. As such, the default configuration of this script—including all BLE Service/Characteristic UUIDs, data parsing functions, and GUI layouts—is tailored specifically to the hardware and objectives of this thesis project.

However, I believe the underlying architecture and extensive customizability make this a valuable tool for a wide range of BLE sensor monitoring applications. While this script was developed for a specific purpose and may still have some issues, I am offering it to anyone who finds it useful and encourage you to build upon it. The seamless integration of live plotting and data replay, in particular, is an invaluable asset for any project involving BLE sensor data.

## Features

- **Real-time Visualization:** Multi-tab interface with various plotting components:
  - Time-series plots for sensor data streams.
  - Interactive pressure/heatmaps with fading Center of Pressure (CoP) tracking.
  - Live 3D IMU orientation visualizer (supports custom STL models).
  - Live Impedance plot for impedance analysis with fading trail.
- **Data Logging:** Capture live sensor data to timestamped CSV files, organized by session.
- **Data Replay:** Load and analyze previous CSV captures with time-scrubbing controls for detailed inspection.
- **Data Export:** Export plots from live or replayed sessions to high-quality PDF files.
- **Highly Customizable:** Configure the application by editing the Python script to:
  - Define new BLE device profiles.
  - Implement custom data parsers for any sensor.
  - Create derived data streams (sensor fusion).
  - Design new custom Components
  - Design custom GUI layouts.

## Screenshots

<p align="center">
  <img src="<INSERT_HEATMAP_IMAGE_PATH_HERE.png>" alt="Heatmap View" width="45%">
  &nbsp; &nbsp;
  <img src="<INSERT_3D_IMU_IMAGE_PATH_HERE.png>" alt="3D IMU View" width="45%">
</p>

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Install dependencies:**
    ```bash
    pip install qasync bleak pandas numpy matplotlib scienceplots PyQt6 pyqtgraph superqt numpy-stl PyOpenGL
    ```
    *Note: For full compatibility, use the provided `requirements.txt`.*

## Usage

Run the main script from your terminal:

```bash
python ExoShoeGUI.py
```

### Demonstration & Replay

To allow users to explore the GUI's features without needing my physical thesis hardware, a set of random sample log files are included in the `sample_logs/` directory.

Use the **`Replay CSV...`** button in the GUI to load one or more of these CSV files. The application's visualization components will populate with the sample data, enabling you to test the replay and export functionalities.

## Adapting for Your Own BLE Device

**Important Note:** The default configuration of this script is tailored *specifically* to the multisensory exoskeleton shoe developed for the author's thesis. To use this application with your own BLE device, you **must** modify the "customizable section" in the main Python script.

Follow these essential steps:

1.  **Define Your Data Handlers:**
    - In the script, locate the section `# 1. --- Data Handlers ... ---`.
    - Write new Python functions to parse the `bytearray` data from your sensor's specific BLE characteristics. Each handler must return a dictionary of data types and their values (e.g., `{'temperature': 25.5, 'humidity': 45.1}`).

2.  **Modify the `DeviceConfig` Object:**
    - Locate the `device_config = DeviceConfig(...)` definition.
    - Change the `name` and `service_uuid` to match your BLE device.
    - **Crucially, update the `characteristics` list.** Remove the existing `CharacteristicConfig` objects and add new ones for your device. Each new entry must link your characteristic's `uuid` to the corresponding data `handler` function you wrote in Step 1.

3.  **Reconfigure the GUI Layout:**
    - Locate the `tab_configs` list at the end of the customizable section.
    - Modify the layouts to display your new data. You will need to change the `data_type` strings within the component `config` dictionaries to match the keys your new data handlers produce.
    - Remove or replace components that are specific to the thesis project (e.g., the insole pressure map, impedance plots) with components relevant to your data.
4.  **Further Possibilities - Creating New GUI Components**
    - The modular architecture allows you to create entirely new visualization components beyond the ones provided. This is powerful for unique sensors or custom data representations. To do so, you create a Python class that inherits from `BaseGuiComponent`.


The built-in **`Help`** window provides a more detailed, step-by-step guide for each of these customization tasks. After making these changes, the application will be tailored to your custom hardware.

## License

This project is licensed under the MIT License.

```
Copyright (c) 2025 Arvin Parvizinia

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
