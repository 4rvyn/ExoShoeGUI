# BLE Sensor Data Acquisition GUI

A modular and customizable PyQt6-based graphical interface for acquiring, visualizing, and logging data from Bluetooth Low Energy (BLE) sensors in real-time.

![Main Demo GIF](<INSERT_DEMO_GIF_PATH_HERE.gif>)

## Features

- **Real-time Visualization:** Multi-tab interface with various plotting components:
  - Time-series plots for sensor data streams.
  - Interactive pressure/heatmaps with Center of Pressure (CoP) tracking.
  - Live 3D IMU orientation visualizer (supports custom STL models).
  - Nyquist plots for impedance analysis.
- **Data Logging:** Capture live sensor data to timestamped CSV files, organized by session.
- **Data Replay:** Load and analyze previous CSV captures with time-scrubbing controls for detailed inspection.
- **Data Export:** Export plots from live or replayed sessions to high-quality PDF files.
- **Highly Customizable:** Configure the application by editing the Python script to:
  - Define new BLE device profiles.
  - Implement custom data parsers for any sensor.
  - Create derived data streams (sensor fusion).
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
    pip install qasync bleak pandas numpy matplotlib scienceplots PyQt6 pyqtgraph superqt numpy-stl
    ```
    *Note: `numpy-stl` is required for loading STL models in the 3D IMU visualizer.*

## Usage

Run the main script from your terminal:

```bash
python <your_script_name>.py
```

The application will launch, allowing you to scan for and connect to your configured BLE device.

## Customization

This application is designed to be easily adapted. All user-modifiable code is located in the **"customizable section"** at the top of the main script. Follow these steps to add a new sensor or change the layout:

1.  **Define a Data Handler:** Write a function to parse the raw `bytearray` from your sensor's BLE characteristic.
2.  **Define Derived Data (Optional):** Create functions to compute new data from existing raw or derived data streams.
3.  **Register Derived Data:** Register your new computation functions.
4.  **Update Device Configuration:** Add your new BLE characteristic UUIDs and handler functions to the `device_config` object.
5.  **Create/Modify GUI Components (Optional):** Define new `BaseGuiComponent` subclasses for custom visualizations.
6.  **Configure Tab Layout:** Add your components to the `tab_configs` list to place them in the GUI.

The script includes a built-in `Help` window that provides a more detailed, step-by-step guide for each of these customization areas.

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
