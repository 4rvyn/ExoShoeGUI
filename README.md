# BLE Sensor Data Acquisition GUI (PyQt6)

This PyQt 6 application captures, visualizes, and logs Bluetooth Low Energy (BLE) sensor data in real time. I built it during my Bachelor’s thesis at RWTH Aachen University (October 2024 – May 2025) to drive a multisensory exoskeleton shoe that combines pressure, inertial, impedance, time‑of‑flight, and optical‑flow readings. Although the default configuration matches that prototype, you can redirect the software to any BLE device by adjusting a few clearly marked sections in the main script.

![Main Demo GIF](assets/demo.gif)

## Main features

While the program is running it streams each BLE characteristic through a user‑defined parser, plots the results live, and writes every sample to a timestamped CSV file. The interface includes time‑series charts, an interactive pressure heatmap with a fading centre‑of‑pressure trace, a 3‑D IMU orientation viewer that can load a custom STL model, and an impedance plot that leaves a trailing history. At the end of a session you can export any visible plot as a publication‑quality PDF.

A built‑in replay engine lets you scrub through recorded CSV files and feed them back into the same visual components, so live and offline analysis use identical code paths. You may add new components, derived data streams, or entire tabs without touching the core event loop.


## Screenshots

![Heatmap View](assets/heatmap_view.png)
![3D IMU View](assets/3d_imu_view.png)


## Quick start

To get started, clone the repository, install the required packages, and launch the GUI:

```bash
git clone https://github.com/4rvyn/ExoShoeGUI
pip install -r requirements.txt
python ExoShoeGUI.py
```

> Dependencies: qasync, bleak, pandas, numpy, matplotlib, scienceplots, PyQt6, pyqtgraph, superqt, numpy-stl, PyOpenGL.


## Try it without hardware

Open the app and click the **[Replay CSV…]**-button. The Sample logs in `log_samples/` let you explore the interface, replay controls, and export features without any BLE device connected.


## Adapt it to your BLE device

You’ll change three things:

1. **Data handlers** — parse your characteristic `bytearray` → return a dict of `{data_type: value}`.
2. **DeviceConfig** — set your device name, service UUID, and characteristic UUID ↔ handler mappings.
3. **Layout** — update `tab_configs` to display the data types your handlers produce (or add new components).

The built‑in **Help** window walks through this. A detailed, step‑by‑step guide and sample logs live in the [`log_samples/`](log_samples/) directory.


## License

This project is licensed under the MIT [LICENSE](LICENSE).
