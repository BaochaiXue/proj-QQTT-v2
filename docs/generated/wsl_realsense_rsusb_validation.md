# WSL RealSense RSUSB Validation

Date: `2026-04-21`

## Goal

Restore live Intel RealSense D455 access inside the current WSL2 Ubuntu workspace for
the QQTT camera preview path after the stock Linux `pyrealsense2` wheel failed during
device power-up.

## Environment

- repo root: `/home/zhangxinjie/proj-QQTT-v2`
- conda env: `record_data_min`
- external librealsense source: `/home/zhangxinjie/external/librealsense`
- external librealsense build: `/home/zhangxinjie/external/librealsense/build-rsusb-py310`
- Python executable: `/home/zhangxinjie/miniconda3/envs/record_data_min/bin/python`

## Attached Cameras

Validated attached D455 ASIC serials inside WSL:

- `224223130498`
- `224223130693`
- `235523061048`

Validated viewer serial labels during successful preview:

- `239222303506`
- `239222300781`
- `239222300412`

## Initial Failure Mode

The stock wheel path was unstable under WSL:

- `rs.context().query_devices()` could sometimes see devices
- repo entrypoints such as `python cameras_viewer.py` failed with:
  - `RuntimeError: failed to set power state`
- a direct pipeline probe failed with:
  - `RuntimeError('No device connected')`

## External Build Commands

Clone upstream `librealsense` and configure an RSUSB-backed Python build:

```bash
git clone --depth 1 --branch v2.57.7 https://github.com/realsenseai/librealsense.git /home/zhangxinjie/external/librealsense
mkdir -p /home/zhangxinjie/external/librealsense/build-rsusb-py310
cd /home/zhangxinjie/external/librealsense/build-rsusb-py310
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DFORCE_RSUSB_BACKEND=ON \
  -DBUILD_PYTHON_BINDINGS:bool=true \
  -DPYTHON_EXECUTABLE=/home/zhangxinjie/miniconda3/envs/record_data_min/bin/python \
  -DBUILD_EXAMPLES=false \
  -DBUILD_GRAPHICAL_EXAMPLES=false
cmake --build /home/zhangxinjie/external/librealsense/build-rsusb-py310 --config Release -j 4
```

Swap the `record_data_min` environment to the RSUSB-built Python modules without using
`sudo make install`:

```bash
pkg=/home/zhangxinjie/miniconda3/envs/record_data_min/lib/python3.10/site-packages/pyrealsense2
build=/home/zhangxinjie/external/librealsense/build-rsusb-py310/Release
cp -a "$pkg/pyrealsense2.cpython-310-x86_64-linux-gnu.so" "$pkg/pyrealsense2.cpython-310-x86_64-linux-gnu.so.wheel-backup"
cp -a "$pkg/pyrsutils.cpython-310-x86_64-linux-gnu.so" "$pkg/pyrsutils.cpython-310-x86_64-linux-gnu.so.wheel-backup"
rm -f "$pkg/pyrealsense2.cpython-310-x86_64-linux-gnu.so" "$pkg/pyrsutils.cpython-310-x86_64-linux-gnu.so"
ln -sfn "$build/pyrealsense2.cpython-310-x86_64-linux-gnu.so" "$pkg/pyrealsense2.cpython-310-x86_64-linux-gnu.so"
ln -sfn "$build/pyrsutils.cpython-310-x86_64-linux-gnu.so" "$pkg/pyrsutils.cpython-310-x86_64-linux-gnu.so"
```

## USB Access Findings

The RSUSB build changed the failure mode from access denial to interface contention:

- before USB node permission fix:
  - `failed to open usb interface: 0, error: RS2_USB_STATUS_ACCESS`
- after granting write access to raw USB nodes:
  - `failed to claim usb interface: 0, error: RS2_USB_STATUS_BUSY`

The temporary permission fix that enabled raw USB access was:

```bash
sudo chmod 666 /dev/bus/usb/002/002 /dev/bus/usb/002/004 /dev/bus/usb/002/005
```

This is only a temporary workaround. It is not a persistent device-permission policy.

The repo now includes a persistent install path that replaces the manual `chmod` step:

```bash
sudo bash env_install/install_wsl_realsense_udev.sh
```

The installed rule is:

```text
ACTION=="add|change", SUBSYSTEM=="usb", DEVTYPE=="usb_device", ATTR{idVendor}=="8086", ATTR{idProduct}=="0b5c", GROUP:="plugdev", MODE:="0660"
```

This keeps the D455 raw USB nodes usable by the current `plugdev` user after future
WSL-side attach events. It does not replace the Windows-side `usbipd` bind / attach
requirement.

## Validation Commands

Stable enumeration through the RSUSB-backed Python binding:

```bash
/home/zhangxinjie/miniconda3/envs/record_data_min/bin/python - <<'PY'
import pyrealsense2 as rs
ctx = rs.context()
infos = ctx.query_devices()
print('query_count', len(infos))
for i in range(len(infos)):
    dev = infos[i]
    print(i, dev.get_info(rs.camera_info.serial_number), dev.get_info(rs.camera_info.name))
PY
```

Expected result after the RSUSB swap:

```text
query_count 3
0 239222303506 Intel RealSense D455
1 239222300781 Intel RealSense D455
2 239222300412 Intel RealSense D455
```

Successful live preview proof-of-life:

```bash
conda run -n record_data_min python cameras_viewer.py
```

Observed successful result:

- three D455 color + depth panels displayed in the QQTT viewer
- serial labels shown:
  - `239222303506`
  - `239222300781`
  - `239222300412`
- viewer overlay showed configured `848x480@30.0fps`
- measured live rate in the successful preview was approximately `14.7` to `14.8` FPS

## Notes

- raw USB node permissions are currently a manual step; this should be replaced by a
  persistent rule before relying on repeated re-attach cycles
- concurrent RealSense processes can still trigger `RS2_USB_STATUS_BUSY`
- the viewer proof-of-life succeeded; `record_data.py` was not re-run after the final
  successful viewer session because the preview process was actively using the cameras
