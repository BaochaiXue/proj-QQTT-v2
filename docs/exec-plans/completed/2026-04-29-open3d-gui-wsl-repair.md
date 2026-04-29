# 2026-04-29 Open3D GUI WSL Repair

## Goal

Make the Open3D GUI viewer path usable under WSL for `scripts/harness/realtime_single_camera_pointcloud.py`, instead of only bypassing Open3D with OpenCV/headless image viewers.

## Scope

- Diagnose the current WSLg/OpenGL/EGL/Open3D runtime behavior with a minimal Open3D GUI repro.
- Prefer environment/runtime fixes before changing visualization semantics.
- Keep OpenCV/headless fallback paths intact.
- If a WSL-specific shim is needed, make it explicit and documented.

## Validation

- Run a minimal Open3D GUI smoke command.
- Run the single-camera demo with `--viewer-backend open3d` when hardware/profile allows.
- Run focused smoke tests and deterministic guards if code changes.

## Result

- Confirmed `FFS-SAM-RS` has Open3D 0.19.0 and `open3d.visualization.gui.ImageWidget`.
- Confirmed the Open3D wheel links system `/lib/x86_64-linux-gnu/libGL.so.1` and `/lib/x86_64-linux-gnu/libGLX.so.0`; conda is not overriding GL/EGL here.
- Confirmed default WSLg GL was falling back to `llvmpipe`, while the repair environment reports `OpenGL renderer string: D3D12 (NVIDIA GeForce RTX 5090 Laptop GPU)` and OpenGL core 4.6.
- Added `scripts/harness/run_wslg_open3d.sh` to force the working WSLg Open3D GUI environment.
- Added the same WSLg Open3D defaults directly to `scripts/harness/realtime_single_camera_pointcloud.py`, so the normal realtime command works without the wrapper on this workstation.
- On this rig, Open3D window creation required `WAYLAND_DISPLAY=""` and `MESA_LOADER_DRIVER_OVERRIDE=d3d12`; fully unsetting `WAYLAND_DISPLAY` or only setting `GALLIUM_DRIVER=d3d12` was not sufficient.
- Added WSLg fast-exit handling for the realtime Open3D path so closing/duration stop first stops the camera pipeline, then exits before the Open3D/Filament teardown crash.

## Checks

- `scripts/harness/run_wslg_open3d.sh glxinfo -B`
- Minimal Open3D GUI `gui.Application.create_window(...)` probe under the wrapper
- `timeout 15s scripts/harness/run_wslg_open3d.sh /home/zhangxinjie/miniconda3/envs/FFS-SAM-RS/bin/python scripts/harness/realtime_single_camera_pointcloud.py --depth-source realsense --profile 848x480 --fps 30 --serial 239222300781 --view-mode orbit --duration-s 3`
- `timeout 15s /home/zhangxinjie/miniconda3/envs/FFS-SAM-RS/bin/python scripts/harness/realtime_single_camera_pointcloud.py --depth-source realsense --profile 848x480 --fps 30 --duration-s 3`
- `/home/zhangxinjie/miniconda3/envs/FFS-SAM-RS/bin/python -m unittest -v tests.test_realtime_single_camera_pointcloud_smoke`
- `git diff --check`
- `/home/zhangxinjie/miniconda3/envs/FFS-SAM-RS/bin/python scripts/harness/check_all.py`
