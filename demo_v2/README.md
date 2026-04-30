# Demo V2: Single-D455 Realtime Viewer

This folder is the active current-week workspace for the standalone
single-camera demo. It streams one Intel RealSense D455 and renders either:

- native RealSense color-aligned RGB-D
- live FFS depth from the D455 IR stereo pair

The demo uses the camera color frame directly: meters, `x` right, `y` down,
`z` forward. It does not read `calibrate.pkl` and does not use any multi-camera
world transform.

## Files

- `realtime_single_camera_pointcloud.py`: single-camera demo entrypoint
- `run_wslg_open3d.sh`: optional WSLg/Open3D GUI environment wrapper

## Native RealSense Depth

```bash
conda run -n FFS-SAM-RS python demo_v2/realtime_single_camera_pointcloud.py \
  --profile 848x480 \
  --fps 60 \
  --depth-source realsense
```

The default camera view uses the fast image backend, preserves valid aligned
depth pixels, and does not apply far clipping (`--depth-max-m 0.0`).

## FFS Depth

```bash
conda run -n FFS-SAM-RS python demo_v2/realtime_single_camera_pointcloud.py \
  --profile 848x480 \
  --fps 60 \
  --depth-source ffs \
  --view-mode camera \
  --debug
```

FFS mode expects the repo-local two-stage TensorRT engine directory unless
`--ffs-trt-model-dir` is provided. It also expects the external
Fast-FoundationStereo repo path from `--ffs-repo`.

## Orbit Point-Cloud View

```bash
conda run -n FFS-SAM-RS python demo_v2/realtime_single_camera_pointcloud.py \
  --depth-source realsense \
  --profile 848x480 \
  --fps 30 \
  --view-mode orbit
```

The script applies the WSLg/Open3D defaults before importing Open3D. The wrapper
is available when you want to force the same environment around another command:

```bash
./demo_v2/run_wslg_open3d.sh conda run -n FFS-SAM-RS python demo_v2/realtime_single_camera_pointcloud.py --view-mode orbit
```

## Moving This Folder

Native RealSense mode can run from this folder as long as the Python environment
has `pyrealsense2`, `open3d`, `numpy`, and optionally `opencv-python` / `numba`.
For FFS mode outside the repo, set:

```bash
export QQTT_REPO_ROOT=/path/to/proj-QQTT-v2
```

or pass explicit `--ffs-repo` and `--ffs-trt-model-dir` paths.
