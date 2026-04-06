# Architecture

## Kept Runtime Surface

The repo is intentionally small.

### Entry Points

- `cameras_viewer.py`
- `cameras_calibrate.py`
- `record_data.py`
- `data_process/record_data_align.py`

### Shared Camera Package

- `qqtt/__init__.py`
- `qqtt/env/__init__.py`
- `qqtt/env/camera/defaults.py`
- `qqtt/env/camera/camera_system.py`
- `qqtt/env/camera/realsense/**`
- `qqtt/env/camera/recording_metadata.py`

### Optional FFS Depth Backend

- `data_process/depth_backends/__init__.py`
- `data_process/depth_backends/geometry.py`
- `data_process/depth_backends/fast_foundation_stereo.py`

### Comparison Visualization

- `data_process/visualization/__init__.py`
- `data_process/visualization/calibration_io.py`
- `data_process/visualization/pointcloud_compare.py`
- `scripts/harness/visual_compare_depth_video.py`

### Tooling / Harness

- `env_install/env_install.sh`
- `scripts/harness/check_scope.py`
- `scripts/harness/check_all.py`
- `tests/test_record_data_align_smoke.py`
- `docs/*`

## Dependency Flow

`cameras_calibrate.py` and `record_data.py` import `CameraSystem`.

`CameraSystem` depends on:

- `qqtt/env/camera/realsense/multi_realsense.py`
- `qqtt/env/camera/realsense/single_realsense.py`
- shared-memory helpers under `qqtt/env/camera/realsense/shared_memory/`

`data_process/record_data_align.py` remains the terminal product stage. It:

- stays stdlib-only at import time so `--help` remains cheap
- lazily imports `data_process/depth_backends/*` only when `--depth_backend ffs|both` is requested
- keeps `realsense` as the default backend

Harness scripts for FFS proof-of-life now reuse `data_process/depth_backends/*` instead of maintaining a second geometry implementation.

`calibrate.pkl` support is intentionally narrow and matches the current producer:

- object type: `list` / `tuple` or `numpy.ndarray`
- shape: `(N, 4, 4)`
- convention: each transform is `camera -> world` (`c2w`)
- ordering: calibration-time camera order

Subset capture cases rely on `metadata["calibration_reference_serials"]` to map case serials back to the full calibration order.

## Architectural Invariants

- No dependency from kept code into deleted downstream packages.
- No physics / rendering exports at the `qqtt` top level.
- Alignment remains the canonical data product of this repo; comparison visualization is an ancillary utility built on aligned cases.
- `depth/` remains the canonical compatibility output in aligned cases.
