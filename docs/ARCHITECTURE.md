# Architecture

## Kept Runtime Surface

The core recording/alignment surface is intentionally small. Sanctioned
single-camera demo, remote FFS proxy, and visualization diagnostics are kept as
explicit boundaries around that core instead of becoming part of the formal
aligned-case data product.

## Entry Points

- `cameras_viewer.py`
- `cameras_viewer_FFS.py`
- `cameras_calibrate.py`
- `record_data.py`
- `record_data_realtime_align.py`
- `data_process/record_data_align.py`
- `scripts/harness/realtime_single_camera_pointcloud.py`
- `demo_v3/realtime_single_camera_realsense_masked_pcd.py`
- `demo_v3_1/realtime_single_camera_realsense_masked_pcd.py`
- `demo_v3_2/realtime_single_camera_ffs_masked_pcd.py`
- `demo_v3_3/realtime_single_camera_ffs_masked_pcd.py`

## Shared Packages

- `qqtt/env/camera/`: RealSense discovery, capture, defaults, preflight policy,
  recording metadata, and depth postprocess helpers.
- `data_process/depth_backends/`: FFS geometry, runners, benchmarking helpers,
  and optional postprocess filters.
- `data_process/visualization/`: aligned-case native-vs-FFS comparison and
  render/output helpers.
- `qqtt/demo/realtime_single_camera_pointcloud.py`: branch-default live
  single-camera point-cloud demo implementation.
- `qqtt/demo/single_demo_v3_runtime.py`: shared launcher for the
  `demo_v3*` entrypoints.
- `qqtt/demo/realtime_masked_edgetam_pcd.py`: shared masked PCD runtime. Its
  enhanced component filter defaults keep one object component and two
  controller components, matching single-object plus two-hand demo scenes.
- `services/ffs_remote/`: single-camera remote FFS depth request/response
  protocol and server/client utilities.

## Dependency Flow

`cameras_calibrate.py`, `record_data.py`, and `record_data_realtime_align.py`
import `CameraSystem`. `CameraSystem` owns RealSense startup, serial ordering,
stream configuration, and shared RGB color controls.

`data_process/record_data_align.py` remains the terminal product stage. It:

- stays cheap to import so `--help` works without CUDA or model imports
- keeps `realsense` as the default backend
- lazily imports `data_process/depth_backends/*` only when FFS output is requested
- writes `metadata.json` for compatibility and `metadata_ext.json` for QQTT-only extensions
- keeps canonical `depth/` as the compatibility depth output

Demo and proxy entrypoints depend on the core camera/runtime pieces in one
direction only. They may consume `CameraSystem`, calibration loaders, FFS
geometry/runners, and aligned-case visualization helpers. Core recording and
alignment entrypoints must not import demo, proxy, or experiment packages.

## Camera Identity Contract

Aligned cases use logical camera indices only as positions in
`metadata["serial_numbers"]`. `color/<camera_idx>/`, `depth/<camera_idx>/`,
intrinsics, depth scales, and other per-camera lists must all have the same
length and order as that serial list.

New calibrations write `calibrate_metadata.json` next to `calibrate.pkl`. The
sidecar records the calibration transform serial order, ChArUco board profile,
world-frame convention, RealSense color distortion coefficients, and
reprojection diagnostics. Recording entry points prefer that sidecar over
inferred connected-device order.

Swapping USB ports is safe because capture uses device serial numbers rather
than port order. Physically moving the camera relative to the scene requires a
new calibration.

## Visualization Boundary

Comparison visualization is an in-scope diagnostic utility built on aligned
cases. It may read native and FFS depth, calibration files, masks, and generated
diagnostic artifacts. It must not become a formal recording/alignment runtime
dependency.

Experiment-only workflow implementations live under
`data_process/visualization/experiments/` and CLIs under
`scripts/harness/experiments/`. `scripts/harness/check_experiment_boundaries.py`
guards that formal runtime code does not import those experiment packages.

## Removed Historical Surface

This branch no longer carries old three-camera demo folders, dual-GPU demo
runtimes, tracker backend registries, staged multi-view demo protocols, or
batch-size-specific demo helper scripts. Use `main` for the protected
multi-camera baseline.
