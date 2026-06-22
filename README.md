# proj-QQTT-v2

This `single-camera` branch handles single-camera RealSense preview,
calibration, synchronized raw capture, aligned case generation, native-vs-FFS
comparison visualization for aligned cases, and sanctioned single-camera
realtime demo/proxy diagnostics. The `main` branch remains the protected
3-camera baseline.

## Scope

This repo supports:

1. RealSense camera preview / debugging
2. single-camera calibration by default
3. synchronized recording with default RealSense RGB-D and optional raw D455 IR stereo capture
4. raw recording alignment and trimming
5. optional Fast-FoundationStereo depth generation during alignment
6. native-vs-FFS aligned depth comparison visualization
7. live FFS preview and single-camera remote FFS proxy/replay diagnostics
8. realtime single-camera point-cloud demos using RealSense or FFS depth

This repo does not include three-camera demo entrypoints, dual-GPU demo
contracts, tracker backend harnesses, shape-prior generation, inverse physics,
Warp training/inference, Gaussian Splatting, teleoperation, robot control,
manipulation policy demos, vendored model repositories, checkpoints, or
generated artifact archives.

See [docs/SCOPE.md](docs/SCOPE.md) for the exact boundary.

## Hardware Assumptions

- 1 Intel RealSense D400-series camera by default on this branch
- a ChArUco calibration board
- Windows or Linux with librealsense-compatible device access
- optional footswitch or keyboard input for recording
- optional `ffmpeg` if you want aligned mp4 files

## Installation

Create and activate a Python 3.10 conda environment, then run:

```bash
bash ./env_install/env_install.sh
```

The install script is camera-only. It installs only the dependencies needed for
preview, calibration, recording, and alignment.

## Preview

Live RealSense preview:

```bash
python cameras_viewer.py --help
python cameras_viewer.py
```

Live RGB + Fast-FoundationStereo preview:

```bash
conda run -n FFS-SAM-RS python cameras_viewer_FFS.py
```

Single-D455 realtime point-cloud demo:

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/diagnostics/demo/realtime_single_camera_pointcloud.py \
  --profile 848x480 \
  --fps 60
```

Versioned single-camera demo entrypoints:

```bash
python demo_v3/realtime_single_camera_realsense_masked_pcd.py --dry-run
python demo_v3_1/realtime_single_camera_realsense_masked_pcd.py --dry-run
python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py --dry-run
python demo_v3_3/realtime_single_camera_ffs_masked_pcd.py --dry-run
```

Demo 3.1, Demo 3.2, and Demo 3.3 require repo-root `table_calibrate.pkl` by
default so their PCD and TAPNext++ marker outputs are in `table_world_z0`. Run table
calibration after moving the camera, table, or mount, or pass
`--table-calibrate <path>` to use an alternate table-world calibration.
The tabletop is `table_z_m = 0.0`; the current calibration convention treats
the workspace above the table as negative Z (`table_z_above_direction =
negative`). Headless captures include per-frame `world_z_stats.jsonl`; table-Z
filter diagnostics are always reported for table-calibrated PCD. Demo 3.1/3.2/3.3
`--demo-visual-mode pcd|tracking` windows, plus Demo 3.2/3.3 headless captures,
enable table-Z deletion by default at `--table-z-filter-threshold-m 0.0`; use
`--disable-table-z-filter` for an unfiltered ablation or pass a larger threshold
explicitly for wider table-band removal.

## Calibration

Calibrate the single-camera setup:

```bash
python cameras_calibrate.py --help
python cameras_calibrate.py
```

Successful calibration writes `calibrate.pkl` in the repo root by default.
Current calibration defaults are optimized for board detection:

```bash
python cameras_calibrate.py --width 1280 --height 720 --fps 5
```

The shared default camera count is `1` on this branch. Use `--num-cam` or
`--serials` only when deliberately running a multi-camera validation.

## Recording

Record a raw case:

```bash
python record_data.py --help
python record_data.py --case_name my_case --capture_mode rgbd
```

Raw cases are written under `data_collect/<case_name>/`. If `calibrate.pkl`
exists, `record_data.py` copies it into the recorded case folder.
For non-interactive captures with `--disable-keyboard-listener`, pass a positive
`--max_frames` so the recording starts and stops deterministically.

```bash
python record_data.py --case_name smoke_case --capture_mode rgbd --max_frames 5 --disable-keyboard-listener
```

Optional FFS raw capture path:

```bash
python record_data.py --case_name my_case --capture_mode stereo_ir --emitter on
```

## Alignment

Align and trim a raw case:

```bash
python data_process/record_data_align.py --help
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --depth_backend realsense
```

Aligned cases are written to `data/<case_name>/`.

Optional FFS alignment backend:

```bash
python data_process/record_data_align.py \
  --case_name my_case \
  --start 0 \
  --end 120 \
  --depth_backend ffs \
  --ffs_repo ../Fast-FoundationStereo \
  --ffs_model_path ../Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth \
  --write_ffs_float_m
```

Realtime native RGB-D formal export:

```bash
python record_data_realtime_align.py --case_name native_rt_baseline
```

This writes one growing case under `data/different_types_real_time/<case_name>/`
with only `color/`, `depth/`, `calibrate.pkl`, and legacy `metadata.json`.

## Compare Native vs FFS

Per-camera diagnostic panels:

```bash
python scripts/harness/diagnostics/depth/visual_compare_depth_panels.py \
  --aligned_root ./data \
  --realsense_case native_case \
  --ffs_case ffs_case \
  --write_mp4 \
  --use_float_ffs_depth_when_available
```

Cross-view reprojection comparison:

```bash
python scripts/harness/diagnostics/depth/visual_compare_reprojection.py \
  --aligned_root ./data \
  --realsense_case native_case \
  --ffs_case ffs_case \
  --camera_pair 0,1 \
  --write_mp4 \
  --use_float_ffs_depth_when_available
```

Single-frame object-centric compare:

```bash
python scripts/harness/diagnostics/visualization/visual_compare_turntable.py --case_name my_case --aligned_root ./data --frame_idx 0
```

Professor-facing summary pack:

```bash
python scripts/harness/diagnostics/visualization/visual_make_professor_triptych.py \
  --aligned_root ./data \
  --realsense_case native_case \
  --ffs_case ffs_case \
  --frame_idx 0
```

## Validation

Default deterministic checks:

```bash
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke
```

Broader validation:

```bash
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile deterministic
```
