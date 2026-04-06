# proj-QQTT-v2

This repository handles 3-camera RealSense preview, calibration, synchronized raw capture, aligned case generation, and native-vs-FFS comparison visualization for aligned cases.

## Scope

This repo is intentionally narrow. It supports only:

1. RealSense camera preview / debugging
2. multi-camera calibration
3. synchronized recording with:
   - default RealSense RGB-D
   - optional raw D455 IR stereo capture
4. raw recording alignment and trimming
5. optional Fast-FoundationStereo depth generation during alignment
6. native-vs-FFS aligned depth comparison visualization

This repo does **not** include:

- segmentation
- dense tracking
- shape-prior generation
- downstream point-cloud processing beyond alignment packaging
- inverse physics
- Warp training / inference
- Gaussian Splatting
- rendering evaluation
- teleoperation or interaction demos

See [docs/SCOPE.md](/c:/Users/zhang/proj-QQTT/docs/SCOPE.md) for the exact boundary.

## Hardware Assumptions

- 3 Intel RealSense D400-series cameras
- a ChArUco calibration board
- Windows or Linux with librealsense-compatible device access
- optional footswitch or keyboard input for recording
- optional `ffmpeg` if you want aligned mp4 files

## Installation

Create and activate a Python 3.10 conda environment, then run:

```bash
bash ./env_install/env_install.sh
```

The install script is camera-only. It installs only the dependencies needed for:

- preview
- calibration
- recording
- alignment

## Preview

Live preview / debugging:

```bash
python cameras_viewer.py --help
python cameras_viewer.py
```

Default preview settings come from the shared camera defaults in [defaults.py](/c:/Users/zhang/proj-QQTT/qqtt/env/camera/defaults.py).

## Calibration

Calibrate the 3-camera setup:

```bash
python cameras_calibrate.py --help
python cameras_calibrate.py
```

Successful calibration writes `calibrate.pkl` in the repo root by default.

## Recording

Record a raw case.

Default path:

```bash
python record_data.py --help
python record_data.py --case_name my_case --capture_mode rgbd
```

If `--case_name` is omitted, a timestamp-based folder name is used.

Raw cases are written under `data_collect/<case_name>/`.

If `calibrate.pkl` exists, `record_data.py` copies it into the recorded case folder.

Optional FFS raw capture path:

```bash
python record_data.py --case_name my_case --capture_mode stereo_ir --emitter on
```

Optional experimental comparison path:

```bash
python record_data.py --case_name my_case --capture_mode both_eval --emitter on
```

`both_eval` is intentionally gated. On the current machine it is blocked by the latest D455 stream capability probe instead of silently dropping streams.

## Alignment

Align and trim a raw case:

```bash
python data_process/record_data_align.py --help
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --depth_backend realsense
```

Defaults:

- `--base_path ./data_collect`
- `--output_path ./data`
- `--depth_backend realsense`
- output fps comes from raw recording metadata unless `--fps` is provided
- mp4 generation is off unless `--write_mp4` is passed

Aligned cases are written to `data/<case_name>/`.

Optional FFS alignment backend:

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --depth_backend ffs --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth --write_ffs_float_m
```

Optional comparison backend:

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --depth_backend both --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth
```

Important:

- raw Fast-FoundationStereo output is not color-aligned by itself
- this repo explicitly reprojects FFS depth from IR-left coordinates into color coordinates during alignment
- canonical aligned `depth/` remains compatibility-oriented

## Compare Native vs FFS

Same-case comparison, when an aligned case contains both native depth and FFS depth:

```bash
python scripts/harness/visual_compare_depth_video.py --case_name my_case --aligned_root ./data --write_mp4
```

Fallback two-case comparison, when `both_eval` is not supported on the current machine:

```bash
python scripts/harness/visual_compare_depth_video.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --renderer fallback --write_mp4 --use_float_ffs_depth_when_available
```

The comparison utility:

- decodes compatible depth to meters
- deprojects with color intrinsics
- uses `calibrate.pkl` camera-to-world transforms
- fuses per-camera point clouds into a common frame
- renders native and FFS point clouds from the same deterministic viewpoint
- writes side-by-side frame sequences and optional videos

## Output Layout

### Raw case layout

```text
data_collect/<case_name>/
  calibrate.pkl
  metadata.json
  color/
    0/<step>.png
    1/<step>.png
    2/<step>.png
  depth/                # for rgbd or both_eval
    0/<step>.npy
    1/<step>.npy
    2/<step>.npy
  ir_left/              # for stereo_ir or both_eval
    0/<step>.png
    1/<step>.png
    2/<step>.png
  ir_right/             # for stereo_ir or both_eval
    0/<step>.png
    1/<step>.png
    2/<step>.png
```

### Aligned case layout

```text
data/<case_name>/
  calibrate.pkl
  metadata.json
  color/
    0/<frame>.png
    1/<frame>.png
    2/<frame>.png
    0.mp4          # only if --write_mp4
    1.mp4          # only if --write_mp4
    2.mp4          # only if --write_mp4
  depth/
    0/<frame>.npy
    1/<frame>.npy
    2/<frame>.npy
  ir_left/              # copied through when present
    0/<frame>.png
    1/<frame>.png
    2/<frame>.png
  ir_right/             # copied through when present
    0/<frame>.png
    1/<frame>.png
    2/<frame>.png
  depth_ffs/            # only for --depth_backend both
    0/<frame>.npy
    1/<frame>.npy
    2/<frame>.npy
  depth_ffs_float_m/    # optional
    0/<frame>.npy
    1/<frame>.npy
    2/<frame>.npy
  comparison/           # optional native-vs-FFS visualization output
    native_frames/
    ffs_frames/
    side_by_side_frames/
    videos/
    metrics.json
    comparison_metadata.json
```

## Validation

Deterministic checks:

```bash
python scripts/harness/check_all.py
```

Manual hardware validation checklist:

- [docs/HARDWARE_VALIDATION.md](/c:/Users/zhang/proj-QQTT/docs/HARDWARE_VALIDATION.md)
- [docs/generated/ffs_depth_backend_integration_validation.md](/c:/Users/zhang/proj-QQTT/docs/generated/ffs_depth_backend_integration_validation.md)
- [docs/generated/ffs_comparison_workflow_validation.md](/c:/Users/zhang/proj-QQTT/docs/generated/ffs_comparison_workflow_validation.md)

## Future Changes

This repo uses lightweight harness engineering:

- short map in [AGENTS.md](/c:/Users/zhang/proj-QQTT/AGENTS.md)
- versioned plans under [docs/exec-plans](/c:/Users/zhang/proj-QQTT/docs/exec-plans)
- deterministic scope guard in [check_scope.py](/c:/Users/zhang/proj-QQTT/scripts/harness/check_scope.py)

Any future change must preserve the camera-only charter.
