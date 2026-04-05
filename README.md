# proj-QQTT-v2

This repository handles 3-camera RealSense preview, calibration, synchronized RGB-D recording, and aligned case generation up to `data_process/record_data_align.py`.

## Scope

This repo is intentionally narrow. It supports only:

1. RealSense camera preview / debugging
2. multi-camera calibration
3. synchronized RGB-D recording
4. raw recording alignment and trimming

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

Record a raw case:

```bash
python record_data.py --help
python record_data.py --case_name my_case
```

If `--case_name` is omitted, a timestamp-based folder name is used.

Raw cases are written under `data_collect/<case_name>/`.

If `calibrate.pkl` exists, `record_data.py` copies it into the recorded case folder.

## Alignment

Align and trim a raw case:

```bash
python data_process/record_data_align.py --help
python data_process/record_data_align.py --case_name my_case --start 0 --end 120
```

Defaults:

- `--base_path ./data_collect`
- `--output_path ./data`
- output fps comes from raw recording metadata unless `--fps` is provided
- mp4 generation is off unless `--write_mp4` is passed

Aligned cases are written to `data/<case_name>/`.

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
  depth/
    0/<step>.npy
    1/<step>.npy
    2/<step>.npy
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
```

## Validation

Deterministic checks:

```bash
python scripts/harness/check_all.py
```

Manual hardware validation checklist:

- [docs/HARDWARE_VALIDATION.md](/c:/Users/zhang/proj-QQTT/docs/HARDWARE_VALIDATION.md)

## Future Changes

This repo uses lightweight harness engineering:

- short map in [AGENTS.md](/c:/Users/zhang/proj-QQTT/AGENTS.md)
- versioned plans under [docs/exec-plans](/c:/Users/zhang/proj-QQTT/docs/exec-plans)
- deterministic scope guard in [check_scope.py](/c:/Users/zhang/proj-QQTT/scripts/harness/check_scope.py)

Any future change must preserve the camera-only charter.
