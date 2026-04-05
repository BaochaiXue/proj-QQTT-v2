# Hardware Validation

Hardware checks are manual. CI does not attempt to validate RealSense behavior.

## Active Hardware Inventory

Current connected cameras:

- `239222303506` - Intel RealSense D455
- `239222300433` - Intel RealSense D455
- `239222300781` - Intel RealSense D455

Design assumption: treat the active setup as 3 homogeneous D455 devices on one D400-family code path.

## Checklist

### Viewer

- 3 D400 cameras are connected
- `python cameras_viewer.py` launches successfully
- each camera shows live color and depth

### Calibration

- `python cameras_calibrate.py` detects the ChArUco board from all 3 cameras
- calibration completes without reprojection failure
- `calibrate.pkl` is written in the repo root

### Recording

- `python record_data.py --case_name smoke_case` creates `data_collect/smoke_case/`
- per-camera `color/<camera>/<step>.png` files are written
- per-camera `depth/<camera>/<step>.npy` files are written
- `metadata.json` exists
- `calibrate.pkl` is copied into the case if available

### Alignment

- `python data_process/record_data_align.py --case_name smoke_case --start <start> --end <end>` completes
- aligned case exists under `data/smoke_case/`
- aligned `metadata.json` exists
- aligned color/depth trees exist for all cameras
- optional `--write_mp4` produces per-camera mp4 files if ffmpeg is installed
