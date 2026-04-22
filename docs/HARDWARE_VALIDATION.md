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
- each panel reports negotiated `configured fps` plus live `measured fps`
- `python cameras_viewer_FFS.py --ffs_repo <repo> --ffs_model_path <weights>` launches successfully
- each FFS panel shows live RGB on top and color-aligned FFS depth on bottom
- each FFS panel reports negotiated profile plus live `capture` and `ffs` fps
- optional TensorRT viewer mode launches successfully with prebuilt engines:
  - `python cameras_viewer_FFS.py --ffs_backend tensorrt --ffs_repo <repo> --ffs_trt_model_dir <engine_dir> --ffs_trt_root <tensorrt_root>`
  - if engine size differs from capture size, startup output reports whether frames will be symmetrically padded or resized before inference
- timed benchmark mode works:
  - `python cameras_viewer_FFS.py --duration-s 20 --stats-log-interval-s 5 --ffs_repo <repo> --ffs_model_path <weights>`
  - stdout prints aggregate and per-camera runtime stats during the run
  - TensorRT mode should also print aggregate and per-camera runtime stats when `--ffs_backend tensorrt` is selected

### Calibration

- `python cameras_calibrate.py` detects the ChArUco board from all 3 cameras
- calibration completes without reprojection failure
- `calibrate.pkl` is written in the repo root

### Recording

- `python record_data.py --case_name smoke_case --capture_mode rgbd` creates `data_collect/smoke_case/`
- per-camera `color/<camera>/<step>.png` files are written
- per-camera `depth/<camera>/<step>.npy` files are written for `rgbd`
- per-camera `ir_left/<camera>/<step>.png` and `ir_right/<camera>/<step>.png` are written for `stereo_ir`
- `metadata.json` exists
- `calibrate.pkl` is copied into the case if available
- short `--max_frames` runs fail quickly instead of hanging forever when one camera stalls and the others keep advancing

### Alignment

- `python data_process/record_data_align.py --case_name smoke_case --start <start> --end <end> --depth_backend realsense` completes
- aligned case exists under `data/smoke_case/`
- grouped aligned layouts such as `data/static/smoke_case/` are also valid when `--output_path` points at a grouped aligned root
- aligned `metadata.json` exists
- aligned `metadata_ext.json` exists for QQTT extension fields
- aligned `color/` exists
- aligned `depth/` exists for `realsense` and `ffs`
- aligned `depth_ffs/` exists only for `both`
- optional `--write_mp4` produces per-camera mp4 files if ffmpeg is installed

### Current D455 Notes

- latest stream probe result: `ir_pair` is stable on all 3 cameras
- latest stream probe result: `rgb_ir_pair` is not stable on all 3 cameras
- latest targeted `30s` revalidation still failed stability thresholds for 3-camera `rgbd_ir_pair` at `848x480@30`, emitter `on`
- short `30`-frame `both_eval` bursts can succeed, but long-duration stability is still not proven
- `record_data.py --capture_mode both_eval` is now warning-allowed experimentally rather than blocked by default
- same-take `rgbd_ir_pair` should still not be promised as a default supported workflow on this machine
- integrated `stereo_ir -> ffs` path has been validated on serial `239222300781`
- fallback two-case comparison video workflow has been validated on serial `239222300781`
