# Workflows

## 1. Preview

```bash
python cameras_viewer.py
```

Use this to verify that all 3 cameras enumerate and stream correctly before calibration or recording.

## 2. Calibrate

```bash
python cameras_calibrate.py
```

This writes `calibrate.pkl` in the repo root by default.

Useful options:

```bash
python cameras_calibrate.py --width 1280 --height 720 --fps 5 --num-cam 3
```

## 3. Record

```bash
python record_data.py --case_name my_case --capture_mode rgbd
```

If `--case_name` is omitted, a timestamp is used.

The recorder writes raw data to `data_collect/<case_name>/`.

Default RealSense path:

```bash
python record_data.py --case_name my_case --capture_mode rgbd
```

Optional FFS raw capture path:

```bash
python record_data.py --case_name my_case --capture_mode stereo_ir --emitter on
```

Optional non-interactive short capture:

```bash
python record_data.py --case_name smoke_case --capture_mode rgbd --max_frames 5 --disable-keyboard-listener
```

Optional single-camera selection for validation:

```bash
python record_data.py --case_name smoke_case --capture_mode stereo_ir --serials 239222300781 --max_frames 5 --disable-keyboard-listener
```

## 4. Align

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --depth_backend realsense
```

Optional mp4 generation:

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --write_mp4
```

Optional output location override:

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --output_path ./data
```

Optional FFS backend:

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --depth_backend ffs --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth --write_ffs_float_m
```

Experimental comparison backend:

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --depth_backend both --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth
```

Important:

- `realsense` remains the default backend.
- `ffs` requires raw `ir_left` / `ir_right` plus runtime geometry metadata.
- `both` is experimental and should only be used when the hardware probe says the same-take stream set is supported.

## 5. Compare

Same-case comparison when an aligned case contains both native and FFS depth:

```bash
python scripts/harness/visual_compare_depth_video.py --case_name my_case --aligned_root ./data --write_mp4
```

Fallback two-case comparison when `both_eval` is not supported:

```bash
python scripts/harness/visual_compare_depth_video.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --renderer fallback --write_mp4 --use_float_ffs_depth_when_available
```

The comparison workflow:

- decodes compatible depth to meters
- deprojects with `K_color`
- transforms to world using `calibrate.pkl`
- fuses the aligned camera clouds
- renders native and FFS clouds from the same deterministic fixed view
