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
python cameras_calibrate.py --width 848 --height 480 --fps 30 --num-cam 3
```

## 3. Record

```bash
python record_data.py --case_name my_case
```

If `--case_name` is omitted, a timestamp is used.

The recorder writes raw RGB-D data to `data_collect/<case_name>/`.

## 4. Align

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120
```

Optional mp4 generation:

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --write_mp4
```

Optional output location override:

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --output_path ./data
```
