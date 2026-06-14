# Workflows

## 1. Preview

```bash
python cameras_viewer.py
```

Use this to verify that the active D455 enumerates and streams correctly before
calibration or recording. This `single-camera` branch defaults to one camera;
pass `--max-cams` or `--serials` only for explicit multi-camera validation.

Single-D455 realtime point-cloud demo:

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/realtime_single_camera_pointcloud.py \
  --profile 848x480 \
  --fps 60 \
  --view-mode camera
```

FFS depth mode for the same demo:

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/realtime_single_camera_pointcloud.py \
  --profile 848x480 \
  --fps 30 \
  --depth-source ffs \
  --view-mode camera \
  --debug
```

Versioned single-camera masked PCD demos:

```bash
python demo_v3/realtime_single_camera_realsense_masked_pcd.py --dry-run
python demo_v3_1/realtime_single_camera_realsense_masked_pcd.py --dry-run
python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py --dry-run
python demo_v3_3/realtime_single_camera_ffs_masked_pcd.py --dry-run
```

Replay the default fake-live camera case through Demo 3.1 as the camera stream:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_1/realtime_single_camera_realsense_masked_pcd.py \
  --input-source fake-live \
  --mode demo \
  --replay-fps 30
```

The default fake-live case is
`data_collect/sloth_both_eval_2min_e45_g35_20260614_155543`. The first
numerically sorted camera-0 step becomes demo `seq=0` for SAM3.1
initialization. Demo 3 / 3.1 consume replayed RGB-D; Demo 3.2 / 3.3 consume
replayed RGB plus IR stereo and compute FFS depth. Playback stops cleanly at
the end of the recording.

Live FFS preview:

```bash
conda run -n FFS-SAM-RS python cameras_viewer_FFS.py
```

Use `--render-mode none` when you want a throughput probe that skips panel
assembly and `cv2.imshow()`:

```bash
conda run -n FFS-SAM-RS python cameras_viewer_FFS.py --render-mode none
```

## 2. Calibrate

```bash
python cameras_calibrate.py
```

The default target is the current lab Calib.io ChArUco board:
`calibio-12x9-30mm`. This writes `calibrate.pkl` and
`calibrate_metadata.json` in the repo root by default.

Useful options:

```bash
python cameras_calibrate.py --width 1280 --height 720 --fps 5
python cameras_calibrate.py --serials 239222300781
python cameras_calibrate.py --exposure 70 --gain 60
```

Rerun calibration after any physical camera-position change.

## 3. Record

Default RealSense RGB-D path:

```bash
python record_data.py --case_name my_case --capture_mode rgbd
```

Optional FFS raw capture path:

```bash
python record_data.py --case_name my_case --capture_mode stereo_ir --emitter on
```

Short non-interactive smoke capture:

```bash
python record_data.py --case_name smoke_case --capture_mode rgbd --max_frames 5 --disable-keyboard-listener
```

Raw recordings are written under `data_collect/<case_name>/`.

## 4. Realtime Native Aligned Export

```bash
python record_data_realtime_align.py --case_name native_rt_baseline
```

The output case is written under `data/different_types_real_time/<case_name>/`
and intentionally keeps only the formal downstream interface:

- `calibrate.pkl`
- `metadata.json`
- `color/0/<frame>.png`
- `depth/0/<frame>.npy`

Runtime stats are written outside the formal case under
`data/different_types_real_time/_logs/`.

## 5. Align

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --depth_backend realsense
```

Optional mp4 generation:

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --write_mp4
```

Optional FFS backend:

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

`realsense` remains the default backend. `ffs` requires raw `ir_left` /
`ir_right` plus runtime geometry metadata. `both` remains an explicit
comparison path.

## 6. Compare Native vs FFS

Per-camera diagnostic panels:

```bash
python scripts/harness/visual_compare_depth_panels.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --write_mp4 --use_float_ffs_depth_when_available
```

Cross-view reprojection comparison:

```bash
python scripts/harness/visual_compare_reprojection.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --camera_pair 0,1 --write_mp4 --use_float_ffs_depth_when_available
```

Single-frame object-centric compare:

```bash
python scripts/harness/visual_compare_turntable.py --case_name my_case --aligned_root ./data --frame_idx 0
```

Professor-facing summary pack:

```bash
python scripts/harness/visual_make_professor_triptych.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --frame_idx 0
```

## 7. Validation

```bash
conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py
conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py --full
```
