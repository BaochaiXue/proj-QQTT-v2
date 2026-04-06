# FFS Comparison Workflow Validation

## Deterministic Validation

Passed in `qqtt-ffs-compat`:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe scripts\harness\check_all.py
```

New software-only coverage includes:

- `tests.test_calibrate_loader_smoke`
- `tests.test_pointcloud_fusion_smoke`
- `tests.test_visual_compare_depth_video_smoke`

## Calibration Schema Supported

Supported `calibrate.pkl` schema:

- object type: `list` / `tuple` or `numpy.ndarray`
- shape `(N, 4, 4)`
- each transform is `camera -> world` (`c2w`)
- ordering matches the calibration-time camera order

Subset capture cases rely on:

- `metadata["calibration_reference_serials"]`

to map the case serials back to the full calibration order.

## Hardware Validation

### Current machine truth

- `both_eval` is still blocked by the latest D455 capability probe
- comparison validation therefore used the required two-case fallback workflow

### Native aligned case

Raw capture:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe record_data.py --capture_mode rgbd --case_name sanity_rgbd_single --max_frames 3 --disable-keyboard-listener --serials 239222300781
```

Alignment:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe data_process\record_data_align.py --case_name sanity_rgbd_single --start 151 --end 153 --depth_backend realsense
```

### FFS aligned case

Raw capture:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe record_data.py --capture_mode stereo_ir --case_name sanity_ffs_schema --max_frames 3 --disable-keyboard-listener --emitter on --serials 239222300781
```

Alignment:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe data_process\record_data_align.py --case_name sanity_ffs_schema --start 148 --end 150 --depth_backend ffs --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth --write_ffs_float_m
```

### Fallback comparison render

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe scripts\harness\visual_compare_depth_video.py --aligned_root C:\Users\zhang\proj-QQTT\data --realsense_case sanity_rgbd_single --ffs_case sanity_ffs_schema --output_dir C:\Users\zhang\proj-QQTT\data\comparison_sanity_single --renderer fallback --write_mp4 --use_float_ffs_depth_when_available
```

Observed result:

- passed
- wrote:
  - `native_frames/`
  - `ffs_frames/`
  - `side_by_side_frames/`
  - `videos/native.mp4`
  - `videos/ffs.mp4`
  - `videos/side_by_side.mp4`
  - `metrics.json`
  - `comparison_metadata.json`

Renderer actually used:

- `fallback`

Frame mapping used:

- `0 -> 0`
- `1 -> 1`
- `2 -> 2`

## Current Conclusion

- The repo now supports a real user-facing native-vs-FFS comparison workflow.
- Same-case comparison remains dependent on whether `both_eval` is truly supported by the current D455 profile.
- On this machine, the honest supported path is the two-case fallback comparison workflow.
