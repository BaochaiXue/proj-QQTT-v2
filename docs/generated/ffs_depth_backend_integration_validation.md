# FFS Depth Backend Integration Validation

## Deterministic Validation

Passed in `qqtt-ffs-compat`:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe scripts\harness\check_all.py
```

This covered:

- CLI help for preview / calibrate / record / align
- scope guard
- legacy alignment smoke test
- FFS geometry tests
- D455 probe schema / matrix tests
- new FFS / both alignment smoke tests
- new recording metadata schema test

## Real Hardware Validation

### A. Default path: `rgbd -> realsense`

Command:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe record_data.py --capture_mode rgbd --case_name sanity_rgbd --max_frames 5 --disable-keyboard-listener
```

Result:

- passed on the current 3-camera D455 setup
- raw case written under `data_collect/sanity_rgbd/`
- `metadata.json` includes v2 recording schema

Alignment command:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe data_process\record_data_align.py --case_name sanity_rgbd --start 152 --end 157 --depth_backend realsense
```

Result:

- passed
- aligned case written under `data/sanity_rgbd/`

### B. FFS path: `stereo_ir -> ffs`

Three-camera attempt:

- attempted with `stereo_ir` on all 3 cameras
- did not complete cleanly
- consistent with the prior D455 stream probe result that `rgb_ir_pair` is not stable on this machine for 3 cameras

Single-camera integrated validation:

Command:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe record_data.py --capture_mode stereo_ir --case_name sanity_ffs_schema --max_frames 3 --disable-keyboard-listener --emitter on --serials 239222300781
```

Result:

- passed on D455 serial `239222300781`
- raw case written under `data_collect/sanity_ffs_schema/`
- raw case includes `color/`, `ir_left/`, `ir_right/`
- runtime geometry metadata includes `K_color`, `K_ir_left`, `K_ir_right`, `T_ir_left_to_right`, `T_ir_left_to_color`, `ir_baseline_m`, and `depth_scale_m_per_unit`
- raw metadata now also records `depth_encoding = uint16_meters_scaled_invalid_zero`

Alignment command:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe data_process\record_data_align.py --case_name sanity_ffs_schema --start 148 --end 150 --depth_backend ffs --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth --write_ffs_float_m
```

Result:

- passed
- aligned case written under `data/sanity_ffs_schema/`
- canonical `depth/` is FFS-generated color-aligned compatible depth
- optional `depth_ffs_float_m/` was written

### C. Comparison path: `both_eval -> both`

Probe status:

- latest D455 stream capability probe concluded `Case E`
- same-take `depth_ir_pair` / `rgbd_ir_pair` are not stable enough to promise on this machine

Validation command:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe record_data.py --capture_mode both_eval --case_name sanity_both --max_frames 5 --disable-keyboard-listener --serials 239222300781 --emitter on
```

Observed result:

- correctly blocked before recording
- error text:
  - `RuntimeError: both_eval is blocked by the latest D455 stream probe on this machine for serials=['239222300781'], 848x480@30, emitter=on.`

## Current Conclusion

- Existing RealSense RGB-D workflow still works.
- Integrated FFS alignment backend works on a real recorded case when raw `stereo_ir` data is available.
- On this machine, `both_eval` should remain blocked.
- On this machine, 3-camera `stereo_ir` should still be treated as unstable until a new probe proves otherwise.
