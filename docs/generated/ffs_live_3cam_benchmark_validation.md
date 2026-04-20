# FFS Live 3-Cam Benchmark Validation

- Date: `2026-04-19`
- Machine: `XinjieZhang`
- GPU: `NVIDIA GeForce RTX 5090 Laptop GPU`
- Environment: `qqtt-ffs-compat`
- Purpose: measure the real `cameras_viewer_FFS.py` online path with **3 cameras active at once**

## Why This Replaces The Earlier Online Conclusion

The earlier `ffs_benchmark_tradeoff_validation.md` numbers came from saved-pair offline screening and are still useful for checkpoint / parameter ranking.

They are **not** the authoritative online-setting numbers because they do not include:

- 3 simultaneous camera streams
- 3 FFS worker processes sharing one GPU
- latest-only queue overwrite behavior
- viewer-side cross-process transfer and color reprojection in the live path

This file records the realistic live 3-camera benchmark instead.

## Common Command Shape

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe cameras_viewer_FFS.py --max-cams 3 --duration-s 20 --stats-log-interval-s 5 --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\20-30-48\model_best_bp2_serialize.pth --ffs_valid_iters 4 --ffs_max_disp 192 --ffs_scale <scale>
```

All runs used:

- model family: `20-30-48`
- `valid_iters=4`
- `max_disp=192`
- target stream profile: `848x480@30`
- live hardware path: 3 connected D455 cameras

## Measured Configs

### 1. `scale=1.0`

Command:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe cameras_viewer_FFS.py --max-cams 3 --duration-s 20 --stats-log-interval-s 5 --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\20-30-48\model_best_bp2_serialize.pth --ffs_scale 1.0 --ffs_valid_iters 4 --ffs_max_disp 192
```

Stabilized runtime stats near the end of the run:

- aggregate capture fps: `99.4`
- aggregate FFS fps: `9.1`
- per camera:
  - `cam0`: `capture=33.1`, `ffs=2.7`, `infer_ms=357.3`, `seq_gap=14`
  - `cam1`: `capture=33.1`, `ffs=2.7`, `infer_ms=365.6`, `seq_gap=16`
  - `cam2`: `capture=33.2`, `ffs=3.7`, `infer_ms=313.1`, `seq_gap=19`

Interpretation:

- full-resolution online 3-camera FFS is far below `30 FPS per camera`
- the system is capture-stable but FFS-compute-limited

### 2. `scale=0.75`

Command:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe cameras_viewer_FFS.py --max-cams 3 --duration-s 20 --stats-log-interval-s 5 --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\20-30-48\model_best_bp2_serialize.pth --ffs_scale 0.75 --ffs_valid_iters 4 --ffs_max_disp 192
```

Stabilized runtime stats near the end of the run:

- aggregate capture fps: `99.4`
- aggregate FFS fps: `12.2`
- per camera:
  - `cam0`: `capture=33.2`, `ffs=4.1`, `infer_ms=248.4`, `seq_gap=14`
  - `cam1`: `capture=33.0`, `ffs=4.1`, `infer_ms=246.7`, `seq_gap=15`
  - `cam2`: `capture=33.2`, `ffs=4.1`, `infer_ms=243.7`, `seq_gap=16`

Interpretation:

- downscaling to `0.75` improves live 3-camera throughput materially
- even so, it is still well below true per-camera real time

### 3. `scale=0.5`

Command:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe cameras_viewer_FFS.py --max-cams 3 --duration-s 20 --stats-log-interval-s 5 --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\20-30-48\model_best_bp2_serialize.pth --ffs_scale 0.5 --ffs_valid_iters 4 --ffs_max_disp 192
```

Stabilized runtime stats near the end of the run:

- aggregate capture fps: `99.4`
- aggregate FFS fps: `22.6`
- per camera:
  - `cam0`: `capture=33.2`, `ffs=7.6`, `infer_ms=139.5`, `seq_gap=9`
  - `cam1`: `capture=33.1`, `ffs=7.5`, `infer_ms=139.6`, `seq_gap=8`
  - `cam2`: `capture=33.1`, `ffs=7.5`, `infer_ms=123.8`, `seq_gap=5`

Interpretation:

- this is the best of the tested live 3-camera configs
- it still does **not** reach `30 FPS per camera`
- it remains far below true online parity with the incoming `33 FPS` capture streams

## Practical Conclusion

For the real live 3-camera viewer path on this machine:

- `scale=1.0` is too slow for online use
- `scale=0.75` is still too slow for online use
- `scale=0.5` is the least bad among the tested `20-30-48 / iters=4 / disp=192` configs

But even the best tested live configuration only reached:

- about `22.6` aggregate FFS fps across 3 cameras
- about `7.5` FFS fps per camera

So the current PyTorch live 3-camera path still fails the real simultaneous-3-camera online requirement.

## Logs

- `data\ffs_benchmarks\live_3cam_scale1.0_stdout.log`
- `data\ffs_benchmarks\live_3cam_scale1.0_stderr.log`
- `data\ffs_benchmarks\live_3cam_scale0.75_stdout.log`
- `data\ffs_benchmarks\live_3cam_scale0.75_stderr.log`
- `data\ffs_benchmarks\live_3cam_scale0.5_stdout.log`
- `data\ffs_benchmarks\live_3cam_scale0.5_stderr.log`
