# FFS Benchmark Tradeoff Validation

- Date: `2026-04-19`
- Machine: `XinjieZhang`
- GPU: `NVIDIA GeForce RTX 5090 Laptop GPU`
- Python / Torch: `3.12.13` / `2.7.0+cu128`
- Environment: `ffs-standalone`
- Important scope note: this is **not** a 4090 run. These numbers are for the local 5090 Laptop GPU only.
- Important interpretation note: this file is for **saved-pair offline screening**, not the authoritative simultaneous-3-camera online result. For the real live viewer benchmark, see `ffs_live_3cam_benchmark_validation.md`.

## Commands

Full-resolution baseline:

```text
C:\Users\zhang\miniconda3\envs\ffs-standalone\python.exe scripts/harness/benchmark_ffs_configs.py --aligned_root data --case_ref static/ffs_30_static_round3_20260414 --camera_idx 0 --frame_idx 0 1 2 --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth --model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\20-26-39\model_best_bp2_serialize.pth --model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\20-30-48\model_best_bp2_serialize.pth --scale 1.0 --valid_iters 8 4 --max_disp 192 --warmup_runs 2 --repeats 4 --target_fps 15 25 30 --out_dir data\ffs_benchmarks\2026-04-19_tradeoff_baseline
```

Multi-scale sweep:

```text
C:\Users\zhang\miniconda3\envs\ffs-standalone\python.exe scripts/harness/benchmark_ffs_configs.py --aligned_root data --case_ref static/ffs_30_static_round3_20260414 --camera_idx 0 --frame_idx 0 1 2 --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth --model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\20-26-39\model_best_bp2_serialize.pth --model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\20-30-48\model_best_bp2_serialize.pth --scale 1.0 0.75 0.5 --valid_iters 8 4 --max_disp 192 --warmup_runs 2 --repeats 4 --target_fps 15 25 30 --out_dir data\ffs_benchmarks\2026-04-19_tradeoff_multiscale
```

Extreme downscale sweep:

```text
C:\Users\zhang\miniconda3\envs\ffs-standalone\python.exe scripts/harness/benchmark_ffs_configs.py --aligned_root data --case_ref static/ffs_30_static_round3_20260414 --camera_idx 0 --frame_idx 0 1 2 --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth --model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\20-30-48\model_best_bp2_serialize.pth --scale 1.0 0.33 0.25 --valid_iters 8 4 --max_disp 192 --warmup_runs 2 --repeats 4 --target_fps 15 25 30 --out_dir data\ffs_benchmarks\2026-04-19_tradeoff_extreme
```

## Outputs

Local saved-pair benchmark result folders were moved under
`data\ffs_benchmarks\_archived_saved_pair_offline\` during the `2026-05-01`
results retention cleanup because they are offline screening results, not live
3-camera benchmark outputs.

- `data\ffs_benchmarks\_archived_saved_pair_offline\2026-04-19_tradeoff_baseline\summary.json`
- `data\ffs_benchmarks\_archived_saved_pair_offline\2026-04-19_tradeoff_baseline\report.md`
- `data\ffs_benchmarks\_archived_saved_pair_offline\2026-04-19_tradeoff_multiscale\summary.json`
- `data\ffs_benchmarks\_archived_saved_pair_offline\2026-04-19_tradeoff_multiscale\report.md`
- `data\ffs_benchmarks\_archived_saved_pair_offline\2026-04-19_tradeoff_extreme\summary.json`
- `data\ffs_benchmarks\_archived_saved_pair_offline\2026-04-19_tradeoff_extreme\report.md`

## Key Findings

Reference config was the slowest expected high-quality setting:

- `23-36-37_scale1_iters8_disp192`

Full-resolution `848x480` PyTorch path:

- best observed mean throughput was only `9.4 FPS`
- best full-resolution speed config was `20-30-48_scale1_iters4_disp192`
- that config still drifted from the reference by:
  - median absolute depth difference `2.49 mm`
  - p90 absolute depth difference `44.97 mm`

Multi-scale sweep:

- best mean-FPS tradeoff was `20-30-48_scale0.75_iters4_disp192`
- metrics:
  - mean latency `66.2 ms`
  - mean throughput `15.1 FPS`
  - median absolute depth difference to reference `4.10 mm`
  - p90 absolute depth difference to reference `81.48 mm`
- no tested config reached `25 FPS` or `30 FPS`

Extreme downscale sweep:

- fastest config was `20-30-48_scale0.33_iters4_disp192`
- metrics:
  - mean latency `60.9 ms`
  - mean throughput `16.4 FPS`
  - median absolute depth difference to reference `21.16 mm`
  - p90 absolute depth difference to reference `308.54 mm`
- pushing below `0.5x` bought only a small FPS gain while sharply degrading agreement with the reference output

## Practical Conclusion

On the `2026-04-19` local `RTX 5090 Laptop GPU` setup, the current QQTT PyTorch integration does **not** reach online-setting `25 FPS` or `30 FPS` on `848x480` RealSense-like inputs, even after aggressive checkpoint / `valid_iters` / `scale` tuning.

The best current PyTorch compromise from this run is:

- `20-30-48_scale0.75_iters4_disp192`

Reason:

- it is the only tested config that crossed `15 FPS` on the mean-latency metric
- it preserved much better reference agreement than the `0.33x` and `0.25x` extreme downscale runs

This means:

- if `15 FPS` is acceptable for the next integration step, start with `20-30-48`, `scale=0.75`, `valid_iters=4`
- if the requirement is truly `25+ FPS`, PyTorch alone is not enough on this repo path; the next step should be the separate ONNX / TensorRT flow from the upstream FFS repo

## Confidence Note

Checked on `2026-04-19`:

- upstream issue `NVlabs/Fast-FoundationStereo#23`
- upstream source `core/foundation_stereo.py`

Current understanding:

- FFS does not expose a supervised or calibrated per-pixel confidence output in the current repo path
- the maintainer suggested that confidence might be inferred from the classifier logits, but described that only as an untested idea
- the relevant source path builds `logits`, then applies `softmax` before disparity regression

## TensorRT Note

The upstream FFS repo keeps TensorRT in a separate ONNX / TRT workflow rather than in the default PyTorch demo path. That path was **not** validated in this repo change; only the current PyTorch integration was benchmarked here.

## Viewer Runtime Smoke

Checked on `2026-04-19` with the same benchmark-derived config:

- model: `20-30-48`
- `scale=1.0`
- `valid_iters=4`
- `max_disp=192`

Single-camera smoke command:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe cameras_viewer_FFS.py --max-cams 1 --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\20-30-48\model_best_bp2_serialize.pth --ffs_scale 1.0 --ffs_valid_iters 4 --ffs_max_disp 192
```

Observed startup output:

- detected camera: `239222300412`
- started successfully at `848x480@30`
- viewer stayed alive for a `20s` smoke window until manually terminated by the validation harness

All-detected-cameras smoke command:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe cameras_viewer_FFS.py --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\20-30-48\model_best_bp2_serialize.pth --ffs_scale 1.0 --ffs_valid_iters 4 --ffs_max_disp 192
```

Observed startup output:

- detected cameras:
  - `239222300412`
  - `239222303506`
  - `239222300781`
- all 3 cameras started successfully at `848x480@30`
- viewer stayed alive for a `20s` smoke window until manually terminated by the validation harness

Captured smoke logs were removed during the `2026-05-01` results retention cleanup. The retained live 3-camera benchmark logs are listed in `docs/generated/ffs_live_3cam_benchmark_validation.md`.
