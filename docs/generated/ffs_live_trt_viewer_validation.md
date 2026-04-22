# FFS Live TensorRT Viewer Validation

- Date: `2026-04-21`
- Machine: `XinjieZhang`
- GPU: `NVIDIA GeForce RTX 5090 Laptop GPU`
- Environment: `qqtt-ffs-compat`
- TensorRT Python package: `10.16.1.11`
- TensorRT SDK root: `C:\Users\zhang\external\TensorRT-10.16.1.11`
- Engine directory: `C:\Users\zhang\proj-QQTT\data\ffs_proof_of_life\trt_two_stage_640x480`
- Scope note: this validates the live `cameras_viewer_FFS.py` viewer path only. It does not change or validate QQTT aligned-case TRT integration.

## Commands

Install TensorRT Python runtime into `qqtt-ffs-compat`:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe -m pip install tensorrt-cu12==10.16.1.11
```

Single-camera live smoke:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe cameras_viewer_FFS.py --max-cams 1 --width 640 --height 480 --duration-s 30 --stats-log-interval-s 5 --ffs_backend tensorrt --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_trt_model_dir C:\Users\zhang\proj-QQTT\data\ffs_proof_of_life\trt_two_stage_640x480 --ffs_trt_root C:\Users\zhang\external\TensorRT-10.16.1.11
```

Three-camera live smoke:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe cameras_viewer_FFS.py --max-cams 3 --width 640 --height 480 --duration-s 20 --stats-log-interval-s 5 --ffs_backend tensorrt --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_trt_model_dir C:\Users\zhang\proj-QQTT\data\ffs_proof_of_life\trt_two_stage_640x480 --ffs_trt_root C:\Users\zhang\external\TensorRT-10.16.1.11
```

## Important Runtime Notes

- The current live TensorRT path reuses the upstream two-stage engines built during the Windows TRT proof-of-life.
- Engines remain fixed-shape. The viewer reads `onnx.yaml` and reports when capture size differs from the engine size.
- This validation used `640x480` capture to match the validated engine shape exactly.
- The worker process still uses the PyTorch GWC volume implementation on the host side because the upstream TensorRT runtime path assumes Triton is available.

## Single-Camera Result

Observed startup:

- detected camera: `239222300412`
- started at `640x480@30`
- startup note reported exact shape match: `TensorRT engine 640x480 matches capture size.`

Stabilized runtime stats:

- around `t=15s`: `capture=33.0`, `ffs=4.4`, `infer_ms=228.4`, `seq_gap=10`
- around `t=20s`: `capture=33.1`, `ffs=4.0`, `infer_ms=223.8`, `seq_gap=11`
- around `t=25.1s`: `capture=33.2`, `ffs=4.1`, `infer_ms=233.1`, `seq_gap=13`

Interpretation:

- the live TensorRT path is working end-to-end on one D455
- the first several seconds are dominated by worker startup / first-run warmup
- after warmup, one-camera throughput stabilized around `4 FPS`

## Three-Camera Result

Observed startup:

- detected cameras:
  - `239222300412`
  - `239222303506`
  - `239222300781`
- all 3 cameras started at `640x480@30`
- each camera printed the engine-match note for `640x480`

Stabilized runtime stats:

- around `t=10s`:
  - aggregate capture fps: `99.7`
  - aggregate FFS fps: `12.9`
  - `cam0`: `capture=33.6`, `ffs=4.3`, `infer_ms=230.7`, `seq_gap=15`
  - `cam1`: `capture=33.0`, `ffs=4.3`, `infer_ms=235.4`, `seq_gap=11`
  - `cam2`: `capture=33.1`, `ffs=4.4`, `infer_ms=234.0`, `seq_gap=9`
- around `t=20s`:
  - aggregate capture fps: `99.3`
  - aggregate FFS fps: `12.5`
  - `cam0`: `capture=33.0`, `ffs=4.2`, `infer_ms=240.5`, `seq_gap=10`
  - `cam1`: `capture=33.2`, `ffs=4.1`, `infer_ms=244.9`, `seq_gap=13`
  - `cam2`: `capture=33.2`, `ffs=4.2`, `infer_ms=250.0`, `seq_gap=12`

Interpretation:

- the optional TensorRT viewer path works with all 3 cameras active
- the viewer remained capture-stable while each worker processed the latest-only queue at about `4.1` to `4.3 FPS` per camera

## Follow-Up Boundary

- This validation does not cover `record_data_align.py` or other QQTT production depth-backend entrypoints.
- This validation also does not cover non-matching engine/capture combinations such as the default `848x480` viewer profile.
