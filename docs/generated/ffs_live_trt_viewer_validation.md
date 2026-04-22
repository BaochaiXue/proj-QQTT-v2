# FFS Live TensorRT Viewer Validation

- Date: `2026-04-22`
- Machine: `XinjieZhang`
- GPU: `NVIDIA GeForce RTX 5090 Laptop GPU`
- Environment: `qqtt-ffs-compat`
- TensorRT Python runtime: `10.16.1.11`
- Engine directory: `/home/zhangxinjie/proj-QQTT-v2/data/ffs_proof_of_life/trt_two_stage_864x480_wsl`
- Scope note: this validates the live `cameras_viewer_FFS.py` viewer path only. It does not change or validate QQTT aligned-case TRT integration.

## Commands

TensorRT runtime install into `qqtt-ffs-compat`:

```text
/home/zhangxinjie/miniconda3/envs/qqtt-ffs-compat/bin/python -m pip install --extra-index-url https://pypi.nvidia.com tensorrt_cu12_bindings==10.16.1.11 tensorrt_cu12_libs==10.16.1.11
```

Default viewer command after the CLI default change:

```text
conda run -n qqtt-ffs-compat python cameras_viewer_FFS.py --ffs_repo /home/zhangxinjie/Fast-FoundationStereo
```

Single-camera live smoke:

```text
conda run -n qqtt-ffs-compat python cameras_viewer_FFS.py --max-cams 1 --width 848 --height 480 --duration-s 30 --stats-log-interval-s 5 --ffs_backend tensorrt --ffs_repo /home/zhangxinjie/Fast-FoundationStereo --ffs_trt_model_dir /home/zhangxinjie/proj-QQTT-v2/data/ffs_proof_of_life/trt_two_stage_864x480_wsl
```

Three-camera live smoke:

```text
conda run -n qqtt-ffs-compat python cameras_viewer_FFS.py --max-cams 3 --width 848 --height 480 --duration-s 20 --stats-log-interval-s 5 --ffs_backend tensorrt --ffs_repo /home/zhangxinjie/Fast-FoundationStereo --ffs_trt_model_dir /home/zhangxinjie/proj-QQTT-v2/data/ffs_proof_of_life/trt_two_stage_864x480_wsl
```

## Important Runtime Notes

- The live TensorRT path reuses the WSL-built two-stage engines from `trt_two_stage_864x480_wsl`.
- `cameras_viewer_FFS.py` now defaults to `--ffs_backend tensorrt` and the repo-local `trt_two_stage_864x480_wsl` engine directory.
- The viewer reads `onnx.yaml` and identifies the `848x480 -> 864x480` path as symmetric replicate padding, not resize.
- The worker process still uses the PyTorch GWC volume implementation on the host side so the QQTT viewer path stays aligned with the current TensorRT backend implementation in the repo.
- The Qt runtime printed `Could not find the Qt platform plugin "wayland"` once at startup, but the viewer still launched correctly on the current display path.

## Single-Camera Result

Observed startup:

- detected camera: `239222303506`
- started at `848x480@30`
- startup note reported padding: `TensorRT engine 864x480; capture 848x480 will be symmetrically padded to 864x480 before inference.`

Stabilized runtime stats:

- around `t=15.0s`: `capture=36.4`, `ffs=3.0`, `infer_ms=324.1`, `seq_gap=14`
- around `t=20.0s`: `capture=36.3`, `ffs=3.4`, `infer_ms=317.1`, `seq_gap=22`
- around `t=25.1s`: `capture=36.1`, `ffs=3.2`, `infer_ms=309.7`, `seq_gap=16`

Interpretation:

- the `848x480` capture with `864x480` engine pad/unpad path is working end-to-end on one D455
- the first several seconds are dominated by worker startup and first-run warmup
- after warmup, one-camera throughput stabilized around `3.2` to `3.4 FPS`

## Three-Camera Result

Observed startup:

- detected cameras:
  - `239222303506`
  - `239222300781`
  - `239222300412`
- all 3 cameras started at `848x480@30`
- each camera printed the `864x480` symmetric-padding startup note

Stabilized runtime stats:

- around `t=10.1s`:
  - aggregate capture fps: `108.3`
  - aggregate FFS fps: `7.5`
  - `cam0`: `capture=36.2`, `ffs=2.6`, `infer_ms=409.3`, `seq_gap=22`
  - `cam1`: `capture=36.4`, `ffs=2.4`, `infer_ms=415.0`, `seq_gap=28`
  - `cam2`: `capture=35.7`, `ffs=2.5`, `infer_ms=377.9`, `seq_gap=26`
- around `t=15.2s`:
  - aggregate capture fps: `109.4`
  - aggregate FFS fps: `7.2`
  - `cam0`: `capture=36.3`, `ffs=2.3`, `infer_ms=500.7`, `seq_gap=24`
  - `cam1`: `capture=35.6`, `ffs=2.1`, `infer_ms=436.5`, `seq_gap=22`
  - `cam2`: `capture=37.5`, `ffs=2.8`, `infer_ms=375.5`, `seq_gap=15`
- around `t=20.2s`:
  - aggregate capture fps: `103.4`
  - aggregate FFS fps: `5.7`
  - `cam0`: `capture=35.1`, `ffs=1.9`, `infer_ms=506.8`, `seq_gap=27`
  - `cam1`: `capture=34.1`, `ffs=1.9`, `infer_ms=520.7`, `seq_gap=26`
  - `cam2`: `capture=34.2`, `ffs=1.9`, `infer_ms=459.4`, `seq_gap=24`

Interpretation:

- the optional TensorRT viewer path works with all 3 cameras active at the default `848x480` capture profile
- the viewer remained capture-stable while each worker processed the latest-only queue at about `1.9` to `2.8 FPS` per camera during the measured window

## Follow-Up Boundary

- This validation does not cover `record_data_align.py` or other QQTT production depth-backend entrypoints.
- This validation does not cover the newer single-ONNX / single-engine upstream workflow.
