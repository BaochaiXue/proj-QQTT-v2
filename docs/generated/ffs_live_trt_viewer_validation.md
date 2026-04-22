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
conda run -n qqtt-ffs-compat python cameras_viewer_FFS.py --max-cams 1 --width 848 --height 480 --duration-s 10 --stats-log-interval-s 5 --ffs_backend tensorrt --ffs_repo /home/zhangxinjie/Fast-FoundationStereo --ffs_trt_model_dir /home/zhangxinjie/proj-QQTT-v2/data/ffs_proof_of_life/trt_two_stage_864x480_wsl
```

Three-camera live smoke:

```text
conda run -n qqtt-ffs-compat python cameras_viewer_FFS.py --max-cams 3 --width 848 --height 480 --duration-s 10 --stats-log-interval-s 5 --ffs_backend tensorrt --ffs_repo /home/zhangxinjie/Fast-FoundationStereo --ffs_trt_model_dir /home/zhangxinjie/proj-QQTT-v2/data/ffs_proof_of_life/trt_two_stage_864x480_wsl
```

## Important Runtime Notes

- The live TensorRT path reuses the WSL-built two-stage engines from `trt_two_stage_864x480_wsl`.
- `cameras_viewer_FFS.py` now defaults to `--ffs_backend tensorrt` and the repo-local `trt_two_stage_864x480_wsl` engine directory.
- The viewer reads `onnx.yaml` and identifies the `848x480 -> 864x480` path as symmetric replicate padding, not resize.
- The live TensorRT worker now runs TRT forward calls on a dedicated non-default CUDA stream, and the current validation run no longer prints the earlier `enqueueV3()` default-stream warning.
- The worker process still uses the PyTorch GWC volume implementation on the host side so the QQTT viewer path stays aligned with the current TensorRT backend implementation in the repo.
- The Qt runtime printed `Could not find the Qt platform plugin "wayland"` once at startup, but the viewer still launched correctly on the current display path.

## Single-Camera Result

Observed startup:

- detected camera: `239222303506`
- started at `848x480@30`
- startup note reported padding: `TensorRT engine 864x480; capture 848x480 will be symmetrically padded to 864x480 before inference.`

Stabilized runtime stats:

- around `t=10.0s`: `capture=36.5`, `ffs=3.7`, `infer_ms=249.0`, `seq_gap=9`

Interpretation:

- the `848x480` capture with `864x480` engine pad/unpad path is working end-to-end on one D455
- the first several seconds are still dominated by worker startup and first-run warmup
- after warmup, the current one-camera smoke reached about `3.7 FPS`
- the earlier TensorRT `enqueueV3()` default-stream warning no longer appeared during this run

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
  - aggregate capture fps: `107.8`
  - aggregate FFS fps: `8.2`
  - `cam0`: `infer_ms=365.2`
  - `cam1`: `infer_ms=329.0`
  - `cam2`: `infer_ms=353.3`

Interpretation:

- the optional TensorRT viewer path works with all 3 cameras active at the default `848x480` capture profile
- the viewer remained capture-stable while each worker processed the latest-only queue at about `2.7` to `2.8 FPS` per camera during the measured window
- the earlier TensorRT `enqueueV3()` default-stream warning no longer appeared during this run

## Follow-Up Boundary

- This validation does not cover `record_data_align.py` or other QQTT production depth-backend entrypoints.
- This validation does not cover the newer single-ONNX / single-engine upstream workflow.
