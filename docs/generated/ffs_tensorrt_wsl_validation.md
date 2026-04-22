# FFS TensorRT WSL Validation

- Date: `2026-04-22`
- Machine: `XinjieZhang`
- GPU: `NVIDIA GeForce RTX 5090 Laptop GPU`
- Driver: `591.90`
- GPU memory: `24463 MiB`
- Environment: `ffs-standalone`
- Python / Torch / Triton / ONNX / TensorRT: `3.10` / `2.7.0+cu128` / `3.3.0` / `1.21.0` / `10.16.1.11`
- Scope note: this validates the external Fast-FoundationStereo two-stage ONNX / TensorRT route on WSL/Linux and produces the `864x480` engines reused by the QQTT live viewer.

## Commands

Python-side dependency install:

```text
/home/zhangxinjie/miniconda3/envs/ffs-standalone/bin/python -m pip install --extra-index-url https://pypi.nvidia.com onnx==1.21.0 tensorrt_cu12_bindings==10.16.1.11 tensorrt_cu12_libs==10.16.1.11
```

End-to-end WSL proof-of-life:

```text
conda run -n ffs-standalone python scripts/harness/verify_ffs_tensorrt_wsl.py
```

## Important WSL Notes

- The current WSL path uses the TensorRT Python builder rather than `trtexec`.
- The local machine did not need a separately unpacked Linux TensorRT SDK root for this validation; the NVIDIA PyPI runtime packages were sufficient.
- The repo-local harness keeps the host-side GWC volume path aligned with the QQTT TensorRT runner by replacing Triton-only calls with the existing PyTorch implementation.
- The current harness runs TensorRT forward calls on an explicit non-default CUDA stream so `enqueueV3()` no longer falls back to TensorRT's default-stream synchronization path.
- The upstream two-stage export still requires dimensions divisible by `32`, so the engine width for the default `848x480` viewer capture was set to `864`.

## Outputs

- `data/ffs_proof_of_life/trt_two_stage_864x480_wsl/feature_runner.onnx`
- `data/ffs_proof_of_life/trt_two_stage_864x480_wsl/post_runner.onnx`
- `data/ffs_proof_of_life/trt_two_stage_864x480_wsl/onnx.yaml`
- `data/ffs_proof_of_life/trt_two_stage_864x480_wsl/feature_runner.engine`
- `data/ffs_proof_of_life/trt_two_stage_864x480_wsl/post_runner.engine`
- `data/ffs_proof_of_life/trt_two_stage_864x480_wsl/feature_engine_build.log`
- `data/ffs_proof_of_life/trt_two_stage_864x480_wsl/post_engine_build.log`
- `data/ffs_proof_of_life/trt_two_stage_864x480_wsl/demo_out/disp_vis.png`
- `data/ffs_proof_of_life/trt_two_stage_864x480_wsl/demo_out/depth_meter.npy`
- `data/ffs_proof_of_life/trt_two_stage_864x480_wsl/demo_out/cloud.ply`

## Results

Dependency verification:

- `import tensorrt` succeeded
- `trt.Builder(trt.Logger())` succeeded
- `import onnx` succeeded

Engine build:

- `feature_runner.engine` built successfully
- `post_runner.engine` built successfully
- `feature_runner.engine` size: about `24 MB`
- `post_runner.engine` size: about `34 MB`

Headless TRT demo:

- completed successfully against the upstream demo pair
- wrote `disp_vis.png`, `depth_meter.npy`, and `cloud.ply`
- resized the upstream `960x540` demo pair to the fixed-engine `864x480` shape recorded in `onnx.yaml`

Latency comparison at `864x480`, `valid_iters=4`, `max_disp=192`:

- TensorRT average after warmup: `48.1 ms`
- PyTorch average after warmup: `81.1 ms`
- Observed speedup: about `1.69x`
- The previous TensorRT `enqueueV3()` default-stream warning did not appear in the current validation run.

## Follow-Up Boundary

- This validation does not cover QQTT `record_data_align.py` or other aligned-case TRT integration.
- This validation does not cover the newer single-ONNX / single-engine upstream workflow.
- Live RealSense viewer validation for the `848x480` capture with `864x480` engine pad/unpad path is recorded separately in `docs/generated/ffs_live_trt_viewer_validation.md`.
