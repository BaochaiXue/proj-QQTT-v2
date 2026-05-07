# demo_2_max RTX 4090 Environment Validation

Date: 2026-05-06 EDT

## Host / GPU

- GPU: dual NVIDIA GeForce RTX 4090
- NVIDIA driver: 570.211.01
- `nvidia-smi` CUDA capability report: 12.8
- `CUDA_HOME=/usr/local/cuda`
- `/usr/local/cuda -> /etc/alternatives/cuda`
- No Linux NVIDIA driver / `cuda-drivers` install was attempted.

## Conda Environment

- Environment: `demo_2_max`
- Creation path: fresh `conda create -y -n demo_2_max python=3.12`
- Reason: `FFS-SAM-RS` was not present on this machine.
- Python: 3.12.13
- Activation hooks:
  - `CUDA_HOME=/usr/local/cuda`
  - `LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:$CUDA_HOME/lib64:...`
  - `PYTHONPATH=$HOME/EdgeTAM:...`
  - `TORCH_CUDA_ARCH_LIST=8.9`
  - `QQTT_SAM31_CHECKPOINT=$HOME/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt`

Final key packages:

```text
torch==2.11.0+cu128
torchvision==0.26.0+cu128
tensorrt-cu12==10.16.1.11
triton==3.6.0
open3d==0.19.0
pyrealsense2==2.57.7.10387
sam3==0.1.0
transformers==5.8.0
accelerate==1.13.0
huggingface_hub==1.14.0
opencv-python==4.13.0.92
atomics==1.0.3
imageio==2.37.3
lz4==4.4.5
zstandard==0.25.0
```

Notes:

- Initial `accelerate` install pulled `torch==2.11.0+cu130`, which imported but could not initialize CUDA with driver 570.211.01. Fixed by installing `torch==2.11.0+cu128` and `torchvision==0.26.0+cu128` from the PyTorch cu128 index.
- Initial `tensorrt==10.16.1.11` pulled CUDA 13 TensorRT packages. `import tensorrt` worked, but `trt.Builder(...)` failed with CUDA error 35. Fixed by replacing the CUDA 13 TensorRT packages with `tensorrt-cu12==10.16.1.11`.
- `python -m pip check` reports the known `sam3` metadata caveat: `sam3 0.1.0` requires `numpy<2,>=1.26`, while this environment keeps `numpy==2.4.4`. Runtime smoke tests passed with this stack.

## External Repos / Assets

- QQTT repo: `/home/xinjie/proj-QQTT-v2`
  - `git pull`: PASS, already up to date
- Fast-FoundationStereo repo: `/home/xinjie/Fast-FoundationStereo`
  - cloned from `https://github.com/NVlabs/Fast-FoundationStereo.git`
- FFS `20-30-48` weight:
  - path: `/home/xinjie/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth`
  - status: PASS, present
  - size: 62,078,956 bytes
  - obtained with `gdown --folder` from the official Fast-FoundationStereo Google Drive folder referenced by upstream `readme.md`
- EdgeTAM repo:
  - expected path: `/home/xinjie/EdgeTAM`
  - status: MISSING
  - no EdgeTAM repo URL was provided in the install instruction, so no clone/setup was attempted
- SAM3.1 checkpoint:
  - expected path: `/home/xinjie/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt`
  - status: MISSING
  - command attempted:

```bash
huggingface-cli download facebook/sam3.1 sam3.1_multiplex.pt \
  --local-dir "$HOME/.cache/huggingface/qqtt_sam31"
```

Result: FAIL, Hugging Face returned 401 gated repo access for `facebook/sam3.1`. The account must be authenticated and approved for the model.

## FFS TensorRT 4090 Engine

Target:

```text
data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864/
```

Status: PASS, built and validated locally on RTX 4090.

Contract:

- model: `20-30-48`
- valid iters: `4`
- engine size: `480x864`
- intended input policy: `848x480 -> pad to 864x480`
- TensorRT builder optimization level: `5`
- fp16: `True`
- workspace: `8 GiB`
- TensorRT: `10.16.1.11`
- torch: `2.11.0+cu128`
- torch CUDA: `12.8`

Generated artifacts:

```text
feature_runner.onnx
post_runner.onnx
feature_runner.engine
post_runner.engine
onnx.yaml
feature_engine_build.log
post_engine_build.log
demo_out/left.png
demo_out/right.png
demo_out/disp_vis.png
demo_out/depth_meter.npy
demo_out/cloud.ply
```

Validation output:

```text
TensorRT average after warmup: 9.6 ms
Verified TensorRT ONNX/engine/demo/profile outputs in /home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864
```

Build notes:

- PyTorch 2.11's default `torch.export` ONNX path failed on the FFS post runner with `AssertionError: sources must not be empty for symbol s24`.
- The successful build used the legacy ONNX exporter by passing `dynamo=False`, then built both TensorRT engines through the TensorRT Python API with `config.builder_optimization_level = 5`.

## Validation Commands

Environment import check: PASS

```bash
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0))
import pyrealsense2 as rs
import open3d as o3d
import tensorrt as trt
import zmq
builder = trt.Builder(trt.Logger())
print("imports ok", "trt", trt.__version__, "builder", bool(builder))
PY
```

Observed:

```text
torch 2.11.0+cu128 cuda 12.8
cuda_available True
gpu NVIDIA GeForce RTX 4090
imports ok trt 10.16.1.11 builder True
```

Compile check: PASS

```bash
python -m py_compile \
  demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  data_process/depth_backends/fast_foundation_stereo.py \
  services/ffs_remote/ffs_depth_server.py \
  services/ffs_remote/ffs_depth_client.py
```

Targeted smoke tests: PASS

```bash
python -m unittest -v \
  tests.test_sam31_mask_helper_smoke \
  tests.test_demo_v2_1_three_view_fused_pcd_smoke
```

Observed:

```text
Ran 45 tests in 2.095s
OK
```

Repo deterministic quick gate: PASS

```bash
python scripts/harness/check_all.py
```

Observed:

```text
Ran 129 tests in 1.845s
OK
[check] quick deterministic checks passed
```

EdgeTAM validation: SKIPPED

```bash
cd ~/EdgeTAM
conda run --no-capture-output -n demo_2_max python verify_edgetam_max.py
```

Reason: `/home/xinjie/EdgeTAM` is missing.
