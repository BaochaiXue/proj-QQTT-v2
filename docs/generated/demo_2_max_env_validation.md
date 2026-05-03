# demo_2_max Environment Validation

Date: 2026-05-03

## Summary

`demo_2_max` was created as a combined local demo environment for EdgeTAM,
Fast-FoundationStereo, RealSense, TensorRT/Open3D, and SAM 3.1.

The environment was cloned from `FFS-SAM-RS` instead of rebuilding CUDA or
PyTorch. `FFS-SAM-RS` already contained the local FFS, RealSense, TensorRT,
Open3D, and SAM 3.1 runtime stack; EdgeTAM is added through an activation hook
that prepends `/home/zhangxinjie/EdgeTAM` to `PYTHONPATH`.

## Exact Setup

```bash
conda create -y -n demo_2_max --clone FFS-SAM-RS
```

Activation hook paths:

```text
/home/zhangxinjie/miniconda3/envs/demo_2_max/etc/conda/activate.d/demo_2_max.sh
/home/zhangxinjie/miniconda3/envs/demo_2_max/etc/conda/deactivate.d/demo_2_max.sh
```

Hook policy:

```text
PYTHONPATH=/home/zhangxinjie/EdgeTAM
CUDA_HOME=/usr/local/cuda
TORCH_CUDA_ARCH_LIST=12.0
QQTT_SAM31_CHECKPOINT=/home/zhangxinjie/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt
```

## Runtime Versions

```text
python 3.12.13
numpy 2.4.4
torch 2.11.0+cu130
torch_cuda 13.0
torchvision 0.26.0+cu130
cv2 4.13.0
pyrealsense2 import-ok
sam3 0.1.0
tensorrt 10.16.1.11
triton 3.6.0
open3d 0.19.0
GPU NVIDIA GeForce RTX 5090 Laptop GPU
```

## Validation Commands

Import and environment hook check:

```bash
conda run --no-capture-output -n demo_2_max python - <<'PY'
import os
from pathlib import Path
import torch, torchvision, cv2, pyrealsense2 as rs, sam3, tensorrt, triton, open3d, sam2._C
from qqtt.env import CameraSystem
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(getattr(rs, "__version__", "import-ok"), getattr(sam3, "__version__", "unknown"))
print(tensorrt.__version__, triton.__version__, open3d.__version__)
print(CameraSystem.__name__)
print(os.environ["PYTHONPATH"])
print(os.environ["CUDA_HOME"])
print(os.environ["TORCH_CUDA_ARCH_LIST"])
print(Path(os.environ["QQTT_SAM31_CHECKPOINT"]).is_file())
PY
```

Result: passed. All listed imports succeeded, `sam2._C` loaded from the local
EdgeTAM repo, CUDA was available, and the SAM 3.1 checkpoint existed.

EdgeTAM verifier:

```bash
cd /home/zhangxinjie/EdgeTAM
conda run --no-capture-output -n demo_2_max python verify_edgetam_max.py
```

Result: passed. The verifier printed `edgetam-max verification PASS` after
building the image predictor and running mask-init and box-init synthetic video
propagation.

SAM 3.1 helper smoke:

```bash
conda run --no-capture-output -n demo_2_max python -m unittest -v tests.test_sam31_mask_helper_smoke
```

Result: passed, 11 tests.

SAM 3.1 predictor construction:

```bash
conda run --no-capture-output -n demo_2_max python - <<'PY'
from pathlib import Path
from scripts.harness.sam31_mask_helper import build_sam31_video_predictor
predictor, checkpoint = build_sam31_video_predictor(
    compile_model=False,
    async_loading_frames=False,
    max_num_objects=2,
)
print(type(predictor).__name__)
print(checkpoint)
print(Path(checkpoint).is_file())
PY
```

Result: passed. Predictor type was `Sam3MultiplexVideoPredictor`, using
`/home/zhangxinjie/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt`.

FFS / RealSense entrypoint help:

```bash
conda run --no-capture-output -n demo_2_max python cameras_viewer_FFS.py --help
conda run --no-capture-output -n demo_2_max python scripts/harness/verify_ffs_tensorrt_wsl.py --help
```

Result: passed. Both CLIs reached argument parsing successfully.

Repo deterministic harness:

```bash
conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py
```

Result: passed. Quick profile ran 88 tests and printed
`quick deterministic checks passed`.

Dependency metadata check:

```bash
conda run --no-capture-output -n demo_2_max python -m pip check
```

Result: failed with the inherited metadata conflict:

```text
sam3 0.1.0 has requirement numpy<2,>=1.26, but you have numpy 2.4.4.
```

This was not changed during `demo_2_max` creation because the source runtime
stack intentionally keeps `numpy==2.4.4`, and the actual import plus SAM 3.1
predictor construction validations succeeded.

## Status

`demo_2_max` is usable for integrated local demo work. Keep specialized
environments such as `edgetam-max`, `edgetam-hf-stream`, `SAM21-max`, and
`FFS-SAM-RS` for isolated benchmark claims.
