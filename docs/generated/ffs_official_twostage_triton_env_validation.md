# FFS Official Two-Stage Triton TensorRT Environment Validation

- Date: `2026-04-28`
- Machine: `XinjieZhang`
- Purpose: create a local environment that can run the official Fast-FoundationStereo two-stage TensorRT path:
  - `feature_runner.engine`
  - official Triton GWC volume kernel
  - `post_runner.engine`

## Environment

Created by cloning the existing FFS standalone environment so existing CUDA/TensorRT/OpenCV/Open3D/RealSense dependencies remain available:

```bash
conda create -y -n ffs-official-trt --clone ffs-standalone
```

The cloned environment initially inherited `triton==3.3.0`, which failed the official GWC kernel with:

```text
SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats
```

Tried `triton==3.3.1`; it failed with the same error.

Installed the first working version:

```bash
/home/zhangxinjie/miniconda3/envs/ffs-official-trt/bin/python -m pip install --no-deps --force-reinstall triton==3.4.0
```

Validated package stack:

```text
python 3.10.18
torch 2.7.0+cu128
triton 3.4.0
tensorrt 10.16.1.11
opencv 4.10.0
timm 1.0.26
omegaconf 2.3.0
pyrealsense2 import OK
```

## Official Triton GWC Probe

Command:

```bash
/home/zhangxinjie/miniconda3/envs/ffs-official-trt/bin/python - <<'PY'
import sys
sys.path.insert(0, '/home/zhangxinjie/Fast-FoundationStereo')
import torch
import triton
from core.submodule import build_gwc_volume_triton, triton as submodule_triton
print('python', sys.version.split()[0])
print('torch', torch.__version__)
print('triton', triton.__version__)
print('cuda_available', torch.cuda.is_available())
print('submodule_triton_available', submodule_triton is not None)
ref=torch.randn(1,128,32,64,device='cuda',dtype=torch.float16)
tar=torch.randn(1,128,32,64,device='cuda',dtype=torch.float16)
torch.cuda.synchronize()
out=build_gwc_volume_triton(ref, tar, 48, 8)
torch.cuda.synchronize()
print('official_triton_gwc_ok', tuple(out.shape), out.dtype, out.device)
PY
```

Outcome:

```text
python 3.10.18
torch 2.7.0+cu128
triton 3.4.0
cuda_available True
submodule_triton_available True
official_triton_gwc_ok (1, 8, 48, 32, 64) torch.float16 cuda:0
```

## Static Sample Two-Stage TRT Validation

Command used real static round data and the existing `20-30-48 / scale=1.0 / valid_iters=4 / max_disp=192` two-stage FP16 engines:

```bash
/home/zhangxinjie/miniconda3/envs/ffs-official-trt/bin/python - <<'PY'
from pathlib import Path
import json
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path('/home/zhangxinjie/proj-QQTT-v2')
FFS_REPO = Path('/home/zhangxinjie/Fast-FoundationStereo')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FFS_REPO))

from data_process.depth_backends.fast_foundation_stereo import FastFoundationStereoTensorRTRunner

case_dir = ROOT / 'data/static/ffs_30_static_round1_20260410_235202'
left = cv2.imread(str(case_dir / 'ir_left/0/0.png'), cv2.IMREAD_GRAYSCALE)
right = cv2.imread(str(case_dir / 'ir_right/0/0.png'), cv2.IMREAD_GRAYSCALE)
meta = json.loads((case_dir / 'metadata.json').read_text())
K = np.asarray(meta['intrinsics'][0], dtype=np.float32)
artifact_dir = ROOT / 'data/experiments/_archived_obsolete/ffs_static_replay_matrix_20260422_sequential_obsolete_fullrun/artifacts/two_stage_fp16/model_20-30-48/scale_1p0_iters_4'
runner = FastFoundationStereoTensorRTRunner(ffs_repo=FFS_REPO, model_dir=artifact_dir)

for _ in range(5):
    out = runner.run_pair(left, right, K_ir_left=K, baseline_m=0.095)
    torch.cuda.synchronize()

times = []
for _ in range(20):
    start = time.perf_counter()
    out = runner.run_pair(left, right, K_ir_left=K, baseline_m=0.095)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - start) * 1000.0)
print('mean_ms', float(np.mean(times)))
print('min_ms', float(np.min(times)))
print('max_ms', float(np.max(times)))
print('disp_shape', out['disparity'].shape)
PY
```

Outcome:

```text
runner_engine 480 864 iters 4 max_disp 192
torch 2.7.0+cu128
triton 3.4.0
official_two_stage_static_sample_ok
mean_ms 16.694541947799735
min_ms 16.022908996092156
max_ms 17.982404999202117
disp_shape (480, 848)
depth_shape (480, 848)
```

This timing is the QQTT runner path on one static IR pair after warmup, with official Triton GWC enabled. It includes QQTT preprocessing, TensorRT execution, official Triton GWC, CPU disparity download, and depth finalization.

## Viewer Import Check

```bash
/home/zhangxinjie/miniconda3/envs/ffs-official-trt/bin/python cameras_viewer_FFS.py --help
```

Outcome: help printed successfully.

## Usage

Use this environment when running the official two-stage TensorRT path:

```bash
conda activate ffs-official-trt
python cameras_viewer_FFS.py \
  --ffs_backend tensorrt \
  --ffs_trt_mode two_stage \
  --ffs_repo /home/zhangxinjie/Fast-FoundationStereo \
  --ffs_trt_model_dir /home/zhangxinjie/proj-QQTT-v2/data/experiments/_archived_obsolete/ffs_static_replay_matrix_20260422_sequential_obsolete_fullrun/artifacts/two_stage_fp16/model_20-30-48/scale_1p0_iters_4
```

Keep `qqtt-ffs-compat` for the broader QQTT harness unless a command specifically needs official two-stage TensorRT with Triton.
