# FFS Batch-3 Viewer Validation

- Date: `2026-04-22`
- Machine: `XinjieZhang`
- GPU: `NVIDIA GeForce RTX 5090 Laptop GPU`
- Environments:
  - `qqtt-ffs-compat`
  - `ffs-standalone`
- Scope note: this note records repo-integrated strict 3-camera batch viewer proof-of-life and batch-3 TRT artifact-generation attempts. It does not claim live 3-camera hardware validation.

## Commands

PyTorch batch runner smoke against the upstream demo pair:

```text
/home/zhangxinjie/miniconda3/envs/qqtt-ffs-compat/bin/python - <<'PY'
from pathlib import Path
import sys
import imageio.v2 as imageio
import numpy as np
ROOT = Path('/home/zhangxinjie/proj-QQTT-v2')
FFS_REPO = Path('/home/zhangxinjie/Fast-FoundationStereo')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FFS_REPO) not in sys.path:
    sys.path.insert(0, str(FFS_REPO))
from data_process.depth_backends.fast_foundation_stereo import FastFoundationStereoRunner
left = imageio.imread(FFS_REPO / 'demo_data/left.png')[..., :3]
right = imageio.imread(FFS_REPO / 'demo_data/right.png')[..., :3]
K = np.array([[423.0,0.0,320.0],[0.0,423.0,240.0],[0.0,0.0,1.0]], dtype=np.float32)
runner = FastFoundationStereoRunner(
    ffs_repo=FFS_REPO,
    model_path=FFS_REPO / 'weights/23-36-37/model_best_bp2_serialize.pth',
    scale=1.0,
    valid_iters=4,
    max_disp=192,
)
outputs = runner.run_batch([
    {'left_image': left, 'right_image': right, 'K_ir_left': K, 'baseline_m': 0.055, 'audit_mode': True}
    for _ in range(3)
])
print('batch_len', len(outputs))
print('audit0', outputs[0]['audit_stats'])
PY
```

Two-stage TRT batch-3 proof-of-life attempt:

```text
/home/zhangxinjie/miniconda3/envs/ffs-standalone/bin/python scripts/harness/verify_ffs_tensorrt_wsl.py --out_dir /home/zhangxinjie/proj-QQTT-v2/data/ffs_proof_of_life/trt_two_stage_batch3_864x480_wsl --batch_size 3 --skip_profiles
```

Single-engine TRT batch-3 proof-of-life attempt:

```text
/home/zhangxinjie/miniconda3/envs/ffs-standalone/bin/python scripts/harness/verify_ffs_single_engine_tensorrt_wsl.py --out_dir /home/zhangxinjie/proj-QQTT-v2/data/ffs_proof_of_life/trt_single_engine_batch3_864x480_wsl_fp32 --batch_size 3
```

## Outputs

PyTorch batch smoke:

- no artifact directory; this was a direct repo-integrated runner smoke

Two-stage TRT batch-3 attempt:

- `data/ffs_proof_of_life/trt_two_stage_batch3_864x480_wsl/feature_runner.onnx`
- `data/ffs_proof_of_life/trt_two_stage_batch3_864x480_wsl/post_runner.onnx`
- `data/ffs_proof_of_life/trt_two_stage_batch3_864x480_wsl/onnx.yaml`
- `data/ffs_proof_of_life/trt_two_stage_batch3_864x480_wsl/feature_engine_build.log`

Single-engine TRT batch-3 attempt:

- `data/ffs_proof_of_life/trt_single_engine_batch3_864x480_wsl_fp32/fast_foundationstereo.onnx`
- `data/ffs_proof_of_life/trt_single_engine_batch3_864x480_wsl_fp32/fast_foundationstereo.yaml`
- `data/ffs_proof_of_life/trt_single_engine_batch3_864x480_wsl_fp32/single_engine_build.log`

## Results

PyTorch batch smoke:

- `run_batch(...)` returned `3` outputs
- each output kept the current per-camera contract
- the repeated upstream demo-pair batch had:
  - `finite_ratio=1.0`
  - `positive_ratio=1.0`
  - `min_disparity≈34.40`
  - `max_disparity≈137.60`

Two-stage TRT batch-3 attempt:

- ONNX export completed for:
  - `feature_runner.onnx`
  - `post_runner.onnx`
  - `onnx.yaml`
- `feature_runner.engine` build failed on this machine
- TensorRT surfaced GPU memory exhaustion during build:
  - `Requested amount of GPU memory ... could not be allocated`
  - `OutOfMemory`
- `feature_engine_build.log` recorded `builder.build_serialized_network returned None`

Single-engine TRT batch-3 attempt:

- batch-3 single ONNX export completed
- FP32 TensorRT engine build failed on this machine
- TensorRT surfaced an internal Myelin/tactic-selection failure while building the batch-3 graph:
  - `MyelinCheckException ... Must have costs`
  - `Could not find any implementation for node {ForeignNode[/Concat.../Unsqueeze_166]}`
- `single_engine_build.log` recorded `builder.build_serialized_network returned None`

## Interpretation

- The repo-integrated strict batch viewer/runtime path is implemented and the PyTorch batch runner works.
- The current machine does not yet produce usable local batch-3 TRT artifacts at `864x480`:
  - two-stage TRT failed at build time due GPU memory exhaustion
  - single-engine TRT failed at build time due TensorRT internal tactic/Myelin failure
- Viewer startup validation now correctly requires batch-3 TRT engines for `--ffs_batch_mode strict3`, so these failing proof-of-life attempts do not silently degrade into incorrect runtime behavior.

## Current Boundary

- Live 3-camera hardware validation for strict batch mode was not run in this note.
- Current TRT batch support in the viewer is implementation-complete but artifact-blocked on this machine until compatible batch-3 engines are available.
