# demo_2_max Native Ubuntu RTX 4090 Environment Validation

Date: 2026-05-17 local shell time

## Summary

Status: partially complete with explicit live-demo blockers.

- Native Ubuntu RTX 4090 host prerequisites: PASS
- `demo_2_max` conda environment: PASS
- PyTorch CUDA 13 stack: PASS
- TensorRT Python and `trtexec` availability: PASS with CLI caveat
- RealSense Python probe: PASS, three D455 devices detected
- Fast-FoundationStereo weights and RTX 4090 TensorRT engines: PASS
- EdgeTAM official repo and CUDA extension: PASS
- SAM3.1 code install: PASS
- SAM3.1 checkpoint: BLOCKED by gated Hugging Face access
- Repo deterministic checks: PASS
- Demo 2.1 formal live smoke: FAIL fast because repo-root `calibrate.pkl` is missing

No WSL `usbipd`, WSLg, WSL udev helper, or RTX 5090 TensorRT artifacts were
used as formal results.

## Machine

- Target: native Ubuntu RTX 4090 desktop
- GPUs: 2 x `NVIDIA GeForce RTX 4090`
- NVIDIA driver: `580.126.09`
- CUDA shown by `nvidia-smi`: `13.0`
- OS: Ubuntu 22.04.5 LTS
- Kernel: `6.8.0-106-generic`
- CUDA toolkit: `/usr/local/cuda`, `nvcc` release `13.2`, `V13.2.78`
- Conda: `conda 26.3.2` from `/home/xinjie/miniforge3`

## Environment

- Conda env: `demo_2_max`
- Python: `3.12.13`
- `torch`: `2.11.0+cu130`
- `torch.version.cuda`: `13.0`
- `torchvision`: `0.26.0+cu130`
- `triton`: `3.6.0`
- `torch.cuda.is_available()`: `True`
- CUDA device name: `NVIDIA GeForce RTX 4090`
- CUDA capability: `(8, 9)`
- `TORCH_CUDA_ARCH_LIST`: `8.9`
- `numpy`: `2.4.5`
- `open3d`: `0.19.0`
- `pyrealsense2`: import PASS
- `transformers`: `5.7.0`
- `accelerate`: `1.13.0`
- `timm`: `1.0.27`
- `sam3`: `0.1.0`

Activation hook:

```text
/home/xinjie/miniforge3/envs/demo_2_max/etc/conda/activate.d/demo_2_max_paths.sh
```

Important activation values:

```text
QQTT_REPO_ROOT=/home/xinjie/proj-QQTT-v2
FFS_REPO=/home/xinjie/Fast-FoundationStereo
EDGETAM_REPO=/home/xinjie/EdgeTAM
CUDA_HOME=/usr/local/cuda
TORCH_CUDA_ARCH_LIST=8.9
QQTT_SAM31_CHECKPOINT=/home/xinjie/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt
PYTHONPATH=$EDGETAM_REPO:$FFS_REPO:$QQTT_REPO_ROOT:...
LD_LIBRARY_PATH includes torch/lib and /usr/local/cuda/lib64
```

`pip check` caveat:

```text
sam3 0.1.0 has requirement numpy<2,>=1.26, but you have numpy 2.4.5.
```

This is the known SAM3 metadata mismatch caveat; `numpy` was kept at `2.4.5`
for the integrated environment.

## TensorRT

- Python package: `tensorrt-cu13==10.16.1.11`
- Python import and builder: PASS
- `trtexec`: `/usr/bin/trtexec`
- `trtexec --version`: prints TensorRT header `TensorRT v101601` but exits
  nonzero after showing the generic help and `Model missing or format not
  recognized`. The binary is present and usable; version reporting is noisy in
  this package layout.

## RealSense

Python probe:

```text
devices: 3
Intel RealSense D455 239222300781
Intel RealSense D455 239222303506
Intel RealSense D455 239222300412
```

No WSL-specific USB setup was run.

## External Repos And Assets

QQTT:

- Path: `/home/xinjie/proj-QQTT-v2`
- Sync: `git pull --ff-only origin main` PASS, already up to date

Fast-FoundationStereo:

- Path: `/home/xinjie/Fast-FoundationStereo`
- Commit: `f8442a5f406d3058e060c48acbd019963e54f490`
- Required weight: `/home/xinjie/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth`
- Required config: `/home/xinjie/Fast-FoundationStereo/weights/20-30-48/cfg.yaml`
- Status: PASS, both files present

EdgeTAM:

- Path: `/home/xinjie/EdgeTAM`
- Commit: `7711e012a30a2402c4eaab637bdb00a521302c91`
- Checkpoint: `/home/xinjie/EdgeTAM/checkpoints/edgetam.pt`
- CUDA extension: PASS, `import sam2._C` succeeds with activation hook

SAM3.1:

- Code: installed from `facebookresearch/sam3.git@main`
- Package version: `sam3==0.1.0`
- Checkpoint path: `/home/xinjie/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt`
- Checkpoint status: BLOCKED, file is missing
- Download blocker: Hugging Face reported gated repo access for `facebook/sam3.1`

## FFS TensorRT Engines

Batch 1 official local Demo 2 / Demo 2.1 engine:

```text
/home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864
```

Artifacts:

```text
feature_runner.engine  21M
post_runner.engine     21M
onnx.yaml
feature_engine_build.log
post_engine_build.log
demo_out/
```

Batch 3 isolated Demo 2.2/default-check engine:

```text
/home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5_batch3/engines/model_20-30-48_iters_4_res_480x864_batch3
```

Artifacts:

```text
feature_runner.engine  22M
post_runner.engine     46M
batch3_metadata.json
batch3_metadata.yaml
onnx.yaml
feature_engine_build.log
post_engine_build.log
```

Engine contract:

- Model: `20-30-48`
- Valid iters: `4`
- Max disparity: `192`
- Real input: `848x480`
- TensorRT input shape: `480x864`
- Padding policy: pad width `848 -> 864`
- Builder optimization level: `5`
- GPU: RTX 4090, compute capability `8.9`
- TensorRT: `10.16.1.11`
- Torch CUDA: `13.0`

The QQTT default FFS TensorRT paths were updated from historical RTX
5090/WSL-labeled paths to the local 4090-native paths above.

## Validation

Passed:

```bash
conda run -n demo_2_max --no-capture-output python -m py_compile \
  scripts/harness/verify_ffs_tensorrt_wsl.py \
  scripts/harness/verify_ffs_single_engine_tensorrt_wsl.py \
  scripts/ffs_trt/build_batch3_4090_engine.py \
  data_process/depth_backends/ffs_defaults.py
```

Passed:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v \
  tests.test_ffs_trt_batch3_scripts
```

Observed:

```text
Ran 7 tests in 0.006s
OK
```

Passed:

```bash
conda run -n demo_2_max --no-capture-output python scripts/harness/check_harness_catalog.py
```

Observed:

```text
[harness-catalog] catalog checks passed
```

Passed:

```bash
conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py
```

Observed:

```text
Ran 253 tests in 2.094s
OK
[check] quick deterministic checks passed
```

Passed:

```bash
conda run -n demo_2_max --no-capture-output python demo_v2/realtime_masked_edgetam_pcd.py --help
```

Passed:

```bash
conda run -n demo_2_max --no-capture-output python demo_v2_1/realtime_three_view_masked_fused_pcd.py --dry-run --preset official-lowfps
```

The dry-run printed the expected `official-lowfps` contract with FFS TensorRT
model dir:

```text
/home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864
```

Passed:

```bash
conda run -n demo_2_max --no-capture-output python demo_v2/realtime_masked_edgetam_pcd.py \
  --profile 848x480 \
  --fps 15 \
  --depth-source ffs \
  --track-mode none \
  --pcd-mode none \
  --render-mode none \
  --duration-s 3 \
  --debug
```

This is a short headless RealSense plus local FFS TensorRT smoke. It exited
with status 0.

Failed as expected until real calibration is present:

```bash
conda run -n demo_2_max --no-capture-output python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset official-lowfps \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --duration-s 3 \
  --render-mode none \
  --debug
```

Observed:

```text
FileNotFoundError: Demo 2.1 requires calibrate.pkl for world fusion: /home/xinjie/proj-QQTT-v2/calibrate.pkl
```

This fail-fast behavior is correct for Demo 2.1 world-coordinate fusion. The
fixture `calibrate.pkl` is used only in unit tests; no fake repo-root
calibration was installed.

## Remaining Operator Actions

1. Authenticate to Hugging Face with an account approved for `facebook/sam3.1`,
   then download `sam3.1_multiplex.pt` to:

```text
/home/xinjie/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt
```

2. Produce or restore the real three-camera calibration at:

```text
/home/xinjie/proj-QQTT-v2/calibrate.pkl
```

3. After those two files exist, rerun the formal three-camera live smoke and
   longer performance candidate.
