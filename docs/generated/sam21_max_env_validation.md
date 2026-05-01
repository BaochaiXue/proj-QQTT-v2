# SAM2.1 MAX Environment Validation

Date: 2026-05-01

## Summary

Status: PASS.

`SAM21-max` is installed on WSL2 Ubuntu for the RTX 5090 Laptop GPU with Python 3.12, PyTorch 2.11.0 CUDA 13.0, SAM2 latest main, SAM2.1 Hiera Large, image predictor support, video predictor support, and a compiled SAM2 CUDA extension. The SAM2 extension build was forced to fail hard on errors; no silent fallback was accepted.

The final warmed still-object round1 video segmentation result is:

| Camera | Propagate ms/frame | Pipeline ms/frame |
| --- | ---: | ---: |
| 0 | 49.50 | 66.94 |
| 1 | 40.49 | 58.41 |
| 2 | 41.04 | 58.41 |
| Mean | 43.68 | 61.25 |

`propagate ms/frame` measures `propagate_in_video` over 30 output frames. `pipeline ms/frame` amortizes `init_state + first prompt + propagate_in_video` over 30 frames, after a same-process warmup pass.

## Hardware / Base Env

- GPU: `NVIDIA GeForce RTX 5090 Laptop GPU`
- Driver: `591.90`
- Compute capability: `12.0`
- Environment: `SAM21-max`
- Python: `3.12.13`
- Torch: `2.11.0+cu130`
- Torch CUDA: `13.0`
- Torchvision: `0.26.0+cu130`
- CUDA available: `True`

`/usr/local/cuda` points to CUDA 12.9 on this machine, so it was not used for the SAM2 extension build. The build used the env-local CUDA 13 path:

```text
/home/zhangxinjie/miniconda3/envs/SAM21-max/lib/python3.12/site-packages/nvidia/cu13
```

## CUDA 13 Compiler Packages

Pinned NVIDIA pip packages:

```text
nvidia-cuda-runtime 13.0.96
nvidia-nvvm 13.0.88
nvidia-cuda-crt 13.0.88
nvidia-cuda-nvcc 13.0.88
nvidia-cuda-cccl 13.0.85
```

Additional repo-check dependency installed after SAM2 validation:

```text
atomics 1.0.3
cffi 2.0.0
pycparser 3.0
```

`nvcc --version`:

```text
Build cuda_13.0.r13.0/compiler.36424714_0
```

`torch.utils.cpp_extension.CUDA_HOME` resolved to the same env-local CUDA 13 path.

One env-local symlink was needed because the CUDA runtime wheel provides `libcudart.so.13`, while the PyTorch extension linker asks for `-lcudart`:

```text
$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so -> libcudart.so.13
```

## SAM2 Source / Checkpoint

SAM2 source:

```text
/home/zhangxinjie/external/sam2
2b90b9f5ceec907a1c18123530e92e794ad901a4 2024-12-15 16:47:17 -0800 remove `.pin_memory()` in `obj_pos` of `SAM2Base` to resolve and error in MPS (#495)
```

Checkpoint:

```text
/home/zhangxinjie/.cache/huggingface/sam2.1/sam2.1_hiera_large.pt
sha256: 2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318
size: 857M
```

Config:

```text
configs/sam2.1/sam2.1_hiera_l.yaml
```

The validation intentionally uses the SAM2.1 config with the SAM2.1 checkpoint. It does not mix `configs/sam2/sam2_hiera_l.yaml` with `sam2.1_hiera_large.pt`.

## Install Commands

Environment was cloned from the existing CUDA 13 / torch 2.11 `FFS-max` base:

```bash
conda create -y -n SAM21-max --clone FFS-max
```

CUDA compiler packages:

```bash
conda run -n SAM21-max python -m pip install --force-reinstall --no-deps \
  nvidia-cuda-runtime==13.0.96 \
  nvidia-nvvm==13.0.88 \
  nvidia-cuda-crt==13.0.88 \
  nvidia-cuda-nvcc==13.0.88 \
  nvidia-cuda-cccl==13.0.85
```

SAM2 extension install was forced to compile and fail on build errors:

```bash
conda run --no-capture-output -n SAM21-max bash -lc '
set -euo pipefail
export CUDA_HOME="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export SAM2_BUILD_CUDA=1
export SAM2_BUILD_ALLOW_ERRORS=0
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=4
python -m pip install -v --no-build-isolation --no-cache-dir -e /home/zhangxinjie/external/sam2
'
```

Checkpoint download:

```bash
curl -L --fail --continue-at - \
  --output /home/zhangxinjie/.cache/huggingface/sam2.1/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

## Build Issues Fixed

1. `fatal error: nv/target: No such file or directory`

   Fixed by installing CUDA CCCL headers: `nvidia-cuda-cccl==13.0.85`.

2. `ptxas fatal: Unsupported .version 9.2; current version is '9.0'`

   The env had `nvidia-nvvm 13.2` paired with `nvidia-cuda-nvcc 13.0`. Fixed by pinning NVVM, CRT, runtime, nvcc, and CCCL to coherent CUDA 13.0 package versions.

3. `ld: cannot find -lcudart`

   Fixed with the env-local `libcudart.so -> libcudart.so.13` symlink described above.

4. `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`

   This occurred only in the `vos_optimized=True` video predictor path. The benchmark now calls `torch.compiler.cudagraph_mark_step_begin()` before each generator step, matching PyTorch's runtime guidance for repeated compiled CUDA Graph invocations.

## Hard Validation

SAM2 extension import:

```text
sam2_file /home/zhangxinjie/external/sam2/sam2/__init__.py
sam2_extension /home/zhangxinjie/external/sam2/sam2/_C.so
sam2_extension_attrs ['get_connected_componnets']
```

`pip check`:

```text
No broken requirements found.
```

Repo deterministic checks:

```text
conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py
Ran 72 tests in 19.130s
OK
quick deterministic checks passed
```

Image predictor smoke on `data_collect/both_30_still_object_round1_20260428/color/0/136.png`:

```text
image_predictor_ok (1, 480, 848) [0.9765625] 70715.0
```

Video predictor:

```text
predictor_class: SAM2VideoPredictorVOS
vos_optimized_requested: true
compile_image_encoder_requested: true
compiled_vos_components_requested:
  - image_encoder
  - memory_encoder
  - memory_attention
  - sam_prompt_encoder
  - sam_mask_decoder
```

## Benchmark Command

Final warmed benchmark command:

```bash
conda run --no-capture-output -n SAM21-max bash -lc '
set -euo pipefail
export CUDA_HOME="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
cd /home/zhangxinjie/proj-QQTT-v2
python docs/generated/sam21_max_still_object_video_benchmark.py \
  --no-image-smoke \
  --warmup-camera 0 \
  --json-output docs/generated/sam21_max_still_object_video_benchmark_results.json \
  > docs/generated/sam21_max_still_object_video_benchmark_warm_run.log 2>&1
'
```

Benchmark script:

```text
docs/generated/sam21_max_still_object_video_benchmark.py
```

Final warmed JSON:

```text
docs/generated/sam21_max_still_object_video_benchmark_results.json
```

Retained comparison outputs:

```text
docs/generated/sam21_max_still_object_video_benchmark_cold_results.json
docs/generated/sam21_max_still_object_video_benchmark_no_warmup_results.json
docs/generated/sam21_max_still_object_video_benchmark_run.log
docs/generated/sam21_max_still_object_video_benchmark_hot_run.log
docs/generated/sam21_max_still_object_video_benchmark_warm_run.log
```
