# EdgeTAM-max Environment Validation

## Target

- Env: `edgetam-max`
- Platform: WSL2 Ubuntu 24.04.4 LTS on RTX 5090 Laptop
- Python: 3.12.13
- torch: 2.11.0+cu130
- torchvision: 0.26.0+cu130
- EdgeTAM repo: `facebookresearch/EdgeTAM`
- EdgeTAM commit: `7711e012a30a2402c4eaab637bdb00a521302c91`
- Checkpoint: `checkpoints/edgetam.pt`
- Config: `configs/edgetam.yaml`
- CUDA extension: `sam2._C` required

## System / CUDA

- `nvidia-smi` worked in WSL.
- Driver Version: 591.90
- `nvidia-smi` CUDA Version: 13.1
- GPU: NVIDIA GeForce RTX 5090 Laptop GPU
- Detected compute capability: `(12, 0)`
- During initial `edgetam-max` setup, system `/usr/local/cuda/bin/nvcc` was CUDA 12.9, so it was not used.
- CUDA 13 toolkit was initially installed inside the conda env, not with `sudo apt install cuda`.
- No `cuda` or `cuda-drivers` apt package was installed.
- Conda env `nvcc`: CUDA 13.0.88
- `CUDA_HOME`: `/home/zhangxinjie/miniconda3/envs/edgetam-max`
- `TORCH_CUDA_ARCH_LIST`: `12.0`

Note: the conda `cuda-toolkit` package set includes `cuda-driver-dev` headers/stubs. This is not a Linux NVIDIA driver install inside WSL.

Post-validation update: a shared WSL CUDA 13 toolkit was later installed at `/usr/local/cuda-13.2`, with `/usr/local/cuda` resolving to that shared toolkit. Future CUDA 13-family extension builds should use `/usr/local/cuda` instead of installing another env-local CUDA toolkit. Keep `edgetam-max` on its recorded env-local CUDA path until the environment is intentionally rebuilt.

## Validation Result

- [x] `nvidia-smi` works in WSL
- [x] No Linux NVIDIA driver install was run in WSL
- [x] `nvcc` is CUDA 13.x
- [x] `torch.version.cuda` starts with `13.`
- [x] `torch.cuda.is_available()`
- [x] GPU is RTX 5090 Laptop
- [x] `TORCH_CUDA_ARCH_LIST` matches detected GPU capability
- [x] EdgeTAM repo commit recorded
- [x] `checkpoints/edgetam.pt` exists, about 54 MB
- [x] `import sam2._C` succeeds
- [x] Image predictor smoke passes
- [x] Video predictor `add_new_mask` smoke passes
- [x] Video predictor `add_new_points_or_box` box smoke passes

## Verification Summary

`verify_edgetam_max.py` completed with:

- `torch cuda: 13.0`
- `SAM2 CUDA extension import OK`
- image masks: `(3, 512, 512)` `float32`
- mask-init video propagation: 8 frames, about 25.6 fps in the saved run
- box-init video propagation: 8 frames, about 34.7 fps in the saved run
- final status: `edgetam-max verification PASS`

## Still Object Case Validation

After environment validation, EdgeTAM was run on a real still-object case:

- case: `/home/zhangxinjie/proj-QQTT-v2/data/still_object/ffs203048_iter4_trt_level5/both_30_still_object_round1_20260428`
- camera: `0`
- frames: `30`
- frame source: aligned `color/0/*.png`, converted to a temporary JPEG video folder for EdgeTAM
- initialization mask source: `sam31_masks/mask/0/0/0.png`
- bbox prompt: derived from the same frame-0 mask, `[290.0, 152.0, 619.0, 480.0]`
- image predictor best score: `0.96875`
- mask-init propagation: 30 frames, about `4.86` fps, mean IoU vs SAM3.1 masks `0.9891`, min IoU `0.9861`
- box-init propagation: 30 frames, about `29.04` fps, mean IoU vs SAM3.1 masks `0.9772`, min IoU `0.9705`
- status: `edgetam still object validation PASS`

Outputs:

- `/home/zhangxinjie/proj-QQTT-v2/data/experiments/edgetam_still_object_round1_cam0_validation_20260501/summary.json`
- `/home/zhangxinjie/proj-QQTT-v2/data/experiments/edgetam_still_object_round1_cam0_validation_20260501/mask/overlay_frame_0000.png`
- `/home/zhangxinjie/proj-QQTT-v2/data/experiments/edgetam_still_object_round1_cam0_validation_20260501/mask/overlay_frame_0015.png`
- `/home/zhangxinjie/proj-QQTT-v2/data/experiments/edgetam_still_object_round1_cam0_validation_20260501/mask/overlay_frame_0029.png`
- `/home/zhangxinjie/proj-QQTT-v2/data/experiments/edgetam_still_object_round1_cam0_validation_20260501/box/overlay_frame_0000.png`
- `/home/zhangxinjie/proj-QQTT-v2/data/experiments/edgetam_still_object_round1_cam0_validation_20260501/box/overlay_frame_0015.png`
- `/home/zhangxinjie/proj-QQTT-v2/data/experiments/edgetam_still_object_round1_cam0_validation_20260501/box/overlay_frame_0029.png`

## Environment Hooks

The env has conda activation hooks:

- `/home/zhangxinjie/miniconda3/envs/edgetam-max/etc/conda/activate.d/edgetam-max.sh`
- `/home/zhangxinjie/miniconda3/envs/edgetam-max/etc/conda/deactivate.d/edgetam-max.sh`

These set `CUDA_HOME` to the conda env and add CUDA 13 plus PyTorch `torch/lib` paths to `LD_LIBRARY_PATH`, so `import sam2._C` works after `conda activate edgetam-max`.

## Saved Reports

- `/home/zhangxinjie/EdgeTAM/edgetam-max-verify.txt`
- `/home/zhangxinjie/EdgeTAM/edgetam-max-pip-freeze.txt`
- `/home/zhangxinjie/EdgeTAM/edgetam-max-conda-list.txt`
- `/home/zhangxinjie/EdgeTAM/edgetam-max-nvidia-smi.txt`
- `/home/zhangxinjie/EdgeTAM/edgetam-max-nvcc.txt`
- `/home/zhangxinjie/EdgeTAM/edgetam-max-git-commit.txt`
- `/home/zhangxinjie/proj-QQTT-v2/docs/generated/edgetam_max_env_validation.md`
