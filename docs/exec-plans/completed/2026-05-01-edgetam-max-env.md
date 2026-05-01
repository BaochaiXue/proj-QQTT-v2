# EdgeTAM-max Environment Setup

## Goal

Install and validate an isolated `edgetam-max` conda environment for EdgeTAM on
RTX 5090 Laptop WSL2.

## Constraints

- Do not install Linux NVIDIA drivers inside WSL.
- Do not install `cuda` or `cuda-drivers`.
- Install only CUDA toolkit 13 if CUDA 13 `nvcc` is missing.
- Do not modify `SAM21-max`.
- Do not install EdgeTAM into `SAM21-max`.
- Do not pass `vos_optimized=True` to EdgeTAM video predictor APIs.

## Validation

- `nvidia-smi` works in WSL and sees RTX 5090 Laptop.
- Python `3.12` in `edgetam-max`.
- `torch==2.11.0` and `torchvision==0.26.0` with CUDA `13.x`.
- CUDA extension import succeeds: `import sam2._C`.
- EdgeTAM image predictor smoke passes.
- EdgeTAM video predictor with initial mask passes.
- EdgeTAM video predictor with box prompt passes.
- Save generated validation outputs and repo validation note.

## Outcome

- Installed `edgetam-max` with Python 3.12.13.
- Used conda-local CUDA toolkit 13.0.88 because system `nvcc` was CUDA 12.9 and sudo apt installation was unavailable.
- Installed `torch==2.11.0+cu130` and `torchvision==0.26.0+cu130`.
- Cloned EdgeTAM main at `7711e012a30a2402c4eaab637bdb00a521302c91`.
- Built `sam2._C` for detected RTX 5090 Laptop capability `12.0`.
- Added conda activation hooks so `import sam2._C` resolves PyTorch shared libraries after `conda activate edgetam-max`.
- Saved validation details in `docs/generated/edgetam_max_env_validation.md`.
