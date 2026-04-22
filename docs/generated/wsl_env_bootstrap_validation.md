# WSL Environment Bootstrap Validation

- Date: `2026-04-21`
- Repo root: `/home/zhangxinjie/proj-QQTT-v2`
- External FFS repo: `/home/zhangxinjie/Fast-FoundationStereo`
- OS: `Ubuntu 24.04.2 LTS` on `WSL2`
- Conda: `25.7.0`
- GPU: `NVIDIA GeForce RTX 5090 Laptop GPU`
- NVIDIA driver / CUDA reported by `nvidia-smi`: `591.90 / 13.1`

## Environment Build Commands

```text
conda create -y -n ffs-standalone --clone rtx5090
conda create -y -n qqtt-ffs-compat --clone rtx5090
conda run -n ffs-standalone python -m pip install timm scikit-image pytest PyTurboJPEG gdown
conda run -n ffs-standalone python -m pip install --force-reinstall numpy==2.2.6 scipy==1.15.3
conda run -n qqtt-ffs-compat python -m pip install timm scikit-image pytest PyTurboJPEG gdown rerun-sdk==0.31.2
```

Notes:

- both target envs were bootstrapped from the local CUDA-capable `rtx5090` reference env
- `ffs-standalone` needed an explicit `numpy` / `scipy` reinstall after clone because the initial WSL import path hit a `scipy.special` error during `open3d` import
- `qqtt-ffs-compat` kept the optional `rerun-sdk==0.31.2` add-on and validated successfully afterward

## External Weights Download

Command used:

```text
conda run -n ffs-standalone gdown --folder https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap?usp=drive_link -O /home/zhangxinjie/Fast-FoundationStereo/weights
```

Observed files:

- `20-26-39/model_best_bp2_serialize.pth` = `67,971,170` bytes
- `20-30-48/model_best_bp2_serialize.pth` = `62,078,956` bytes
- `23-36-37/model_best_bp2_serialize.pth` = `71,098,210` bytes
- matching `cfg.yaml` files were written for all three checkpoints

## Validation Commands

FFS env import sanity:

```text
conda run -n ffs-standalone python -c "import torch, torchvision, timm, open3d, turbojpeg, skimage, gdown, numpy; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('timm', timm.__version__); print('open3d', open3d.__version__); print('numpy', numpy.__version__); print('skimage', skimage.__version__)"
```

QQTT + FFS env import sanity:

```text
conda run -n qqtt-ffs-compat python -c "import torch, torchvision, timm, open3d, turbojpeg, skimage, gdown, pyrealsense2, rerun, numpy; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('timm', timm.__version__); print('open3d', open3d.__version__); print('numpy', numpy.__version__); print('pyrealsense2 ok'); print('rerun', rerun.__version__)"
```

Official FFS demo proof-of-life:

```text
conda run -n ffs-standalone python scripts/harness/verify_ffs_demo.py --ffs_repo /home/zhangxinjie/Fast-FoundationStereo --model_path /home/zhangxinjie/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth --doc_path /home/zhangxinjie/proj-QQTT-v2/docs/generated/wsl_env_bootstrap_validation.md --out_dir /home/zhangxinjie/proj-QQTT-v2/data/ffs_proof_of_life/official_demo_wsl
```

QQTT deterministic validation:

```text
conda run -n qqtt-ffs-compat python scripts/harness/check_all.py
```

## Validation Outcomes

### `ffs-standalone`

- import sanity passed with:
  - `torch 2.7.0+cu128`
  - `torchvision 0.22.0+cu128`
  - `timm 1.0.26`
  - `open3d 0.19.0`
  - `numpy 2.2.6`
  - `skimage 0.25.2`
- official Fast-FoundationStereo demo passed
- verified output artifacts under `data/ffs_proof_of_life/official_demo_wsl/`:
  - `disp_vis.png`
  - `depth_meter.npy`
  - `cloud.ply`

### `qqtt-ffs-compat`

- import sanity passed with:
  - `torch 2.7.0+cu128`
  - `torchvision 0.22.0+cu128`
  - `timm 1.0.26`
  - `open3d 0.19.0`
  - `numpy 2.2.6`
  - `pyrealsense2` import ok
  - `rerun 0.31.2`
- `scripts/harness/check_all.py` exited `0`
- repo-side deterministic CLI / unittest / pytest surface passed in WSL

## Scope Notes

- this bootstrap validated Python imports, the external FFS demo, and the repo's deterministic checks
- no live RealSense capture was exercised in this pass
- hardware streaming remains a manual validation surface as required by repo policy
