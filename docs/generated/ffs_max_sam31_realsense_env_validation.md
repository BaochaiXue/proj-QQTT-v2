# FFS-max SAM 3.1 RealSense Environment Validation

- Date: `2026-04-29`
- Repo root: `/home/xinjie/proj-QQTT-v2`
- Source env: `FFS-max`
- Target env: `FFS-max-sam31-rs`
- GPU observed during smoke: `NVIDIA GeForce RTX 4090`
- SAM 3.1 checkpoint: `/home/xinjie/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt`

## Goal

Clone `FFS-max` and add QQTT-compatible SAM 3.1 and RealSense support while preserving the FFS torch/CUDA stack.

The current official SAM 3.1 sources used for this setup are:

- Hugging Face checkpoint repo: `https://huggingface.co/facebook/sam3.1`
- Official code repo: `https://github.com/facebookresearch/sam3`

Important upstream notes confirmed from those sources:

- `facebook/sam3.1` is gated and hosts the SAM 3.1 checkpoint files, including `sam3.1_multiplex.pt`.
- The official SAM 3 repo says SAM 3.1 checkpoints require the latest repo code.
- The official SAM 3 repo lists Python 3.12+, PyTorch 2.7+, and CUDA 12.6+ as prerequisites.

## Environment Commands

Clone:

```text
conda create -y -n FFS-max-sam31-rs --clone FFS-max
```

RealSense / QQTT camera add-ons:

```text
conda run -n FFS-max-sam31-rs python -m pip install pyrealsense2==2.56.5.9235
conda run -n FFS-max-sam31-rs python -m pip install atomics pynput threadpoolctl
```

SAM 3.1 add-ons:

```text
conda run -n FFS-max-sam31-rs python -m pip install --no-deps --force-reinstall git+https://github.com/facebookresearch/sam3.git@main
conda run -n FFS-max-sam31-rs python -m pip install ftfy iopath regex webencodings
conda run -n FFS-max-sam31-rs python -m pip install ftfy==6.1.1 pycocotools
```

Observed official SAM 3 source commit during install:

```text
c97c893969003d3e6803fd5d679f21e515aef5ce
```

Checkpoint download and env wiring:

```text
mkdir -p /home/xinjie/.cache/huggingface/qqtt_sam31
conda run -n FFS-max-sam31-rs hf download facebook/sam3.1 sam3.1_multiplex.pt --local-dir /home/xinjie/.cache/huggingface/qqtt_sam31
conda env config vars set -n FFS-max-sam31-rs QQTT_SAM31_CHECKPOINT=/home/xinjie/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt
```

## Package Snapshot

Preserved FFS stack:

- `python==3.12.13`
- `torch==2.11.0+cu130`
- `torchvision==0.26.0`
- `cuda-toolkit==13.0.2`
- `nvidia-cuda-runtime==13.0.96`
- `nvidia-cudnn-cu13==9.19.0.56`
- `nvidia-nccl-cu13==2.28.9`
- `tensorrt-cu13==10.16.1.11`
- `triton==3.6.0`

Added support packages:

- `sam3==0.1.0` from official GitHub main at `c97c893969003d3e6803fd5d679f21e515aef5ce`
- `pyrealsense2==2.56.5.9235`
- `atomics==1.0.3`
- `pynput==1.8.1`
- `threadpoolctl==3.6.0`
- `ftfy==6.1.1`
- `iopath==0.1.10`
- `pycocotools==2.0.11`
- `regex==2026.4.4`
- `webencodings==0.5.1`

Existing source-stack package intentionally preserved:

- `numpy==2.4.4`

## Validation Commands And Outcomes

Torch/CUDA invariance:

```text
conda run -n FFS-max-sam31-rs python -c "import sys, torch; print(sys.version.split()[0]); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

Observed:

- Python `3.12.13`
- torch `2.11.0+cu130`
- torch CUDA `13.0`
- CUDA visible on `NVIDIA GeForce RTX 4090`

RealSense import and QQTT camera import:

```text
conda run -n FFS-max-sam31-rs python -c "import pyrealsense2 as rs; print(getattr(rs, '__version__', 'import-ok'))"
conda run -n FFS-max-sam31-rs python -c "import atomics, pynput, threadpoolctl; import qqtt.env; from qqtt.env import CameraSystem; print(CameraSystem.__name__)"
```

Observed:

- `pyrealsense2 import-ok`
- `CameraSystem`

SAM 3.1 import and builder:

```text
conda run -n FFS-max-sam31-rs python -c "import sam3, sam3.model_builder as mb; print(getattr(sam3, '__version__', 'unknown')); print(hasattr(mb, 'build_sam3_video_predictor')); print(hasattr(mb, 'download_ckpt_from_hf'))"
```

Observed:

- `sam3 0.1.0`
- `build_sam3_video_predictor=True`
- `download_ckpt_from_hf=True`

SAM 3.1 checkpoint:

```text
stat -c '%n %s bytes' /home/xinjie/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt
```

Observed:

- `3502755717` bytes

QQTT SAM 3.1 predictor initialization:

```text
conda run -n FFS-max-sam31-rs python -c "from scripts.harness.sam31_mask_helper import build_sam31_video_predictor, resolve_sam31_bpe_path, resolve_sam31_checkpoint_path; checkpoint=resolve_sam31_checkpoint_path(); print('checkpoint', checkpoint); print('bpe', resolve_sam31_bpe_path(checkpoint)); predictor, checkpoint = build_sam31_video_predictor(checkpoint_path=checkpoint, async_loading_frames=False, compile_model=False, max_num_objects=1); print('predictor_type', type(predictor).__name__); shutdown=getattr(predictor, 'shutdown', None); shutdown() if callable(shutdown) else None; print('sam31_predictor_init_ok')"
```

Observed:

- checkpoint resolved from `QQTT_SAM31_CHECKPOINT`
- BPE resolved from the installed official package:
  - `/home/xinjie/miniconda3/envs/FFS-max-sam31-rs/lib/python3.12/site-packages/sam3/assets/bpe_simple_vocab_16e6.txt.gz`
- predictor type: `Sam3MultiplexVideoPredictor`
- final status: `sam31_predictor_init_ok`

CLI/import paths:

```text
conda run -n FFS-max-sam31-rs python scripts/harness/probe_d455_ir_pair.py --help
conda run -n FFS-max-sam31-rs python record_data.py --help
conda run -n FFS-max-sam31-rs python data_process/record_data_align.py --help
conda run -n FFS-max-sam31-rs python scripts/harness/generate_sam31_masks.py --help
```

Observed:

- all commands exited `0`

Deterministic tests:

```text
conda run -n FFS-max-sam31-rs python -m unittest -v tests.test_sam31_mask_helper_smoke
conda run -n FFS-max-sam31-rs python -m unittest -v tests.test_record_data_preflight_message_smoke
conda run -n FFS-max-sam31-rs python scripts/harness/check_all.py
```

Observed:

- `tests.test_sam31_mask_helper_smoke`: `Ran 11 tests ... OK`
- `tests.test_record_data_preflight_message_smoke`: `Ran 5 tests ... OK`
- `scripts/harness/check_all.py`: passed after adding the repo camera-only `atomics pynput threadpoolctl` dependencies

## Caveats

- `conda run -n FFS-max-sam31-rs python -m pip check` reports:
  - `sam3 0.1.0 has requirement numpy<2,>=1.26, but you have numpy 2.4.4`
- I did not downgrade NumPy because `numpy==2.4.4` came from the cloned `FFS-max` environment and the requested priority was to keep the FFS max torch/CUDA stack intact.
- Runtime validation still succeeded with `numpy==2.4.4`: SAM 3.1 imported, the QQTT helper tests passed, and the `sam3.1_multiplex.pt` video predictor initialized.
- No physical D455 capture was run in this validation. RealSense support here means import/CLI/runtime wiring is present; hardware proof still belongs in the manual hardware checklist.
