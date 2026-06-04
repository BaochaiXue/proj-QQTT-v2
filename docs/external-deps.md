# External Dependencies

## Fast-FoundationStereo

- External repo path: sibling `../Fast-FoundationStereo` by default, resolved from the QQTT repo root
- Purpose: optional external stereo depth backend evaluation for D455 IR stereo pairs
- Vendoring policy: keep external, do not copy source or weights into the QQTT repo

## SAM 3.1

- Official code repo: `https://github.com/facebookresearch/sam3`
- Official checkpoint repo: `https://huggingface.co/facebook/sam3.1`
- Local checkpoint path: `/home/xinjie/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt`
- Model file size: `3,502,755,717` bytes
- How obtained: `conda run -n FFS-max-sam31-rs hf download facebook/sam3.1 sam3.1_multiplex.pt --local-dir /home/xinjie/.cache/huggingface/qqtt_sam31`
- Date obtained: `2026-04-29`
- Runtime role: optional SAM 3.1 sidecar mask generation for aligned-case visualization diagnostics
- Vendoring policy: keep external, do not copy source or weights into the QQTT repo

## Demo 3.3 FuturePhysTwin / SAM 3D Objects Warmup

- FuturePhysTwin local root: `/home/xinjie/FuturePhysTwin`
- Original SAM 3D Objects repo: `https://github.com/facebookresearch/sam-3d-objects`
- Original SAM 3D Objects local root: `/home/xinjie/external/sam-3d-objects`
- Conda environment: `demo_3_3_max`
- Demo 3.3 default Python launcher: current Python interpreter. Start Demo 3.3
  with `conda run --no-capture-output -n demo_3_3_max python ...` so the live
  runtime, detached completion worker, and FuturePhysTwin/SAM3D route inherit
  one environment.
- Runtime role: explicitly enabled Demo 3.3 warmup-only single-view
  shape-prior generation using FuturePhysTwin's `image_upscale.py ->
  segment_util_image.py -> data_process_sam3d/shape_prior.py ->
  data_process/align.py -> data_process_sam3d/data_process_sample.py
  --shape_prior` route. Demo 3.3 keeps this disabled by default for live demo
  runs.
- Vendoring policy: keep FuturePhysTwin, SAM 3D Objects, checkpoints, and
  weights external; Demo 3.3 writes only diagnostic cache artifacts under its
  output root and does not change formal aligned-case outputs

## CoTracker3

- Official code repo: `https://github.com/facebookresearch/co-tracker`
- Official checkpoint repo: `https://huggingface.co/facebook/cotracker3`
- Local repo path: `/home/xinjie/co-tracker`
- Local checkpoint directory: `/home/xinjie/co-tracker/checkpoints`
- Local checkpoints:
  - `scaled_online.pth`: `101,695,610` bytes,
    sha256 `205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218`
  - `scaled_offline.pth`: `101,890,938` bytes,
    sha256 `2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834`
  - `baseline_online.pth`: `101,694,458` bytes,
    sha256 `8b30b2f239de9987323b729d9115cc5163720a07348a97d045095cd9ebdb7b3a`
  - `baseline_offline.pth`: `101,889,786` bytes,
    sha256 `da09bbac871f7398e5b29c4de5213652658949737bc158840b101678ba8ad1df`
- How obtained: reconciled all `.pth` files with
  `/home/xinjie/.local/bin/hf download facebook/cotracker3 --local-dir /home/xinjie/co-tracker/checkpoints --include "*.pth"`
- Date obtained: `2026-05-17`
- Runtime role: Demo 3 async CoTracker3 online overlay baseline and offline /
  baseline model availability for replay, benchmark, and comparison diagnostics
- PyTorch Hub cache: `~/.cache/torch/hub/checkpoints/*.pth` points at the
  local checkpoint files so Demo 3's default torch.hub path can start without
  redownloading weights
- Vendoring policy: keep external, do not copy source or weights into the QQTT repo

## Selected Checkpoint

- Checkpoint name: `23-36-37`
- Model file: `../Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth`
- Config file: `../Fast-FoundationStereo/weights/23-36-37/cfg.yaml`
- Model file size: `71,098,210` bytes
- Additional local benchmark checkpoints:
  - `../Fast-FoundationStereo/weights/20-26-39/model_best_bp2_serialize.pth`
  - `../Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth`
- How obtained: downloaded from the official Google Drive weights folder referenced by the Fast-FoundationStereo README using `conda run -n ffs-standalone gdown --folder`
- Date obtained: `2026-04-21`
- Runtime role: baseline checkpoint for official demo validation, proof-of-life, and optional `--depth_backend ffs|both` alignment
