# External Dependencies

## Fast-FoundationStereo

- External repo path: `/home/zhangxinjie/Fast-FoundationStereo`
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

## Selected Checkpoint

- Checkpoint name: `23-36-37`
- Model file: `/home/zhangxinjie/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth`
- Config file: `/home/zhangxinjie/Fast-FoundationStereo/weights/23-36-37/cfg.yaml`
- Model file size: `71,098,210` bytes
- Additional local benchmark checkpoints:
  - `/home/zhangxinjie/Fast-FoundationStereo/weights/20-26-39/model_best_bp2_serialize.pth`
  - `/home/zhangxinjie/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth`
- How obtained: downloaded from the official Google Drive weights folder referenced by the Fast-FoundationStereo README using `conda run -n ffs-standalone gdown --folder`
- Date obtained: `2026-04-21`
- Runtime role: baseline checkpoint for official demo validation, proof-of-life, and optional `--depth_backend ffs|both` alignment
