# External Dependencies

## Fast-FoundationStereo

- External repo path: `/home/zhangxinjie/Fast-FoundationStereo`
- Purpose: optional external stereo depth backend evaluation for D455 IR stereo pairs
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
