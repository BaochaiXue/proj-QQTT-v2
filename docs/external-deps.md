# External Dependencies

## Fast-FoundationStereo

- External repo path: `C:\Users\zhang\external\Fast-FoundationStereo`
- Purpose: optional external stereo depth backend evaluation for D455 IR stereo pairs
- Vendoring policy: keep external, do not copy source or weights into the QQTT repo

## Selected Checkpoint

- Checkpoint name: `23-36-37`
- Model file: `C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth`
- Config file: `C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\cfg.yaml`
- Model file size: `71,098,210` bytes
- How obtained: downloaded from the official Google Drive weights folder referenced by the Fast-FoundationStereo README using `gdown --folder`
- Date obtained: `2026-04-05`
- Runtime role: baseline checkpoint for official demo validation, proof-of-life, and optional `--depth_backend ffs|both` alignment
