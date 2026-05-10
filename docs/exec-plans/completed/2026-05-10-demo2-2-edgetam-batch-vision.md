# 2026-05-10 Demo 2.2 EdgeTAM Batch Vision Probe

## Goal

Add an experiment option that batches the three camera RGB frames through the
HF EdgeTAM vision encoder, then reuses the split features in each independent
camera video session.

## Scope

- Keep the current Demo 2.2 default unchanged.
- Add an explicit CLI option for the batch-vision path.
- Implement only the lightweight feature-cache path:
  `processor(images=[cam0, cam1, cam2]) -> model.get_image_features(batch=3)`.
- Keep memory attention, mask decoder, memory encoder, and session state
  per-camera.
- Add deterministic tests and profile metrics.
- Run a hardware profile if RealSense/RTX runtime is available.

## Validation

- PASS: targeted unit tests for contract and batch feature cache splitting.
- PASS: Demo 2.2 dry-run contract for the new option.
- PASS: hardware profile completed with `--no-parallel-init`.
- Result: EdgeTAM cycle median changed from the prior batch3 reference
  `109.17 ms` to `107.46 ms`; this is a working path but not enough to make
  it the default.
- PASS: `scripts/harness/check_all.py`.
