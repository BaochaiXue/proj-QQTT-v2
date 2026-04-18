# 2026-04-17 SAM 3.1 Sidecar Mask Helper

## Goal

Add a bounded `SAM 3.1` sidecar helper inside the current workspace so object masks can be generated from QQTT case assets without running scripts out of the local `PhysTwin` checkout.

The helper must:

- live under `scripts/harness/` rather than `data_process/segment*.py`
- support QQTT case layouts that expose either:
  - `color/<camera>.mp4`
  - `color/<camera>/*.png`
- keep checkpoints external to the repo and accept an explicit path or Hugging Face cache resolution
- write PhysTwin-compatible mask artifacts under a caller-selected output directory

## Non-Goals

- no change to canonical recording or alignment outputs
- no change to `env_install/env_install.sh`
- no requirement that deterministic repo checks install `sam3`
- no attempt to fold segmentation into the repo's formal in-scope product surface

## Files To Touch

- new `scripts/harness/sam31_mask_helper.py`
- new `scripts/harness/generate_sam31_masks.py`
- new smoke/unit tests for path discovery and prompt parsing
- `scripts/harness/README.md`

## Implementation Plan

1. add a helper module that:
   - resolves case-local RGB sources from either mp4 sidecars or per-camera frame directories
   - normalizes text prompts
   - extracts temporary JPEG frames when needed
   - lazily imports `sam3` only during real inference
   - writes `mask/mask_info_<camera>.json` plus `mask/<camera>/<obj>/<frame>.png`
2. add a thin CLI wrapper that mirrors existing harness patterns and writes a compact `summary.json`
3. add deterministic tests for:
   - prompt splitting
   - case source discovery
   - output-path contract
   - CLI `--help`
4. document the helper in the harness map, with an explicit note that checkpoints remain external and Hugging Face login is expected beforehand

## Validation Plan

- `python scripts/harness/generate_sam31_masks.py --help`
- `python -m unittest -v tests.test_sam31_mask_helper_smoke`
