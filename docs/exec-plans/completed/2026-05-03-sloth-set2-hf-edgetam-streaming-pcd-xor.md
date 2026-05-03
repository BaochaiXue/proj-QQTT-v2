# Sloth Set 2 HF EdgeTAM Streaming PCD XOR

## Goal

Run the HF EdgeTAMVideo path on `data/different_types/sloth_set_2_motion_ffs` as a true frame-by-frame streaming experiment, initialized from the SAM 3.1 frame-0 mask, then render an EdgeTAM-vs-SAM3.1 fused PCD XOR GIF.

## Implementation

- Keep the already-removed HF EdgeTAM smoke artifacts deleted and remove the remaining synthetic smoke CLI from the harness catalog.
- Extend `run_hf_edgetam_streaming_realcase.py` with custom case arguments and a separate `sam31_mask_root` so the Sloth Set 2 run can reuse generated SAM 3.1 masks without hardcoding default benchmark cases.
- Preserve the streaming contract in the HF runner: read one PNG frame at a time, preprocess one image, call `model(inference_session=..., frame=...)`, and never pass a full video path.
- Add a focused Sloth Set 2 experiment CLI that renders a single EdgeTAM row by three camera columns PCD XOR panel against SAM 3.1.
- Add lightweight tests for custom case selection, mask output labels, and single-variant PCD overlay rendering.

## Validation

- `python scripts/harness/check_harness_catalog.py`
- `python -m unittest -v tests.test_sam21_checkpoint_ladder_panel_smoke`
- `python scripts/harness/check_all.py`
- Run the three operator commands for SAM3.1 mask generation, HF EdgeTAM streaming, and PCD XOR GIF generation.

## Outcome

- SAM3.1 reference masks were generated with `FFS-SAM-RS` because `FFS-max-sam31-rs` is not present on this machine.
- HF EdgeTAM streaming ran in `edgetam-hf-stream` on all 93 frames for cam0/cam1/cam2 with `mask` prompt mode.
- PCD XOR GIF and benchmark reports were written under `result/sloth_set_2_motion_ffs_hf_edgetam_streaming_pcd_xor/` and `docs/generated/`.
- Final validation passed:
  - `conda run --no-capture-output -n SAM21-max python scripts/harness/check_harness_catalog.py`
  - `conda run --no-capture-output -n SAM21-max python -m unittest -v tests.test_sam21_checkpoint_ladder_panel_smoke`
  - `conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py`
