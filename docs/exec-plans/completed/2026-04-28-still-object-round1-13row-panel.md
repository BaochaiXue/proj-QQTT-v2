# 2026-04-28 Still Object Round 1 13-Row Panel

## Goal

Regenerate still-object round 1 as a single aligned/masked experiment using the current FFS default:

- environment: `FFS-SAM-RS`
- checkpoint: `20-30-48`
- valid iterations: `4`
- runtime: two-stage ONNX/TensorRT
- builder optimization level: `5`
- real input: `848x480`, padded to `864x480`

## Scope

- Align raw `data_collect/both_30_still_object_round1_20260428` into an aligned static case that contains native `depth/`, `ir_left/`, `ir_right/`, and FFS `depth_ffs/`.
- Use the level-5 TensorRT runner for FFS alignment.
- Generate or reuse SAM 3.1 object masks for frame 0.
- Produce a 13x3 panel with the requested RGB, IR, masked depth, projected PCD, enhanced-filtered PCD, removed-point projection, and direct RGB removed-point rows.
- Keep outputs inside one experiment result folder.

## Validation

- Confirm aligned case metadata records the TRT FFS setting.
- Confirm masks exist for all three cameras.
- Run the new visualization workflow on frame 0.
- Run focused smoke checks and the deterministic harness guard if code changes are made.

## Result

- Output root: `data/experiments/still_object_round1_projection_panel_13x3_ffs203048_iter4_trt_level5`
- Aligned case: `aligned/both_30_still_object_round1_20260428`
- Panel: `panel/still_object_round1_projection_panel_13x3_frame_0000.png`
- Text summary: `panel/result_summary.txt`
- JSON summary: `panel/summary.json`

## Notes

- The raw still-object round1 files use source frame ids `136..190`, not `0..29`; the harness default was set to that range.
- The script can resume after an interrupted SAM run by regenerating only missing camera masks.
- The run used `FFS-SAM-RS`, `20-30-48`, `valid_iters=4`, two-stage TensorRT, `builder_optimization_level=5`, and real `848x480` inputs padded to `864x480`.

## Checks

- `python -m unittest -v tests.test_still_object_projection_panel_smoke`
- `python -m unittest -v tests.test_floating_point_diagnostics_smoke tests.test_enhanced_phystwin_removed_overlay_smoke.EnhancedPhystwinRemovedOverlaySmokeTest.test_trace_masks_match_existing_enhanced_filter`
- `python scripts/harness/check_all.py`
