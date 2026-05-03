# HF EdgeTAM Streaming Default Vision Compile

## Goal

Make the validated `vision-reduce-overhead` compile path the default for the HF
EdgeTAM realcase streaming benchmark while keeping an explicit eager fallback.

## Scope

- Change `run_hf_edgetam_streaming_realcase.py` default `--compile-mode` from
  `none` to `vision-reduce-overhead`.
- Keep `--compile-mode none` available for eager control / regression checks.
- Update tests that assert the CLI default.
- Update generated compile-ablation notes so they no longer say eager is the
  default.
- Do not change `edgetam-max`, `edgetam-hf-stream`, model checkpoints, or
  previously generated mask outputs.

## Validation

- Confirm CLI parsing defaults to `vision-reduce-overhead`.
- Run the HF realcase CLI help in `edgetam-hf-stream`.
- Run focused unittest coverage and `check_all.py`.

## Outcome

- Changed `run_hf_edgetam_streaming_realcase.py` so `--compile-mode` defaults
  to `vision-reduce-overhead`.
- Updated CLI help to state that `--compile-mode none` is the eager-control
  path.
- Updated tests to assert the new default.
- Updated environment and generated ablation docs to record the new default.
- Confirmed default parsing returns `vision-reduce-overhead`.
- Confirmed `edgetam-hf-stream` CLI help shows the new default wording.
- Ran a real-frame smoke without passing `--compile-mode`; it loaded
  `yonigozlan/EdgeTAM-hf`, printed `compile mode=vision-reduce-overhead
  targets=['vision_encoder']`, and completed on `sloth_set_2_motion_ffs` cam0
  frames `0..2`.
- Validation passed:
  - `conda run --no-capture-output -n SAM21-max python -m json.tool docs/generated/sloth_set_2_motion_ffs_hf_edgetam_streaming_compile_ablation.json`
  - `conda run --no-capture-output -n SAM21-max python -m unittest -v tests.test_sam21_checkpoint_ladder_panel_smoke`
  - `conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py`
