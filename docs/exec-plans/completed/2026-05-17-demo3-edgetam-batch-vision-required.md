# Demo 3 EdgeTAM Batch Vision Encoder Requirement

## Goal

Make Demo 3 always use the shared HF EdgeTAM batch vision encoder path for live
mask generation, matching the intended Demo 2.2-style mask runtime behavior.

## Scope

- Force Demo 3 shared-runtime argv to include `--edgetam-batch-vision-encoder`.
- Surface the requirement in Demo 3 contract/profile metadata.
- Add deterministic tests that fail if Demo 3 stops passing the batch vision
  encoder flag to the shared runtime.
- Update Demo 3 docs.

## Non-Goals

- Do not change Demo 2.2 defaults.
- Do not introduce FFS into Demo 3.
- Do not change CoTracker query count or overlay behavior.
- Do not run hardware validation in CI.

## Validation

- Targeted Demo 3 contract tests.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`.

## Outcome

- Demo 3 shared-runtime argv now always includes
  `--edgetam-batch-vision-encoder`.
- Demo 3 dry-run contract and profile summaries expose
  `edgetam_batch_vision_encoder = true`.
- Demo 3 README and runtime contract document that HF EdgeTAM uses the shared
  batch vision encoder path.
- Added deterministic coverage that verifies the flag reaches the shared
  runtime adapter.
- Completed validation:
  - `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo3_contract`
  - `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
