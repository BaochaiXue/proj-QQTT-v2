# Demo 3.2 Shape Prior Mask-Depth Start

## Goal

Move Demo 3.2 shape-prior warmup submission earlier so the SAM3D request starts
as soon as the first valid object-mask plus color-aligned depth snapshot is
available, without waiting for TAPNext++ to publish the first strict render pair.

## Implementation Notes

- Add `async-after-first-mask-depth-pair` to the shape-prior start-policy
  contract and make it the Demo 3.2/default delegate policy.
- Keep `async-after-first-strict-pair`, `blocking-before-first-output`, and
  `after-teardown` as explicit policies.
- Submit the new policy directly from completed `PcdBuildResult` paths:
  lossless PCD worker, non-lossless PCD worker, and strict paired worker.
- Keep strict-pair publication responsible for the old strict-pair policy and
  for attaching any ready shape-prior layer to render packets.
- Do not change shape-prior payload generation, SAM3D/upscale behavior,
  alignment, sampling, PCD filtering, or table-Z filtering.

## Validation

- Add red/green tests for parser/default/delegate behavior and early submission
  before tracker pairing.
- Run:
  `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo32_shape_prior_warmup tests.test_single_demo_v3_runtime tests.test_demo32_side_by_side_panel`
- Run:
  `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
