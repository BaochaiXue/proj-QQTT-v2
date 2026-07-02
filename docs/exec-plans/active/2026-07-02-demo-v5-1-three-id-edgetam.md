# Demo v5.1 Three-ID EdgeTAM

## Requirement

Demo v5.1 must always run one HF EdgeTAM streaming session with separate
`hand_a`, `object`, and `hand_b` object ids when controller/object tracking is
active. `controller_mask` remains the union of the two hands. This is not a
runtime mode and must not be controlled by `--controller-instance-mode`.

## Plan

- [x] Remove the `--controller-instance-mode` behavior switch from
      `demo_v5_1/main_data_processing.py`.
- [x] Make controller tracking intrinsically use `hand_a` and `hand_b` ids.
- [x] Keep object-only and track-mode none behavior unchanged.
- [x] Update the outer runner contract/tests so the removed flag is rejected
      and the default identities are `hand_a`, `object`, `hand_b`.
- [x] Document the one-session, three-id EdgeTAM contract in
      `demo_v5_1/design_spec.md`.
- [x] Run focused tests and smoke validation.

## Validation

- Passed:
  `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_demo_v5_1_default_config.py tests/test_demo_v5_1_shape_prior_simplification.py -q`
- Passed:
  `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
