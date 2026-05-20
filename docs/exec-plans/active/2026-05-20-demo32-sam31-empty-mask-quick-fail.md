# Demo 3.2 SAM3.1 Empty-Mask Quick Fail

## Goal

Fail early during Demo 3.2 warmup when SAM3.1 first-frame initialization does
not produce usable object and controller masks. The runtime should not continue
into EdgeTAM sessions, FFS fusion, LiteTracker query-init, or render waiting
when either semantic target is missing.

## Scope

- Add a shared runtime SAM3.1 mask validation gate immediately after live
  first-frame masks are resolved.
- Default to requiring non-empty SAM3.1 object/controller masks for live demos.
- Surface the gate in Demo 3.1 / Demo 3.2 contracts and dry-run output.
- Add focused deterministic tests.

## Non-Goals

- Do not change SAM3.1 prompting or segmentation quality.
- Do not retry more than the existing `--sam31-init-max-attempts` policy.
- Do not affect saved-mask debug mode beyond reporting contract defaults.

## Validation

- Focused unit tests for shared runtime validation and Demo 3.2 contract.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
