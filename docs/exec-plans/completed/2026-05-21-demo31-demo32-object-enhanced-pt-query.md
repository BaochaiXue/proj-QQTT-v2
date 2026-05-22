# Demo 3.1/3.2 Object Enhanced-PT Query + Render

## Goal

Use the existing object `enhanced-pt` postprocess for both rendered object PCD
and tracker query eligibility in Demo 3.1 and Demo 3.2, while keeping
controller filtering on `pt-filter` so two-hand controllers are not collapsed by
the largest-component prior.

## Scope

- Default object render/query path: `fixed-cap` + `enhanced-pt`.
- Default controller render/query path: `pt-filter`.
- Keep explicit `--object-point-control phystwin-volume` as a diagnostic
  override.
- Generalize the existing Demo 3.2 standard trackable-mask filter to Demo 3.1.
- Update contracts and focused tests.

## Outcome

- Demo 3.1 and Demo 3.2 contracts now report standard filter query masks by
  default:
  `trackable_mask_source=standard_filter_survivors`,
  `tracking_input_mask_semantics=standard_filter_trackable_masks`, and
  `tracker_query_source=union_trackable_mask`.
- Object render and query eligibility use `enhanced-pt` with
  `object_point_control=fixed-cap` by default.
- Controller render and query eligibility remain on `pt-filter`.
- `--trackable-mask-build-policy disabled` remains the diagnostic raw semantic
  mask fallback.

## Validation

- `python -m py_compile qqtt/demo/demo31_runtime.py qqtt/demo/demo3_runtime.py qqtt/demo/trackable_mask_filter.py qqtt/demo/services/profile_schema.py`
- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo31_dual_gpu_contract tests.test_demo32_trackable_mask_filter tests.test_demo3_cotracker_worker`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
- `git diff --check`
