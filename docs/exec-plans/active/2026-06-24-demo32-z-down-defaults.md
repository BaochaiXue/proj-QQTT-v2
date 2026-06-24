# Demo 3.2 Z-Down Defaults

## Goal

Preserve compatibility with existing Demo 3.2 table-world artifacts by keeping
the workspace side of `table_world_z0` on negative Z whenever a component needs a
default table-Z direction.

## Plan

1. Add failing tests for Shape Prior snapshot, remote protocol fallback,
   remote worker alignment config, and single-view alignment config defaults.
2. Change only defaults/fallbacks from positive to negative while preserving
   explicit `positive` support for diagnostics.
3. Update docs to state that missing table-Z metadata falls back to negative Z.
4. Run focused shape-prior/table-Z tests and smoke validation.

## Validation

- RED: `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo32_shape_prior_warmup.ShapePriorProtocolAndSnapshotTest.test_snapshot_defaults_to_negative_table_z_direction tests.test_demo32_shape_prior_warmup.ShapePriorProtocolAndSnapshotTest.test_protocol_fallback_defaults_to_negative_table_z_direction tests.test_demo32_shape_prior_warmup.ShapePriorWorkerSam3DInputTest.test_worker_alignment_config_defaults_to_negative_table_z_direction tests.test_demo32_shape_prior_warmup.SingleViewShapeAlignmentTest.test_alignment_config_defaults_to_negative_table_z_direction`
- PASS: `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo32_shape_prior_warmup`
- PASS: `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo32_shape_prior_warmup tests.test_single_demo_v3_runtime tests.test_single_demo_tapnextpp_overlay tests.test_demo32_headless_render_helper`
- PASS: `python -m py_compile qqtt/demo/shape_prior_warmup.py qqtt/demo/single_view_shape_align.py services/shape_prior_remote/protocol.py services/shape_prior_remote/server.py tests/test_demo32_shape_prior_warmup.py`
- PASS: `git diff --check`
- PASS: `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
