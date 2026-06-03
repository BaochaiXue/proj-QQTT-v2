# Demo 3.3 Overlay Substage Profile

## Goal

Add fine-grained timing to the Demo 3.1/3.3 tracker overlay render path so the
existing broad `overlay_ms` cost can be attributed without changing tracking,
shape-prior, render, or point-count behavior.

## Scope

- Add low-overhead per-render timing fields:
  - `surface_snap_ms`
  - `lift_ms`
  - `semantic_color_ms`
  - `bbox_filter_ms`
  - `overlay_concat_ms`
  - `control_marker_expand_ms`
- Keep the existing `overlay_ms` total timing and overlay semantics unchanged.
- Add focused contract tests that the fields exist and are populated.
- Run Demo 3.3 live under `demo_3_3_max` and inspect the generated profile.

## Follow-Up Attribution Fields

The first live run showed the requested six fields were all small while
`overlay_ms` remained large, so the profile was extended with:

- `tracker_result_take_ms`
- `overlay_processing_ms`
- `overlay_unattributed_ms`
- `render_packet_match_ms`
- `bbox_reference_ms`
- `control_point_select_ms`
- `frame_provenance_ms`
- `render_packet_replace_ms`

These are still observation-only timings. They preserve the existing
`overlay_ms` contract and do not change render output or tracking quality.

## Verification Evidence

- `conda run --no-capture-output -n demo_3_3_max python -m py_compile qqtt/demo/demo31_runtime.py qqtt/demo/services/profile_schema.py tests/test_demo31_dual_gpu_contract.py`
- `conda run --no-capture-output -n demo_2_max python -m pytest tests/test_demo31_dual_gpu_contract.py::Demo31DualGpuContractTest::test_surface_marker_mode_snaps_to_surface_without_legacy_lift tests/test_demo31_dual_gpu_contract.py::Demo31DualGpuContractTest::test_all_tracks_anchor_mode_disables_surface_mask_and_bbox_gates tests/test_demo31_dual_gpu_contract.py::Demo31DualGpuContractTest::test_renderer_colors_tracked_object_and_controller_points_by_semantic_label -q`
- `conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py`
- Live Demo 3.3 rendered profile:
  `docs/generated/demo33_overlay_substage_profile_20260603_60s_v3_shared_runtime.json`

## Current Finding

For the v3 60s rendered live run, the broad `overlay_ms` p50 was about
233 ms. The large contributors were `control_point_select_ms` p50 about
138 ms and `render_packet_match_ms` p50 about 91 ms. The originally requested
color/lift/concat/marker fields were small, and `overlay_unattributed_ms` p50
was reduced to about 0.5 ms after the expanded attribution fields.

## Non-Goals

- Do not reduce query count or displayed point count.
- Do not change object/controller coloring semantics.
- Do not skip shape prior.
- Do not change renderer layer mode or tracker backend.
