# Demo 3.1 TAPNext++ 4096/View Live Rendered Profile

Date: 2026-05-20

Backend: `tapnextpp`
Query budget: `4096` per camera, `12288` total across three views
Render mode: `pointcloud`
Overlay display scope: `controller`
Visualization mode: `3d-surface-markers`
Tracking render gate: `--wait-for-tracking-overlay`
Duration: `120s`
Warmup excluded by shared runtime profile: `20s`

The TAPNext++ adapter was corrected before these runs:

- RGB frames are normalized to the official TAPNext++ `float32 [-1, 1]` range.
- PyTorch TAPNext++ raw tracks are parsed as `yx`, then scaled back to QQTT original-frame `yx`.

| execution_mode | rendered_fps_after_warmup | rendered_groups_after_warmup | tracker_publish_fps | tracker_model_ms_p50 | tracker_model_ms_p95 | tracker_e2e_ms_p50 | tracker_e2e_ms_p95 | gpu0_mem_used_mb_p50 | gpu1_mem_used_mb_p50 | gpu0_util_p50 | gpu1_util_p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| serial | 6.373 | 719 | 6.373 | 147.616 | 150.385 | 151.902 | 156.290 | 2976.188 | 8264.563 | 41.0 | 94.0 |
| batch-views | 5.730 | 647 | 5.729 | 168.219 | 170.341 | 170.656 | 173.769 | 2978.188 | 8519.750 | 42.0 | 95.0 |

Validation counters:

| execution_mode | query_count_by_camera | batch_updates | serial_group_updates | batch_errors | first_render_group | warmup_skipped | render_blocked | exact_render_packets | missing_render_packets | missing_lift_inputs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| serial | `{0: 4096, 1: 4096, 2: 4096}` | 0 | 722 | 0 | 0 | 0 | 0 | 722 | 0 | 0 |
| batch-views | `{0: 4096, 1: 4096, 2: 4096}` | 649 | 0 | 0 | 0 | 0 | 0 | 649 | 0 | 0 |

Raw local profile files:

- `docs/generated/demo31_tapnextpp_rendered_profile/serial_q4096_live_fixed_yx_120s.json`
- `docs/generated/demo31_tapnextpp_rendered_profile/serial_q4096_live_fixed_yx_120s_shared_runtime.json`
- `docs/generated/demo31_tapnextpp_rendered_profile/batch_views_q4096_live_fixed_yx_120s.json`
- `docs/generated/demo31_tapnextpp_rendered_profile/batch_views_q4096_live_fixed_yx_120s_shared_runtime.json`

Both live runs reported one RealSense timeout/restart from serial `239222300412`.
