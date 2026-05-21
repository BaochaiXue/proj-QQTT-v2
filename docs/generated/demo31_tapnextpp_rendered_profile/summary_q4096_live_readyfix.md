# Demo 3.1 TAPNext++ 4096/View Ready Timing Fix

Date: 2026-05-20

Backend: `tapnextpp`
Query budget: `4096` per camera, `12288` total across three views
Render mode: `pointcloud`
Tracking render gate: `--wait-for-tracking-overlay`
Duration: `45s`

The warmup timing bug was that the runtime could leave child-process status
events in the queue until a late snapshot or teardown. That made
`ready_receive_s` look like the tracker became ready after the whole run, even
though the child had emitted `ready_perf_s` much earlier.

The runtime now records:

- `ready_event_after_process_start_s`: when the child emitted ready.
- `ready_receive_after_process_start_s`: when the runtime drained that event.
- `ready_queue_lag_ms`: queue delay between those two timestamps.

| execution_mode | tracker_ready_event_s | tracker_ready_receive_s | ready_queue_lag_ms | tracker_total_init_ms | first_complete_inference_s | first_render_s | camera_startup_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| serial | 7.157 | 7.188 | 30.508 | 7051.569 | 19.678 | 20.074 | 10944.046 |
| batch-views | 3.922 | 4.086 | 163.634 | 3811.322 | 19.784 | 20.373 | 10973.201 |

Rendered live results after the timing fix:

| execution_mode | rendered_fps_after_warmup | rendered_groups_after_warmup | fusion_fps_after_warmup | tracker_model_ms_p50 | tracker_model_ms_p95 | tracker_e2e_ms_p50 | tracker_e2e_ms_p95 | exact_render_packets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| serial | 6.833 | 260 | 14.847 | 136.201 | 138.524 | 140.695 | 143.886 | 262 |
| batch-views | 6.413 | 243 | 15.465 | 147.981 | 149.364 | 150.760 | 153.707 | 244 |

Validation counters:

| execution_mode | query_count_by_camera | first_render_group | warmup_skipped | render_blocked |
| --- | --- | ---: | ---: | ---: |
| serial | `{0: 4096, 1: 4096, 2: 4096}` | 0 | 0 | 0 |
| batch-views | `{0: 4096, 1: 4096, 2: 4096}` | 0 | 0 | 0 |

Raw local profile files:

- `docs/generated/demo31_tapnextpp_rendered_profile/serial_q4096_live_readyfix_45s.json`
- `docs/generated/demo31_tapnextpp_rendered_profile/serial_q4096_live_readyfix_45s_shared_runtime.json`
- `docs/generated/demo31_tapnextpp_rendered_profile/batch_views_q4096_live_readyfix_45s.json`
- `docs/generated/demo31_tapnextpp_rendered_profile/batch_views_q4096_live_readyfix_45s_shared_runtime.json`

Both live runs reported one RealSense timeout/restart from serial
`239222300412`; the tracker ready timestamps were still in the expected
single-digit-second range.
