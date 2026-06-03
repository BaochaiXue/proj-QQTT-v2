# Demo 3.3 Overlay Substage Profile v3

Source profile:
`docs/generated/demo33_overlay_substage_profile_20260603_60s_v3_shared_runtime.json`

Live run:

- Environment: `demo_3_3_max`
- Duration: 60s rendered live run
- Render mode: `pointcloud`
- Default overlay display scope: `union`
- Render FPS after warmup: `4.03`
- Rendered groups after warmup: `156`
- Note: one RealSense frame timeout/restart occurred during the run.

Overlay p50/p95:

| field | p50 ms | p95 ms |
| --- | ---: | ---: |
| `overlay_ms` | `233.31` | `374.47` |
| `tracker_result_take_ms` | `0.31` | `0.62` |
| `overlay_processing_ms` | `233.00` | `374.20` |
| `render_packet_match_ms` | `90.56` | `236.21` |
| `control_point_select_ms` | `137.93` | `152.23` |
| `lift_ms` | `1.94` | `3.70` |
| `semantic_color_ms` | `0.13` | `0.24` |
| `overlay_concat_ms` | `0.11` | `0.48` |
| `control_marker_expand_ms` | `0.54` | `2.09` |
| `surface_snap_ms` | `0.00` | `0.00` |
| `bbox_filter_ms` | `0.00` | `0.00` |
| `bbox_reference_ms` | `0.04` | `0.21` |
| `frame_provenance_ms` | `0.01` | `0.02` |
| `render_packet_replace_ms` | `0.02` | `0.03` |
| `overlay_unattributed_ms` | `0.54` | `1.09` |

Finding:

The object/controller semantic coloring change is not the runtime bottleneck.
The expensive pieces in the current all-tracks 3D overlay path are
`control_point_select_ms` and `render_packet_match_ms`. The requested
color/lift/concat/marker expansion substages are small relative to the broad
`overlay_ms`.
