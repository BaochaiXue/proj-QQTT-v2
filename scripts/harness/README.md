# Harness Script Map

This folder is intentionally limited to thin command-line wrappers, probe/check utilities, and a small number of bounded one-off diagnostics.

Rule of thumb:

- keep orchestration, rendering, and analysis logic in `data_process/visualization/` or `data_process/depth_backends/`
- keep `scripts/harness/` as the user-facing CLI shell around that logic
- keep RealSense / environment checks here because they are operational harness utilities, not production library code

## Keep Here

### Checks / Guards

- `check_all.py`
- `check_scope.py`
- `check_visual_architecture.py`

### Hardware / External Proof-of-Life

- `verify_ffs_demo.py`
- `probe_d455_ir_pair.py`
- `probe_d455_stream_capability.py`
- `render_d455_stream_probe_report.py`
- `run_ffs_on_saved_pair.py`
- `reproject_ffs_to_color.py`

### Current User-Facing Compare CLIs

- `visual_compare_depth_panels.py`
- `visual_compare_reprojection.py`
- `visual_compare_depth_video.py`
- `visual_compare_depth_triplet_ply.py`
- `visual_compare_turntable.py`
- `visual_compare_rerun.py`
- `visual_make_match_board.py`
- `visual_make_professor_triptych.py`

These should stay thin wrappers around workflow modules under `data_process/visualization/workflows/`.

### Focused Diagnostics

- `audit_ffs_left_right.py`
- `compare_face_smoothness.py`
- `visual_compare_stereo_order_pcd.py`

These remain in harness because they are bounded, investigation-oriented tools rather than core product workflows.

## Do Not Grow Here

- large shared rendering/math helpers
- reusable point-cloud IO or crop logic
- reusable layout builders
- reusable calibration / view-planning logic

Those belong under `data_process/visualization/` and should be imported by these wrappers.

## Current Cleanup Result

- thin wrappers remain in `scripts/harness/`
- old cache/junk directories have been removed
- legacy duplicate generated-doc references have been collapsed onto the newer cleanup docs under `docs/generated/`
