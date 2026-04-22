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

`check_all.py` now has two deterministic profiles:

- default `python scripts/harness/check_all.py`
  - fast quick profile
  - intended to finish in under one minute on a healthy repo environment
- explicit `python scripts/harness/check_all.py --full`
  - broader legacy validation surface
  - use for larger refactors or when you need the old wide regression net

### Hardware / External Proof-of-Life

- `verify_ffs_demo.py`
- `verify_ffs_tensorrt_windows.py`
- `verify_ffs_tensorrt_wsl.py`
- `probe_d455_ir_pair.py`
- `probe_d455_stream_capability.py`
- `render_d455_stream_probe_report.py`
- `run_ffs_on_saved_pair.py`
- `reproject_ffs_to_color.py`
- `generate_sam31_masks.py`

### Data Cleanup

- `cleanup_different_types_cases.py`

This script is for downstream-facing final-case cleanup under `data/different_types/` and is intentionally separate from the repo-internal aligned-case visualization workflows.
It preserves the canonical `color/` and `depth/` frame trees plus optional `color/0.mp4`, `1.mp4`, and `2.mp4` RGB sidecars when present. Execute mode also backfills those color mp4 sidecars from `color/<camera>/*.png` if they are missing.

### Current User-Facing Compare CLIs

- `visual_compare_depth_panels.py`
- `visual_compare_reprojection.py`
- `visual_compare_depth_video.py`
- `visual_compare_depth_triplet_ply.py`
- `visual_compare_depth_triplet_video.py`
- `visual_compare_masked_pointcloud.py`
- `visual_compare_masked_camera_views.py`
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

### External Sidecars

- `generate_sam31_masks.py`

This is a workspace-local helper for running external `SAM 3.1` segmentation against QQTT case assets.
It is intentionally kept out of `data_process/segment*.py` because the repo scope guard bans reintroducing that PhysTwin-era file surface.
Use it only as an operator-side sidecar:

- it reads `color/<camera>.mp4` when present, otherwise `color/<camera>/*.png`
- it writes mask artifacts under a caller-selected output directory, defaulting to `<case_root>/sam31_masks/`
- it expects `sam3` to be installed in the active environment and Hugging Face auth/checkpoint access to be handled outside the repo
- checkpoints remain external and should be passed by path or resolved from Hugging Face cache via login / `QQTT_SAM31_CHECKPOINT`
- for current PyPI `sam3`, keep `bpe_simple_vocab_16e6.txt.gz` external as well, preferably next to the checkpoint, or set `QQTT_SAM31_BPE_PATH`

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
