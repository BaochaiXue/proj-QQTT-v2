# Results Retention Cleanup

- Date: `2026-05-01`
- Scope: local result folders under `data/experiments/` and `data/ffs_benchmarks/`

## Deleted Result Folders

Smoke or preview outputs removed from `data/experiments/`:

- `_smoke_single_config`
- `still_object_rope_frame0_cam0_orbit_3x4_mask_erode_sweep_highlight_gif_ffs203048_iter4_trt_level5_smoke`
- `still_object_rope_frame0_cam0_orbit_6x2_enhanced_pt_like_gif_ffs203048_iter4_trt_level5_smoke`
- `still_object_rope_frame0_cam0_orbit_6x2_gif_ffs203048_iter4_trt_level5_smoke`
- `still_object_rope_frame0_cam0_orbit_6x2_mask_erode_sweep_gif_ffs203048_iter4_trt_level5_smoke`
- `still_object_rope_frame0_cam0_orbit_6x2_mask_erode_sweep_highlight_gif_ffs203048_iter4_trt_level5_smoke`
- `still_object_round1_frame0_cam0_orbit_gif_ffs203048_iter4_trt_level5_header_smoke`
- `still_object_round1_frame0_cam0_orbit_gif_ffs203048_iter4_trt_level5_header_smoke2`
- `still_object_round1_frame0_cam0_orbit_gif_ffs203048_iter4_trt_level5_rgb_smoke`
- `still_object_round1_frame0_cam0_orbit_gif_ffs203048_iter4_trt_level5_smoke`
- `still_object_round1_frame0_cam0_orbit_gif_ffs203048_iter4_trt_level5_smoke_crop`
- `still_object_rope_frame0_cam0_orbit_6x2_enhanced_pt_like_gif_ffs203048_iter4_trt_level5_preview`
- `still_object_rope_frame0_cam0_orbit_6x2_gif_ffs203048_iter4_trt_level5_preview`

Interrupted or debug-only local outputs removed:

- `ffs_trt_debug_23_export_trace`
- `sam31_still_object_view_benchmark_both_30_still_object_round1_20260428_qqtt_ffs_compat`
- `sam31_timing_static_round7_cam0_human_hand_object`
- `logs/generate_sam31_round4_20260427_154340.log`
- `logs/latest_sam31_timing_log.txt`
- `data/ffs_benchmarks/viewer_run_2026-04-19*.log`

## Archived / Renamed

Moved obsolete but potentially useful roots under `data/experiments/_archived_obsolete/`:

- `ffs_static_replay_matrix_20260422_fullrun` ->
  `ffs_static_replay_matrix_20260422_sequential_obsolete_fullrun`
- `ffs_trt_static_rounds_848x480_pad864_rtx5090_laptop_20260428` ->
  `ffs_trt_static_rounds_848x480_pad864_pre_builderopt5_obsolete_rtx5090_laptop_20260428`
- `realtime_orbit_wsl_diagnosis_20260429` ->
  `realtime_orbit_wsl_diagnosis_debug_20260429`

Moved invalid-for-QQTT TensorRT control roots under
`data/experiments/_archived_invalid_for_qqtt/`:

- `ffs_official_table_trt_rtx5090_laptop_20260428` ->
  `ffs_official_table_trt_640x480_random_invalid_for_qqtt_20260428`
- `ffs_official_table_trt_static_rounds_rtx5090_laptop_20260428` ->
  `ffs_official_table_trt_static_rounds_640x480_resized_invalid_for_qqtt_20260428`

Moved saved-pair offline PyTorch screening roots under
`data/ffs_benchmarks/_archived_saved_pair_offline/`:

- `2026-04-19_tradeoff_baseline`
- `2026-04-19_tradeoff_multiscale`
- `2026-04-19_tradeoff_extreme`
- `2026-04-19_rerun_baseline`
- `2026-04-19_rerun_multiscale`
- `2026-04-19_rerun_extreme`
- `2026-04-19_rerun_focus`

## Current Top-Level Results To Prefer

- `data/experiments/ffs_trt_static_rounds_848x480_pad864_builderopt5_rtx5090_laptop_20260428`
- `data/experiments/ffs_static_replay_matrix_concurrent3view_20260422_fullrun`
- `data/experiments/sam21_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5_stable_throughput`
- `data/ffs_benchmarks/live_3cam_scale*.log`

Reporting rule: use archived roots only for historical context or artifact reuse. Do not cite archived smoke, invalid QQTT controls, or obsolete sequential static replay as current performance results.
