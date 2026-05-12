# Demo 2.2 BatchTam TRT memory path final profile

## External BatchTam

- report: `/home/zhangxinjie/EdgeTAM-HF-batched/docs/generated/edgetam_trt_component_export_report.json`
- closed_loop_usable: `True`
- recommended_trt_scope: `memory_path_all`
- memory_attention_buckets: `16/16`
- demo22_trt_integration_allowed: `True`

## No-render BatchTam TRT

- pass: `False`
- profile: local diagnostic profile not committed (`docs/generated/demo22_full_batchtam_trt_memory_path_no_render_profile.json`)
- reason: process exited nonzero after RealSense capture failure; profile is diagnostic only, not a valid final run
- capture_group_fps: `9.909`
- raw_fusion_fps: `5.666`
- filter_fps: `5.666`
- render_fps: `0.000`
- edgetam p50/p90/p95 ms: `{'p50': 49.173571984283626, 'p90': 53.85562003357336, 'p95': 61.30280898651108}`
- ffs p50/p90/p95 ms: `{'p50': 59.08032902516425, 'p90': 66.90397398779169, 'p95': 77.34480203362182}`
- gpu_owner p50/p90/p95 ms: `{'p50': 108.77444199286401, 'p90': 122.67923401668668, 'p95': 136.41353597631678}`
- fallback used: `False`
- native depth fallback: `False`

## Pointcloud BatchTam TRT

- pass: `False`
- reason: not run because only 1 of 3 RealSense cameras is currently visible after the failed no-render run

## Comparison

| backend | profile | filter_fps | edgetam_p50_ms | ffs_p50_ms | gpu_owner_p50_ms | valid_final_fps |
| --- | --- | --- | --- | --- | --- | --- |
| hf_batch_vision_seq_session reduce-overhead | docs/generated/demo22_single_object_batchvision_stuffed_animal_no_render_profile.json | 8.326 | 55.740 | 56.818 | 112.946 | no |
| hf_batched_multisession torch pointcloud | docs/generated/demo22_full_hf_batched_multisession_stuffed_animal_pointcloud_profile.json | 8.102 | 59.666 | 57.982 | 118.390 | yes |
| BatchTam TRT no-render diagnostic | local diagnostic profile not committed | 5.666 | 49.174 | 59.080 | 108.774 | no |

## Final decision

- final_fps: `n/a`
- final_fps_source: `n/a`
- BatchTam partial no-render filter_fps: `5.666`
- BatchTam/Torch EdgeTAM p50 speedup: `1.189x`
- 15_fps_target_pass: `False`
- merge_ready: `False`
- blocker: Need all 3 RealSense cameras visible and a clean no-render + pointcloud full run; current WSL sees only 1 camera.
- next_optimization: After hardware is restored and profile is valid, evaluate default-stream TensorRT warning and overlap FFS/BatchTam if FPS remains below 15.
