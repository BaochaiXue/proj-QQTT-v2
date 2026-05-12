# Demo 2.2 full hf_batched_multisession final profile

## External EdgeTAM backend
- external_repo: /home/zhangxinjie/EdgeTAM-HF-batched
- external_github_repo: https://github.com/BaochaiXue/transformers/tree/feat/edgetam-batched-multisession-runtime
- external_branch: feat/edgetam-batched-multisession-runtime
- external_commit: 4a41c4a45fa6f0541f3d62aff95867de06d90c92
- external_report: /home/zhangxinjie/EdgeTAM-HF-batched/docs/generated/edgetam_batched_multisession_final_report.json
- external_report_source_commit: 5a2a86567e6a42ddf1178b6660dc51c1f9509972
- backend: hf_batched_multisession
- precision_mode: memory_path_fp32
- compile_mode: reduce-overhead
- contract_pass: True

## Strict backend contract
- batch_memory_attention: True
- batch_mask_decoder: True
- batch_memory_encoder: True
- batched_state_scatter: True
- used_public_session_step_in_hot_path: False
- partial_fallback_used: False
- edgetam_backend_fallback_used: False
- contract pass: True
- fallback used: False

## No-render profile
- profile file: docs/generated/demo22_full_hf_batched_multisession_stuffed_animal_no_render_profile.json
- capture_group_fps: 13.814
- raw_fusion_fps: 8.253
- filter_fps: 8.253
- render_fps: 0.000
- edgetam_stage_wall p50/p90/p95 ms: 58.465/64.294/66.649
- ffs_cycle p50/p90/p95 ms: 56.857/61.017/64.472
- gpu_owner_total p50/p90/p95 ms: 115.968/124.104/129.342
- object_points min/median/p90: {'min': 10800, 'median': 10922, 'p90': 10976}
- filtered_points min/median/p90: {'min': 10800, 'median': 10922, 'p90': 10976}
- pass/fail: pass
- reason: pass

## Pointcloud profile
- profile file: docs/generated/demo22_full_hf_batched_multisession_stuffed_animal_pointcloud_profile.json
- capture_group_fps: 14.129
- raw_fusion_fps: 8.102
- filter_fps: 8.102
- render_fps: 4.114
- edgetam_stage_wall p50/p90/p95 ms: 59.666/66.479/70.027
- ffs_cycle p50/p90/p95 ms: 57.982/63.001/69.278
- gpu_owner_total p50/p90/p95 ms: 118.390/127.471/133.437
- visible pointcloud: True
- object_points min/median/p90: {'min': 10775, 'median': 10958, 'p90': 11017}
- pass/fail: pass
- reason: pass

## Final decision
- final_fps: 8.102
- final_fps_source: docs/generated/demo22_full_hf_batched_multisession_stuffed_animal_pointcloud_profile.json
- final_fps_metric: filter_fps
- 15_fps_target_pass: False
- bottleneck: single-owner GPU critical path: FFS cycle + full batched EdgeTAM + fusion/filter; gpu_owner_total p50 is about 118.39 ms
- merge_ready: False
- integration_ready_but_perf_below_15fps: True

Note: these runs used `--no-parallel-init` so SAM3.1 first-frame initialization is isolated from EdgeTAM/Inductor import and compile startup.
