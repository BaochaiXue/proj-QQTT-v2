# Harness Engineering Compact Index

This is the compressed entrypoint for generated harness engineering artifacts.
The historical files remain in `docs/generated/`; start here before opening
individual validation notes. Treat older generated reports as historical unless
they are linked below as current source-of-truth reports.

Machine-readable inventory:

- `docs/generated/harness_engineering_artifact_inventory.json`
- inventory scope: tracked files under `docs/generated/`; ignored local scratch
  outputs may exist but are not part of this source-controlled inventory
- last synchronized locally: 2026-05-11

## Current Operator Truth

| Area | Current status | Primary artifacts |
| --- | --- | --- |
| Scope boundary | The repo is camera-first, with sanctioned demo/proxy/tracking diagnostics. The formal data product still stops at recording, alignment, realtime native aligned export, and aligned native-vs-FFS comparison; demo masks/tracks must not become formal aligned-case outputs. | `../SCOPE.md`, `../ARCHITECTURE.md`, `../../AGENTS.md` |
| Demo 2.1.5 current quality profile | Latest full local `stuffed animal` + `towel` FFS quality profile is not realtime: point-cloud render reaches `2.50 FPS`; raw/filter/fusion are about `5.00 FPS`; complete group ratio is `0.368`; bottleneck is still upstream supply. The matching no-render replicate reaches `4.49 FPS` fused/filter output and is diagnostic only. | `demo215_full_local_ffs_fused_pcd_object_stuffed_animal_controller_towel_pointcloud_replicated_profile.md`, `demo215_full_local_ffs_fused_pcd_object_stuffed_animal_controller_towel_no_render_replicated_profile.md` |
| Demo 2.2 async filtered fused PCD | Current Demo 2.2 profiles still fail the `14.40 FPS` pass threshold. Batch-3 FFS profile reaches `5.37 FPS`, batch-vision profile reaches `5.25 FPS`, and parallel-init profile reaches `4.92 FPS`; bottleneck remains upstream supply. | `demo2_2_async_filter_batch3_ffs_20s_warmup_20s_formal_profile.md`, `demo2_2_async_filter_batchvision_40s_warmup_20s_formal_profile.md`, `demo2_2_async_filter_parallel_init_20s_warmup_20s_formal_profile.md` |
| Demo 2 single-camera FFS quality path | Local FFS + HF EdgeTAM compiled remains the single-camera quality reference on the 5090 laptop. Native RealSense depth remains fallback/debug for quality comparisons. | `demo2_local_ffs_professor_speed_polish.md`, `demo2_depth_source_fps_comparison.md`, `demo2_realtime_masked_edgetam_pcd_profile_isolation.md` |
| Demo v0.3 remote FFS | Active remote FFS track. The 100-kit IR triplet replay dataset is prepared on the 5090 with 20 warmup kits and 100 measured kits at 15 kit-FPS. 4090 transfer/profile and remote matrix are still pending, so do not claim the staged remote pipeline has passed. | `demo_v03_100kit_replay_validation.md`, `docs/exec-plans/active/2026-05-08-demo-v0-3-100kit-staged-remote-ffs.md` |
| Remote FFS | Full-frame `848x480` remote FFS over low-throughput relay was not realtime. Direct LAN/Tailscale and 4090 real-depth reports are diagnostics; formal Demo 2 quality still requires FFS-derived depth as the main output. | `demo2_remote_ffs_service_validation.md`, `demo2_remote_ffs_4090_server_5090_client_validation.md`, `demo2_remote_ffs_4090_real_depth_server.md` |
| Demo 3 tracking diagnostics | CoTracker3 online is available; ONNX Runtime CUDA and TensorRT providers are available. LocoTrack, TAPIR, TAPNext, NVOFA, and VPI LK are unavailable or not configured in the current local stack. The ONNX/TRT probe validates provider plumbing only, not model exportability. | `demo3_tracking_backend_stack.md`, `demo3_onnx_trt_probe.md`, `demo3_cotracker3_tracking_benchmark/both_30_still_object_round1_20260428/profile.md` |
| HF EdgeTAM streaming | HF EdgeTAM frame-by-frame streaming is validated on real QQTT frames. Current mitigation work treats GPU utilization as diagnostic; p50/p90 latency and end-to-end p90 remain the primary KPIs. | `hf_edgetam_streaming_processor_session_validation.md`, `demo215_hf_edgetam_gpu_underutilization_mitigation.md` |
| FFS contract | Current formal quality contract is `20-30-48`, `valid_iters=4`, `848x480 -> 864x480`, TensorRT `builderOptimizationLevel=5`. | `ffs_benchmark_tradeoff_validation.md`, `ffs_live_trt_viewer_validation.md`, `ffs_depth_backend_integration_validation.md` |
| SAM3.1 live init | Formal live demos use SAM3.1 first-frame initialization. Demo 2 and Demo 2.1 now use image one-frame segmentation for live init, not one-frame video propagation. | `demo2_1_shared_ffs_per_camera_edgetam_validation.md`, `sam31_env_validation.md` |
| Object/controller PCD filtering | Object PCD uses enhanced-PT. Controller/hand PCD uses ordinary PT filter; do not union object/controller before filtering. | `sloth_set_2_motion_ffs_hf_edgetam_object_enhanced_controller_pt_filter_pcd_benchmark.md`, `demo2_1_shared_ffs_per_camera_edgetam_validation.md` |
| Repo guards | Scope, catalog, visual architecture, experiment-boundary checks, demo/proxy/tracking smoke tests, and generated script path checks are active in quick `check_all.py`. Run them from `demo_2_max`, not bare base Python. | `repo_hardening_validation.md`, `contract_hardening_validation.md` |

## Inventory Summary

Tracked generated artifact inventory at the time of this refresh: `390` files.
See the machine-readable inventory for byte counts. Local ignored scratch
outputs are excluded.

| Category | Files | Purpose |
| --- | ---: | --- |
| `index` | 3 | This README, the compact index, and the machine-readable inventory. |
| `demo2_1_three_view_runtime` | 108 | Demo 2.1 three-camera fused-PCD profiles, prompts, logs, and validation notes. |
| `demo2_1_5_runtime` | 34 | Demo 2.1.5 async filtered fused-PCD, GPU, and mitigation profiles. |
| `demo2_2_runtime` | 27 | Demo 2.2 async filtered fused-PCD, batch-3 FFS, batch-vision, and runtime split reports. |
| `demo2_single_camera_runtime` | 25 | Single-camera Demo 2 local/remote FFS profiles and commands. |
| `demo3_tracking` | 50 | Demo 3 tracking backend stack, benchmark, and lifted-PCD overlay artifacts. |
| `remote_ffs_demo_runtime` | 25 | Demo v0.2/v0.3 remote FFS replay, proxy, 4090, and LAN benchmark notes. |
| `sam_edgetam_mask_benchmarks` | 62 | SAM2.1/SAM3.1, EdgeTAM mask, PCD, hand/controller, and compile experiments. |
| `edgetam_integration` | 16 | EdgeTAM environment, ONNX/TRT, and streaming proof outputs. |
| `ffs_validation` | 17 | FFS environment, TensorRT, static replay, live viewer, and backend validation. |
| `hardware_capture_wsl` | 7 | D455, WSL, firmware, and capture probe outputs. |
| `repo_contract_retention` | 7 | Cleanup, retention, scope, and contract hardening notes. |
| `visualization_stack` | 3 | Visualization ownership and cleanup validation. |
| `supporting_assets_and_raw_outputs` | 6 | Small JSON/raw helper outputs not covered by the main categories. |

The exact file list and byte counts are in
`harness_engineering_artifact_inventory.json`.

## How To Add New Generated Artifacts

- Prefer extending the validation file for an existing theme over creating a
  new near-duplicate report.
- Keep short terminal logs as raw `.txt`/`.log` only when they support a
  linked validation note.
- Put current claims in this compact index when they supersede older reports.
- Do not delete historical artifacts unless a separate cleanup pass records
  what was removed and why.

## Fast Lookup

- Latest Demo 2.1.5 full local FFS fused-PCD profile:
  `demo215_full_local_ffs_fused_pcd_object_stuffed_animal_controller_towel_pointcloud_replicated_profile.md`
- Latest Demo 2.2 batch-3 FFS profile:
  `demo2_2_async_filter_batch3_ffs_20s_warmup_20s_formal_profile.md`
- Demo v0.3 remote FFS transfer/profile status:
  `demo_v03_100kit_replay_validation.md`
- Demo 3 tracking backend status:
  `demo3_tracking_backend_stack.md`
- Harness engineering map:
  `scripts/harness/README.md`
- Deterministic validation:
  `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
