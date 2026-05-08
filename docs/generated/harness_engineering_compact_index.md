# Harness Engineering Compact Index

This is the compressed entrypoint for generated harness engineering artifacts.
The historical files remain in `docs/generated/`; start here before opening
individual validation notes.

Machine-readable inventory:

- `docs/generated/harness_engineering_artifact_inventory.json`

## Current Operator Truth

| Area | Current status | Primary artifacts |
| --- | --- | --- |
| Demo 2.1 three-view fused PCD | Live SAM3.1 image one-frame init works for object-only. `visual-5fps` currently reaches `3.06 FPS` after warmup in the latest object-only profile; bottleneck is upstream mask/depth supply, not Open3D. Capture groups now have a temporal-coherence gate before FFS. | `demo2_1_shared_ffs_per_camera_edgetam_validation.md`, `demo2_1_visual5fps_image_sam31_profile_object_only_120s.md` |
| Demo 2 single-camera FFS quality path | Local FFS + HF EdgeTAM compiled remains the official quality path on the 5090 laptop. Native RealSense depth is fallback/debug, not final demo output. | `demo2_local_ffs_professor_speed_polish.md`, `demo2_depth_source_fps_comparison.md`, `demo2_realtime_masked_edgetam_pcd_profile_isolation.md` |
| Demo v0.3 remote FFS | Active remote FFS track. Use a fixed 100-kit IR triplet replay folder, 20 warmup kits, and 100 measured kits at 15 kit-FPS. Demo v0.2 is legacy/source-data reference only. | `demo_v03_100kit_replay_validation.md`, `docs/exec-plans/active/2026-05-08-demo-v0-3-100kit-staged-remote-ffs.md` |
| Remote FFS | Full-frame `848x480` remote FFS over low-throughput relay was not realtime. Direct Tailscale testing exists, but formal Demo 2 quality still requires FFS-derived depth as the main output. | `demo2_remote_ffs_service_validation.md` |
| HF EdgeTAM streaming | HF EdgeTAM frame-by-frame streaming is validated on real QQTT frames. Default compile mode for current live/demo paths is `vision-reduce-overhead`. | `hf_edgetam_streaming_processor_session_validation.md`, `sloth_set_2_motion_ffs_hf_edgetam_streaming_compile_ablation.md` |
| FFS contract | Current formal quality contract is `20-30-48`, `valid_iters=4`, `848x480 -> 864x480`, TensorRT `builderOptimizationLevel=5`. | `ffs_benchmark_tradeoff_validation.md`, `ffs_live_trt_viewer_validation.md`, `ffs_depth_backend_integration_validation.md` |
| SAM3.1 live init | Formal live demos use SAM3.1 first-frame initialization. Demo 2 and Demo 2.1 now use image one-frame segmentation for live init, not one-frame video propagation. | `demo2_1_shared_ffs_per_camera_edgetam_validation.md`, `sam31_env_validation.md` |
| Object/controller PCD filtering | Object PCD uses enhanced-PT. Controller/hand PCD uses ordinary PT filter; do not union object/controller before filtering. | `sloth_set_2_motion_ffs_hf_edgetam_object_enhanced_controller_pt_filter_pcd_benchmark.md`, `demo2_1_shared_ffs_per_camera_edgetam_validation.md` |
| Repo guards | Scope, catalog, visual architecture, and experiment-boundary checks are active and part of `check_all.py`. | `repo_hardening_validation.md`, `contract_hardening_validation.md` |

## Inventory Summary

Generated artifact inventory at the time of compression: `165` files.

| Category | Files | Purpose |
| --- | ---: | --- |
| `index` | 3 | This README, the compact index, and the machine-readable inventory. |
| `demo2_1_three_view_runtime` | 25 | Three-camera fused PCD runtime profiles, logs, and validation notes. |
| `demo2_single_camera_runtime` | 18 | Single-camera Demo 2 local/remote FFS profiles and commands. |
| `sloth_set_2_edgetam_sam31_experiments` | 44 | Sloth Set 2 EdgeTAM/SAM3.1 mask, PCD, hand/controller, and compile experiments. |
| `sam_family_benchmarks` | 17 | SAM2.1/SAM3.1 benchmark outputs and masks/quality assets. |
| `edgetam_integration` | 16 | EdgeTAM environment, ONNX/TRT, and streaming proof outputs. |
| `ffs_validation` | 19 | FFS environment, TensorRT, static replay, live viewer, and backend validation. |
| `hardware_capture_wsl` | 7 | D455, WSL, firmware, and capture probe outputs. |
| `repo_contract_retention` | 6 | Cleanup, retention, scope, and contract hardening notes. |
| `visualization_stack` | 3 | Visualization ownership and cleanup validation. |
| `supporting_assets_and_raw_outputs` | 3 | Small JSON/raw helper outputs not covered by the main categories. |

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

- Latest Demo 2.1 live profile:
  `demo2_1_visual5fps_image_sam31_profile_object_only_120s.md`
- Demo 2.1 source-of-truth validation:
  `demo2_1_shared_ffs_per_camera_edgetam_validation.md`
- Harness engineering map:
  `scripts/harness/README.md`
- Deterministic validation:
  `python scripts/harness/check_all.py`
