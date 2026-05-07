# Generated Docs Index

This folder keeps generated validation notes, inventories, profile summaries,
and small reusable helper assets. It is intentionally not a curated product
manual; use the compressed harness index first.

## Start Here

- `harness_engineering_compact_index.md`: concise human-facing summary of the
  current harness engineering state.
- `harness_engineering_artifact_inventory.json`: machine-readable inventory of
  generated artifacts by category.
- `../SCOPE.md`, `../WORKFLOWS.md`, and `../ARCHITECTURE.md`: stable repo
  contracts outside generated outputs.

## Current Source-Of-Truth Reports

| Area | Primary report |
| --- | --- |
| Demo 2.1 three-view fused PCD | `demo2_1_shared_ffs_per_camera_edgetam_validation.md` |
| Demo 2.1 single GPU-owner pipeline | `demo2_1_single_gpu_owner_pipeline.md` |
| Demo 2.1 temporary controller prompt probe | `demo2_1_controller_prompt_probe.md` |
| Demo 2.1 towel-controller single-owner benchmark | `demo2_1_controller_towel_single_owner_benchmark.md` |
| Demo 2.1 single EdgeTAM GPU saturation probe | `demo2_1_single_edgetam_gpu_saturation_probe.md` |
| Demo 2.1 three EdgeTAM GPU probe | `demo2_1_three_edgetam_gpu_probe.md` |
| Latest Demo 2.1 visual profile | `demo2_1_visual5fps_image_sam31_profile_object_only_120s.md` |
| Demo 2.1 pin-memory H2D ablation | `demo2_1_pin_memory_ablation.md` |
| Demo 2 single-camera local FFS path | `demo2_local_ffs_professor_speed_polish.md` |
| Demo 2 depth source comparison | `demo2_depth_source_fps_comparison.md` |
| Remote FFS service | `demo2_remote_ffs_service_validation.md`; latest 4090/5090 probe: `demo2_remote_ffs_4090_server_5090_client_validation.md`; WSL-5090 synthetic LAN echo: `demo2_remote_ffs_echo_wsl5090_client.md`; WSL-5090 real-IR depth benchmark: `demo2_remote_ffs_real_ir_depth_benchmark_wsl5090.md` |
| HF EdgeTAM streaming | `hf_edgetam_streaming_processor_session_validation.md` |
| HF EdgeTAM compile ablation | `sloth_set_2_motion_ffs_hf_edgetam_streaming_compile_ablation.md` |
| FFS backend integration | `ffs_depth_backend_integration_validation.md` |
| FFS speed/quality tradeoff | `ffs_benchmark_tradeoff_validation.md` |
| SAM3.1 environment | `sam31_env_validation.md` |
| Visualization stack | `visual_stack_cleanup_validation.md` |
| Repo contracts | `contract_hardening_validation.md` |

## Artifact Families

- Hardware / WSL / RealSense probes: `d455_*`, `wsl_*`,
  `both_eval_*`.
- FFS validation and runtime profiles: `ffs_*`, `demo2_*`.
- SAM / EdgeTAM benchmarks: `sam21_*`, `sam31_*`, `edgetam_*`,
  `hf_edgetam_*`, `sloth_*`.
- Demo 2.1 live runtime profiles: `demo2_1_*`.
- Repo and retention hardening: `repo_*`, `contract_*`,
  `*_cleanup_*`, `*_retention_*`.
- Reusable helper assets: small `.json`, `.png`, `.txt`, and `.log`
  files linked by the reports above.

## Retention Guidance

- Prefer extending an existing validation report over creating a near-duplicate
  top-level note.
- Keep raw logs/results only when a Markdown report or compact index explains
  why they matter.
- Do not remove historical artifacts without a recorded cleanup pass.
- Put newly superseding claims into `harness_engineering_compact_index.md`.

## Checks

```bash
python scripts/harness/check_all.py
python scripts/harness/check_all.py --full
```
