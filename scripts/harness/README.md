# Harness Engineering Map

`scripts/harness/` is the operator-facing edge of the repo: CLIs, probes, deterministic guards, and bounded diagnostics. Reusable calibration, geometry, point-cloud, layout, and render logic belongs under `data_process/`.

## Filesystem Contract

- Keep stable public CLIs at their existing paths unless docs and tests move with them.
- Put one-off visualization experiments under `scripts/harness/experiments/`.
- Keep formal recording/alignment code free of `scripts.harness.experiments` and `data_process.visualization.experiments` imports.
- Keep external repos, checkpoints, TensorRT engines, SAM assets, and generated proof outputs outside harness code.
- Add every new harness Python file to `scripts/harness/_catalog.py`.

## Catalog And Checks

`_catalog.py` is the compact source of truth for harness categories and help-surface coverage. `check_harness_catalog.py` verifies that every public harness Python file is categorized, and `check_all.py` reads its harness help commands from the same catalog.

Current catalog summary:

| Category | Count | Meaning |
| --- | ---: | --- |
| `checks` | 5 | Deterministic repo, scope, architecture, experiment-boundary, and catalog guards. |
| `hardware_external` | 13 | RealSense probes, SAM/FFS/TensorRT proof tools, WSLg/Open3D helper, and static replay benchmarks. |
| `mask_support` | 4 | SAM 3.1 mask generation, mask helper, object-case registry, and single-pair reprojection. |
| `formal_cleanup` | 1 | Downstream-facing cleanup for `data/different_types/`. |
| `current_compare` | 12 | In-scope aligned native-vs-FFS comparison visualizations. |
| `experiments` | 20 | Experiment-only visualization workflows under `scripts/harness/experiments/`. |
| `focused_diagnostics` | 3 | Narrow audits and quality diagnostics. |

```bash
python scripts/harness/check_harness_catalog.py
python scripts/harness/check_all.py
python scripts/harness/check_all.py --full
```

`check_all.py` defaults to the quick profile: help-surface checks, deterministic
guards, and non-rendering core smoke tests. Render-heavy visualization smoke
tests that create temporary image/video outputs run in `check_all.py --full`.

## CUDA 13 Toolkit Policy

- Shared CUDA 13 toolkit is `/usr/local/cuda`, currently resolved through alternatives to `/usr/local/cuda-13.2`.
- Future CUDA 13-family extension builds from harness or external proof workflows should reuse this shared toolkit instead of installing another env-local CUDA toolkit.
- Standard build exports:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=12.0
```

- Do not install duplicate `cuda-toolkit`, `cuda-nvcc`, or similar compiler stacks into new conda environments just to build CUDA 13 extensions.
- Do not install `cuda`, `cuda-drivers`, or Linux NVIDIA driver packages inside WSL.
- Existing validated environments may keep their current env-local CUDA paths until they are intentionally rebuilt; new CUDA 13 build work should start from the shared toolkit policy above.

## FFS Defaults

- Environment: `FFS-SAM-RS`.
- Checkpoint/config: `20-30-48`, `valid_iters=4`, `max_disp=192`, two-stage ONNX/TensorRT.
- TensorRT artifact: `result/ffs_trt_static_rounds_848x480_pad864_builderopt5_rtx5090_laptop_20260428/engines/model_20-30-48_iters_4_res_480x864/`.
- Local performance claims should use real RealSense `848x480` `ir_left` / `ir_right` inputs unless explicitly labeled synthetic or control.
- Models requiring multiples of `32` pad `848x480` to `864x480` and unpad outputs afterward.
- `640x480` runs are official-table reproduction/control benchmarks only.

## FFS Performance Boundary

- Live PyTorch 3-camera FFS is not realtime on the RTX 5090 laptop. The best recorded `20-30-48 / valid_iters=4 / scale=0.5` live run reached about `22.6` aggregate FFS FPS, or about `7.5` FPS per camera. Source: `docs/generated/ffs_live_3cam_benchmark_validation.md`.
- Static replay / TensorRT proxy is a separate result family. The current proxy target is `20-30-48 / valid_iters=4 / 848x480 -> 864x480 / builderOptimizationLevel=5`, and it has basically reached the target. Source: `result/ffs_trt_static_rounds_848x480_pad864_builderopt5_rtx5090_laptop_20260428/report.md`.
- Do not label saved-pair, static replay, or TensorRT proxy numbers as "live PyTorch 3-camera realtime".
- `benchmark_ffs_configs.py` is for saved-pair PyTorch screening; `run_ffs_static_replay_matrix.py` is for offline static replay / TensorRT proxy measurement; `cameras_viewer_FFS.py --ffs_backend pytorch` is the authoritative live PyTorch path.

## Primary Entrypoints

- Checks: `check_all.py`, `check_harness_catalog.py`, `check_scope.py`, `check_experiment_boundaries.py`, `check_visual_architecture.py`.
- Hardware and external proofs: `probe_d455_*`, `verify_ffs_*`, `benchmark_ffs_configs.py`, `benchmark_sam31_still_object_views.py`, `run_ffs_static_replay_matrix.py`, `realtime_single_camera_pointcloud.py`, `run_wslg_open3d.sh`.
- Mask support: `generate_sam31_masks.py`, `sam31_mask_helper.py`, `object_case_registry.py`, `reproject_ffs_to_color.py`.
- Formal cleanup: `cleanup_different_types_cases.py`.
- Current compare CLIs: `visual_compare_depth_panels.py`, `visual_compare_reprojection.py`, `visual_compare_depth_video.py`, `visual_compare_depth_triplet_*`, `visual_compare_masked_*`, `visual_compare_turntable.py`, `visual_compare_rerun.py`, `visual_make_*`, `visual_compare_stereo_order_pcd.py`.
- Experiment CLIs: everything under `scripts/harness/experiments/`; these call experiment implementation modules and stay out of formal runtime imports.
- Focused diagnostics: `audit_ffs_left_right.py`, `compare_face_smoothness.py`, `diagnose_floating_point_sources.py`.

## Object Cases

- Use `scripts.harness.object_case_registry` for raw object-case lookup by `(object_set, round_id)`.
- Keep old `static_object` captures separate from 2026-04-28 `still_object` captures.
- Current still-object raw captures are `both_30_still_object_round1_20260428`, `round2`, `round3`, `round4`, `round7`, and `round8`; the round5-6 skip is intentional.
- Aligned still-object outputs should use a distinct namespace such as `data/still_object/`.

## Retention Policy

Keep current user-facing CLIs, deterministic checks, hardware probes, and bounded diagnostics. Remove ignored cache directories, keep reusable implementation out of harness, and record durable external-dependency proof results under `docs/generated/`.

Result retention policy:

- Delete local smoke, preview, interrupted, and debug-only result roots once the corresponding full result exists.
- Keep obsolete-but-useful artifacts under `result/_archived_obsolete/`.
- Keep invalid-for-QQTT controls under `result/_archived_invalid_for_qqtt/`.
- Keep saved-pair offline FFS screening under `data/ffs_benchmarks/_archived_saved_pair_offline/`.
- Prefer the top-level `builderopt5`, `concurrent3view`, `stable_throughput`, and `live_3cam_scale*.log` results for current reporting.
