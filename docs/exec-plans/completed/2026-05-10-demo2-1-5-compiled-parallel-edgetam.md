# 2026-05-10 Demo 2.1.5 Compiled Parallel EdgeTAM

## Goal

Add a Demo 2.1.5 compiled parallel EdgeTAM worker path as an explicit option.

## Required Runtime Shape

- Demo 2.1.5 native RealSense depth path.
- `gpu_pipeline_mode=separate-workers`.
- Three persistent camera workers: `edgetam-cam0`, `edgetam-cam1`, `edgetam-cam2`.
- Each worker initializes its own HF EdgeTAM model/session via `_init_hf_model(camera_idx)`.
- `edgetam_model_topology=replicated`.
- `compile_mode=vision-reduce-overhead`.
- `gpu_gate=off`.
- Existing compiled vision encoder output clone wrapper remains active for this gate-off multi-worker path.
- The wrapper marks the CUDAGraph step at the compiled vision encoder boundary before each wrapped forward.
- No dummy compile prewarm is forced by this preset; the live camera workers own their model/session startup.
- The first compiled forward is serialized across camera workers to avoid PyTorch Inductor CUDAGraph first-capture TLS races; steady-state forwards remain parallel.
- EdgeTAM input tensors use a reusable pinned CPU + CUDA pixel slot per worker so compiled CUDAGraph inputs keep stable storage.

## Plan

1. Add a Demo 2.1.5 preset for the compiled parallel worker path.
2. Make the public Demo 2.1.5 `--parallel-edgetam` option select this preset.
3. Keep the staged/threadpool path available only through `--experimental-staged-parallel`.
4. Add deterministic contract tests for the new preset and thread layout.
5. Run focused tests, dry-run the public CLI, then run the repo deterministic checks.

## Validation

- `conda run --no-capture-output -n demo_2_max python -m unittest tests.test_demo_v2_1_5_realsense_depth_smoke`
- `conda run --no-capture-output -n demo_2_max python demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py --parallel-edgetam --dry-run`
- `conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py`
- `git diff --check`

Result: PASS for deterministic checks. Hardware profiling is required before treating this path as a performance result.
