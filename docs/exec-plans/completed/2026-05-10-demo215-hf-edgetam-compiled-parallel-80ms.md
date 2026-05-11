# Demo 2.1.5 HF EdgeTAM Compiled Parallel 80ms Plan

## Goal
Make Demo 2.1.5 report and enforce three-camera HF EdgeTAM segmentation stage wall time for `object=stuffed animal`, `controller=towel`, controller-object tracking. Target is `edgetam_stage_wall_ms` p50 < 80 ms.

## Steps
1. Fix public wrapper precedence so `--mask-only-debug --parallel-edgetam` remains mask-only while preserving parallel EdgeTAM settings.
2. Add explicit runtime contract fields for compile target status, graph output policy, per-camera stream mode, and stage-wall gate.
3. Record per-camera EdgeTAM job start/publish times and compute complete three-camera mask group stage wall metrics for separate-workers and staged paths.
4. Generalize compiled-module output clone wrapping for reduce-overhead graph modes and record compiled module ids/types per camera.
5. Add an analyzer script that compares profile JSONs, emits markdown/json, and fails when no profile passes p50 < 80 ms.
6. Add deterministic tests for wrapper mapping, graph output policy defaults, batch-vision flagging, and analyzer pass/fail behavior.
7. Run baseline and compile-mode live profiles where hardware is available; keep PR draft if no mode passes the 80 ms gate.

## Constraints
- Do not use `controller-prompt=rag`; current scene uses `towel`.
- Do not put full-device `torch.cuda.synchronize()` in the hot path except under explicit profiling or existing stream completion boundaries.
- Do not change production defaults without evidence.

## Result
- Strict replicated three-worker compiled path did not meet the 80 ms gate; best replicated profile was `vision-reduce-overhead` with p50 `96.11 ms`.
- Experimental batch-vision shared-model path met the stage-wall gate with p50 `77.92 ms`, p90 `86.24 ms`, and complete mask group FPS `12.48`.
- Components compile modes are not promotable yet: `components-max-autotune-no-cudagraphs` and `components-reduce-overhead` produced no valid steady-state stage samples in this run.
- Primary report: `docs/generated/demo215_edgetam80_compiled_parallel_report.md`.
