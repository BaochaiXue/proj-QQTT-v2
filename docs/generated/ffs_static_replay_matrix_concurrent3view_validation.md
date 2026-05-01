# Static Replay FFS Matrix Concurrent 3-View Validation

- Date: `2026-04-22`
- Machine: `XinjieZhang`
- Environment: `qqtt-ffs-compat`
- Purpose: rerun the full static replay matrix with the corrected benchmark semantics:
  - `3` views benchmarked simultaneously within each round
  - `batch = 1`
  - fresh benchmark results
  - reused TRT artifacts only

Current interpretation (`2026-05-01`): this is an offline static replay / TensorRT proxy report, not a live realtime report. Keep it separate from `docs/generated/ffs_live_3cam_benchmark_validation.md`, where live PyTorch 3-camera remains `RED / not realtime`.

## Command

```bash
conda run -n qqtt-ffs-compat python scripts/harness/run_ffs_static_replay_matrix.py --output_root /home/zhangxinjie/proj-QQTT-v2/result/ffs_static_replay_matrix_concurrent3view_20260422_fullrun --artifact_root /home/zhangxinjie/proj-QQTT-v2/result/_archived_obsolete/ffs_static_replay_matrix_20260422_sequential_obsolete_fullrun/artifacts --reuse_artifacts
```

## Inputs

- `data/static/ffs_30_static_round1_20260410_235202`
- `data/static/ffs_30_static_round2_20260414`
- `data/static/ffs_30_static_round3_20260414`

Matrix:

- `model ∈ {23-36-37, 20-26-39, 20-30-48}`
- `scale ∈ {1.0, 0.75, 0.5}`
- `valid_iters ∈ {8, 4, 2}`
- `engine ∈ {single_engine_fp32, two_stage_fp16}`
- `batch = 1`

Semantics:

- each round launches `3` subprocess workers, one per camera
- workers warm up on frames `0..9`
- workers measure frames `0..29`
- each `Round X cam Y FPS` is a simultaneous-3-view per-camera result
- `overall_mean_fps = mean(all 9 per-camera FPS values)`

## Output Roots

- new benchmark output root:
  - `/home/zhangxinjie/proj-QQTT-v2/result/ffs_static_replay_matrix_concurrent3view_20260422_fullrun`
- reused artifact root:
  - `/home/zhangxinjie/proj-QQTT-v2/result/_archived_obsolete/ffs_static_replay_matrix_20260422_sequential_obsolete_fullrun/artifacts`

Key outputs:

- `results.csv`
- `manifest.json`
- `ppt/ffs_static_replay_matrix.pptx`

## Outcome

- successful experiments: `54 / 54`
- failed experiments: `0 / 54`
- PPT slide count: `54`
- each experiment contributes:
  - one top-of-slide summary block
  - one frame-10 masked FFS-only `3x3` PCD panel

Engine summary from `results.csv`:

- `two_stage_fp16`
  - mean overall FPS: `18.672`
  - best overall FPS: `36.697`
  - worst overall FPS: `5.643`
- `single_engine_fp32`
  - mean overall FPS: `14.491`
  - best overall FPS: `29.817`
  - worst overall FPS: `5.145`

Top 3 configs by concurrent-3-view `overall_mean_fps`:

1. `two_stage_fp16__model_20-30-48__scale_0p5__iters_2`
   - overall mean FPS: `36.697`
2. `two_stage_fp16__model_20-30-48__scale_0p5__iters_4`
   - overall mean FPS: `35.308`
3. `two_stage_fp16__model_20-26-39__scale_0p5__iters_2`
   - overall mean FPS: `32.368`

Proxy reporting note:

- these are concurrent static replay proxy FPS values over recorded aligned rounds
- they do not prove live PyTorch 3-camera realtime
- the current level-5 proxy artifact default is `20-30-48 / valid_iters=4 / 848x480 -> 864x480 / builderOptimizationLevel=5`

Bottom 3 configs:

1. `two_stage_fp16__model_23-36-37__scale_1p0__iters_4`
   - overall mean FPS: `5.960`
2. `two_stage_fp16__model_23-36-37__scale_1p0__iters_8`
   - overall mean FPS: `5.643`
3. `single_engine_fp32__model_23-36-37__scale_1p0__iters_8`
   - overall mean FPS: `5.145`

## Spot Checks

- slide 1:
  - `two_stage_fp16 | model=20-30-48 | scale=0.5 | valid_iters=2 | overall_mean_fps=36.70`
- slide 54:
  - `single_engine_fp32 | model=23-36-37 | scale=1.0 | valid_iters=8 | overall_mean_fps=5.15`
- requested target config:
  - `two_stage_fp16 | model=20-30-48 | scale=0.75 | valid_iters=8`
  - found on slide `17`
  - overall mean FPS: `21.04`

## Mask Note

This rerun used a fresh output root, but the local mask generation path still fell back to the existing static frame-0 stuffed-animal masks:

- `generation_mode = copied_static_frame0_fallback`
- one fallback mask cache per round under the new output root

So the benchmark and PCD boards are fresh, but the frame-10 masks still come from the static fallback copy path rather than a newly generated SAM 3.1 result.

## Obsolescence Note

The older output root:

- `/home/zhangxinjie/proj-QQTT-v2/result/_archived_obsolete/ffs_static_replay_matrix_20260422_sequential_obsolete_fullrun`

used the obsolete single-camera sequential benchmark semantics and should now be treated as historical only. It may still be reused as an artifact cache, but not as the source of current benchmark conclusions.

## Validation Commands

```bash
conda run -n qqtt-ffs-compat python -m unittest -v tests.test_ffs_static_replay_matrix_smoke
conda run -n qqtt-ffs-compat python scripts/harness/check_all.py
```

## Runtime Notes

- Open3D offscreen rendering completed successfully under EGL headless mode.
- The run emitted repeated DRI3 / EGL warnings on this machine, but all PCD boards and the PPTX were written successfully.
