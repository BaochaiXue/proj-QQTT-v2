# Static Replay FFS Matrix Validation

## Environment Prep

Executed in `qqtt-ffs-compat`.

Additional local packages installed into that environment for this workflow:

```bash
conda run -n qqtt-ffs-compat python -m pip install python-pptx
conda run -n qqtt-ffs-compat python -m pip install onnx
```

Resolved versions:

- `python-pptx==1.0.2`
- `onnx==1.21.0`

## Command

Incremental repair / rerun against the existing full-run root:

```bash
/home/zhangxinjie/miniconda3/envs/qqtt-ffs-compat/bin/python scripts/harness/run_ffs_static_replay_matrix.py --output_root /home/zhangxinjie/proj-QQTT-v2/data/experiments/ffs_static_replay_matrix_20260422_fullrun --reuse_artifacts
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

Visual settings:

- `frame_idx = 10`
- `mask_prompt = stuffed animal`

## Incremental Repair Note

This run reused the existing full-run root and only repaired the interrupted tail of the expanded matrix.

State before repair:

- completed experiments: `51 / 54`
- missing experiments:
  - `two_stage_fp16__model_20-30-48__scale_0p8__iters_4`
  - `two_stage_fp16__model_20-30-48__scale_0p8__iters_2`
  - `two_stage_fp16__model_20-30-48__scale_0p5__iters_8`
- interrupted partial artifact cleaned before rerun:
  - `artifacts/two_stage_fp16/model_20-30-48/scale_0p8_iters_4/`

State after repair:

- completed experiments: `54 / 54`
- failed experiments: `0 / 54`

## Mask Note

This incremental repair intentionally reused the existing experiment root, including its `mask_cache/`.

- no completed experiment was rerun
- no existing PCD panel was regenerated
- the current deck therefore preserves the previously cached static replay masks for the already-finished experiments

If a future run needs all `54` slides regenerated under the now-working SAM 3.1 environment, that should be done as a separate full visual rerender task.

## Outputs

- output root:
  - `/home/zhangxinjie/proj-QQTT-v2/data/experiments/ffs_static_replay_matrix_20260422_fullrun`
- ranked CSV:
  - `/home/zhangxinjie/proj-QQTT-v2/data/experiments/ffs_static_replay_matrix_20260422_fullrun/results.csv`
- manifest:
  - `/home/zhangxinjie/proj-QQTT-v2/data/experiments/ffs_static_replay_matrix_20260422_fullrun/manifest.json`
- PPTX:
  - `/home/zhangxinjie/proj-QQTT-v2/data/experiments/ffs_static_replay_matrix_20260422_fullrun/ppt/ffs_static_replay_matrix.pptx`

## Outcome

- successful experiments: `54 / 54`
- failed experiments: `0 / 54`
- PPTX slide count: `54`
  - `54` experiment slides = `54 × 1`

Engine summary from `results.csv`:

- `two_stage_fp16`
  - mean overall FPS: `37.768`
  - best overall FPS: `69.815`
  - worst overall FPS: `14.443`
- `single_engine_fp32`
  - mean overall FPS: `29.098`
  - best overall FPS: `52.585`
  - worst overall FPS: `12.914`

Top 3 configs by overall mean FPS:

1. `two_stage_fp16__model_20-30-48__scale_0p5__iters_2`
   - overall mean FPS: `69.815`
2. `two_stage_fp16__model_20-30-48__scale_0p5__iters_4`
   - overall mean FPS: `65.196`
3. `two_stage_fp16__model_20-26-39__scale_0p5__iters_2`
   - overall mean FPS: `62.301`

Bottom 3 configs by overall mean FPS:

1. `two_stage_fp16__model_23-36-37__scale_1p0__iters_8`
   - overall mean FPS: `14.443`
2. `single_engine_fp32__model_23-36-37__scale_1p0__iters_4`
   - overall mean FPS: `14.286`
3. `single_engine_fp32__model_23-36-37__scale_1p0__iters_8`
   - overall mean FPS: `12.914`

## PPT Contract

The exported deck is now the compact single-slide format:

- one slide per experiment
- top-of-slide text only:
  - `engine`
  - `model`
  - `scale`
  - `valid_iters`
  - `overall_mean_fps`
  - `Round 1/2/3 cam1/2/3 FPS`
- main image:
  - frame-10 masked FFS-only `3x3` PCD board
- no internal `experiment_id` shown on the slide
- `scale=0.75` is rendered as human-readable `0.75`, not the internal tokenized directory name `0p8`

## Notes

- Open3D rendering completed under EGL headless mode and emitted repeated DRI3 / EGL warnings on this machine, but the boards and PPTX were written successfully.
- Validation checks run after the repair:

```bash
conda run -n qqtt-ffs-compat python -m unittest -v tests.test_ffs_static_replay_matrix_smoke
conda run -n qqtt-ffs-compat python scripts/harness/check_all.py
```
