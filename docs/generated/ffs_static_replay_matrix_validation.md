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

```bash
/home/zhangxinjie/miniconda3/envs/qqtt-ffs-compat/bin/python scripts/harness/run_ffs_static_replay_matrix.py --output_root /home/zhangxinjie/proj-QQTT-v2/data/experiments/ffs_static_replay_matrix_20260422_fullrun --reuse_artifacts
```

## Inputs

- `data/static/ffs_30_static_round1_20260410_235202`
- `data/static/ffs_30_static_round2_20260414`
- `data/static/ffs_30_static_round3_20260414`

Matrix:

- `model ∈ {23-36-37, 20-26-39, 20-30-48}`
- `scale ∈ {1.0, 0.5}`
- `valid_iters ∈ {4, 2}`
- `engine ∈ {single_engine_fp32, two_stage_fp16}`
- `batch = 1`

Visual settings:

- `frame_idx = 10`
- `mask_prompt = stuffed animal`

## Mask Note

The current machine could not resolve a local SAM 3.1 checkpoint because the upstream HuggingFace repo is gated. For this static-only replay workflow, the harness therefore used its built-in static fallback:

- copied the existing frame-0 stuffed-animal masks from the static comparison outputs
- duplicated them to `frame_idx=10` under the experiment-local `mask_cache/`

This fallback is limited to the static replay harness and is recorded in each cached mask `summary.json`.

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

- successful experiments: `24 / 24`
- failed experiments: `0 / 24`
- PPTX slide count: `48`
  - `48` experiment slides = `24 × 2`

Engine summary from `results.csv`:

- `two_stage_fp16`
  - mean overall FPS: `40.795`
  - best overall FPS: `69.815`
  - worst overall FPS: `15.481`
- `single_engine_fp32`
  - mean overall FPS: `31.157`
  - best overall FPS: `52.585`
  - worst overall FPS: `14.286`

Top 3 configs by overall mean FPS:

1. `two_stage_fp16__model_20-30-48__scale_0p5__iters_2`
   - overall mean FPS: `69.815`
2. `two_stage_fp16__model_20-30-48__scale_0p5__iters_4`
   - overall mean FPS: `65.196`
3. `two_stage_fp16__model_20-26-39__scale_0p5__iters_2`
   - overall mean FPS: `62.301`

Bottom 3 configs by overall mean FPS:

1. `two_stage_fp16__model_23-36-37__scale_1p0__iters_4`
   - overall mean FPS: `15.481`
2. `single_engine_fp32__model_23-36-37__scale_1p0__iters_2`
   - overall mean FPS: `15.338`
3. `single_engine_fp32__model_23-36-37__scale_1p0__iters_4`
   - overall mean FPS: `14.286`

## Notes

- Open3D rendering completed under EGL headless mode and emitted repeated DRI3 / EGL warnings on this machine, but the boards and PPTX were written successfully.
- The experiment-local PPTX embeds:
  - one summary slide per experiment
  - one masked FFS-only PCD `3x3` board per experiment
