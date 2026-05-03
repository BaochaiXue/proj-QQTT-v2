# HF EdgeTAM Streaming Compile Ablation

## Goal

Try `torch.compile` on the current HF EdgeTAM frame-by-frame streaming path
without changing the default eager benchmark or production-like behavior.

## Scope

- Extend `scripts/harness/experiments/run_hf_edgetam_streaming_realcase.py` with
  explicit compile modes.
- Keep `--compile-mode none` as the default and preserve the current baseline
  timing path.
- Record compile metadata in JSON/Markdown so results cannot be mixed with the
  eager baseline.
- Use real QQTT frame-by-frame streaming input only:
  - `frame_by_frame_streaming=true`
  - `offline_video_input_used=false`
  - `frame_source=png_loop`
- Do not modify `edgetam-max`, `edgetam-hf-stream`, or model checkpoints.

## Compile Modes

- `none`: current eager baseline.
- `model-default`: compile the HF model wrapper with `mode="default"`,
  `fullgraph=False`, `dynamic=False`.
- `model-reduce-overhead`: compile the HF model wrapper with
  `mode="reduce-overhead"`, `fullgraph=False`, `dynamic=False`.
- `vision-reduce-overhead`: compile detected vision-related submodules only.
- `components-reduce-overhead`: compile detected vision, memory, encoder, and
  decoder submodules where stable module names exist.

## Validation

- Add lightweight tests for CLI parsing and compile-target selection metadata.
- Run deterministic harness checks.
- Run at least a local-data compile smoke on `sloth_set_2_motion_ffs` with
  frame-by-frame PNG input and mask prompt.
- Record command outcomes in generated docs.

## Outcome

- Added `--compile-mode` to
  `scripts/harness/experiments/run_hf_edgetam_streaming_realcase.py`.
- Default remains `--compile-mode none`, so the current eager path is preserved.
- Added compile metadata to JSON, Markdown, and quality outputs.
- Added warmup-failure reporting so unstable compile modes still write
  generated result artifacts before returning non-zero.
- Added tests for default parsing and explicit compile-target selection.
- Ran full local-data `sloth_set_2_motion_ffs` compile ablation:
  - same-run eager control passed
  - `vision-reduce-overhead` passed and is the recommended experimental mode
  - full-model compile modes failed during warmup
  - components compile failed with CUDA Graph output overwrite during warmup
- `vision-reduce-overhead` improved same-run eager median subsequent latency
  from `16.24 ms` to `11.47 ms` and model-only FPS from `65.22` to `96.19`.
- Compiled masks vs same-run eager masks: `279` frames compared, `0` missing,
  mean mask IoU `0.996102`, median `0.997898`.
- Summary recorded in
  `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_streaming_compile_ablation.md`.
- Validation passed:
  - `conda run --no-capture-output -n SAM21-max python -m json.tool docs/generated/sloth_set_2_motion_ffs_hf_edgetam_streaming_compile_ablation.json`
  - `conda run --no-capture-output -n SAM21-max python -m unittest -v tests.test_sam21_checkpoint_ladder_panel_smoke`
  - `conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py`
