# Sloth Set 2 HF EdgeTAM Streaming Compile Ablation

Date: 2026-05-03

## Contract

- Case: `data/different_types/sloth_set_2_motion_ffs`
- Cameras: `0,1,2`
- Frames: all `93` frames per camera
- Prompt: SAM 3.1 frame-0 `stuffed animal` mask prompt
- Streaming input: one PNG at a time through the HF session API
- `frame_by_frame_streaming=true`
- `offline_video_input_used=false`
- `frame_source=png_loop`
- Environment: `edgetam-hf-stream`

## Result

The safe compile mode is `vision-reduce-overhead`: compile only
`model.vision_encoder` with `torch.compile(mode="reduce-overhead",
fullgraph=False, dynamic=False)`.

It passed the full real-data run and improved same-run eager throughput. The
realcase CLI now defaults to this mode; pass `--compile-mode none` when an eager
control is required.

| mode | status | targets | subsequent median ms | e2e FPS median | model-only FPS median | note |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `none` | pass | none | 16.24 | 60.83 | 65.22 | same-run eager control |
| `vision-reduce-overhead` | pass | `vision_encoder` | 11.47 | 86.19 | 96.19 | recommended experimental mode |
| `model-default` | fail | full HF model | n/a | n/a | n/a | Inductor failed on `inference_session.get_frame(...).unsqueeze(0)` |
| `model-reduce-overhead` | fail | full HF model | n/a | n/a | n/a | same full-model session/state compile failure |
| `components-reduce-overhead` | fail | vision, memory, encoder, decoder | n/a | n/a | n/a | CUDA Graph output overwrite during warmup |

Compared with the same-run eager control, `vision-reduce-overhead` changed:

```text
subsequent median latency: 16.24 ms -> 11.47 ms
latency reduction: about 29.4%
e2e FPS median: 60.83 -> 86.19
model-only FPS median: 65.22 -> 96.19
```

The older baseline from
`sloth_set_2_motion_ffs_hf_edgetam_streaming_results.json` was 18.93 ms
subsequent median and 55.80 model-only FPS. The same-run eager control is the
fairer comparison because it was produced by the updated script and `5` warmup
frames.

## Quality Gate

`vision-reduce-overhead` masks were compared against the same-run eager masks:

```text
frames compared: 279
missing frames: 0
mean 2D mask IoU vs eager: 0.996102
median 2D mask IoU vs eager: 0.997898
min 2D mask IoU vs eager: 0.980968
mean area delta vs eager: -1.80 px
```

This clears the gate for using this mode as the default HF realcase streaming
benchmark path. It should still be labeled as the HF vision-compiled backend
when comparing against eager or other model families.

## Artifacts

- Eager control:
  `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_streaming_compile_eager_none_results.json`
- Vision compile:
  `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_streaming_compile_vision_reduce_overhead_results.json`
- Vision vs eager quality:
  `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_streaming_compile_vision_reduce_overhead_vs_same_run_eager_quality.json`
- Full model default failure:
  `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_streaming_compile_model_default_results.json`
- Full model reduce-overhead failure:
  `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_streaming_compile_model_reduce_overhead_results.json`
- Components reduce-overhead failure:
  `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_streaming_compile_components_reduce_overhead_results.json`

## Decision

- Default to `--compile-mode vision-reduce-overhead`.
- Keep `--compile-mode none` available for explicit eager-control and regression
  checks.
- Do not use full-model compile for HF streaming; the session/state path is not
  compile-stable in this environment.
- Do not use `components-reduce-overhead` until the CUDA Graph output overwrite
  issue is addressed with a targeted clone/marker strategy and re-benchmarked.
