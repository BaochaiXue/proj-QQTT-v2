# Demo 2.2 Async Filter 15 FPS Camera Input Benchmark

Status: hardware benchmark run completed; camera input increase helped, but the
strict Demo 2.2 pass threshold was still not reached.

## Target

- Keep the Demo 2.2 fastest known GPU schedule:
  - `single-owner`
  - `FFS cam0 -> cam1 -> cam2`
  - `EdgeTAM cam0 -> cam1 -> cam2`
- Raise RealSense input from `5 FPS` to `15 FPS`.
- Keep fused/render target at `5 FPS`.
- Keep local FFS TensorRT and compiled EdgeTAM.
- Keep async latest-wins object/controller PCD filtering.
- Render latest filtered fused PCD only.

## Command

```bash
conda run --no-capture-output -n demo_2_max python demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py \
  --preset demo2.2-async-filter-5fps \
  --fps 15 \
  --duration-s 120 \
  --profile-warmup-exclude-s 40 \
  --profile-json-output docs/generated/demo2_2_async_filter_15fps_input_profile.json \
  --debug
```

The dry-run contract confirmed `fps=15`, `fusion_target_fps=5.0`, and
`gpu_pipeline=single-owner`.

## Result

| Metric | Previous 5 FPS input | New 15 FPS input |
| --- | ---: | ---: |
| capture group FPS | `2.66` | `4.53` |
| raw fusion FPS | `2.64` | `4.52` |
| filter output FPS | `2.64` | `4.52` |
| filtered render FPS | `2.64` | `4.52` |
| complete fused groups | `213 / 221` | `351 / 354` |
| complete group ratio | `0.964` | `0.992` |
| capture skew median | `43.00 ms` | `19.12 ms` |
| capture skew p95 | `63.99 ms` | `52.26 ms` |
| target deficit vs 5 FPS | `2.36 FPS` | `0.48 FPS` |

Result: **FAIL** against the Demo 2.2 `4.8 FPS` pass threshold, but the failure
margin is now small: `4.52 FPS` measured vs `4.8 FPS` required.

## Latency Summary

| Stage | 5 FPS input median | 15 FPS input median |
| --- | ---: | ---: |
| GPU owner total | `177.91 ms` | `189.80 ms` |
| FFS cycle | `73.50 ms` | `87.09 ms` |
| EdgeTAM cycle | `102.20 ms` | `104.79 ms` |
| raw fusion | `7.75 ms` | `8.45 ms` |
| async filter total | `34.30 ms` | `36.59 ms` |
| render | `0.41 ms` | `0.39 ms` |

The higher camera input rate fixed most of the coherent-group starvation. The
remaining gap is not Open3D render; render is still below `1 ms` median. The
remaining losses come from upstream scheduling jitter and occasional FFS/filter
spikes.

## Interpretation

This confirms the previous `2.64 FPS` Demo 2.2 result was mostly a camera
coherent-group supply problem, not a lack of median GPU compute capacity.

The median GPU owner time in the 15 FPS input run was `189.80 ms`, which is
close to the `200 ms` budget for 5 FPS. Normal EdgeTAM latency remained stable at
about `31 ms` per camera. The system now often runs at or near `5 FPS`, but
periodic dips keep the full warmup-excluded average at `4.52 FPS`.

## Recommendation

Use `15 FPS` camera input for the local pure-5090 Demo 2.2 path. Keep GPU
execution single-owner and sequential. The next optimization should focus on
reducing jitter:

- smooth or decouple 15 FPS capture grouping down to a stable 5 FPS consumer;
- reduce FFS p95/max spikes;
- reduce enhanced object filter spikes above `200 ms`;
- consider enabling the reusable EdgeTAM CUDA pixel-value slots on the
  single-owner path without switching to three-way EdgeTAM parallelism.

## Artifacts

- Profile JSON: `docs/generated/demo2_2_async_filter_15fps_input_profile.json`
- Profile Markdown: `docs/generated/demo2_2_async_filter_15fps_input_profile.md`
