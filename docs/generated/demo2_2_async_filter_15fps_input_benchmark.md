# Demo 2.2 Async Filter 15 FPS Camera Input Benchmark

Status: hardware benchmark run completed. `15 FPS` capture is now the Demo 2.2
default, and the formal Demo 2.2 report target is now also `15 FPS` for the full
3-camera object+controller tracking path.

## Target

- Keep the Demo 2.2 fastest known GPU schedule:
  - `single-owner`
  - `FFS cam0 -> cam1 -> cam2`
  - `EdgeTAM cam0 -> cam1 -> cam2`
- Use the Demo 2.2 default RealSense input rate: `15 FPS`.
- Keep profile/pass target at `15 FPS`.
- Build capture groups at the camera input cadence, `15 FPS`, so the async
  pipeline can consume the newest coherent group instead of being throttled at
  the pass target.
- Keep local FFS TensorRT and compiled EdgeTAM.
- Keep async latest-wins object/controller PCD filtering.
- Render latest filtered fused PCD only.

## Command

```bash
conda run --no-capture-output -n demo_2_max python demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py \
  --preset demo2.2-async-filter-5fps \
  --duration-s 120 \
  --profile-warmup-exclude-s 40 \
  --profile-json-output docs/generated/demo2_2_async_filter_15fps_input_profile.json \
  --debug
```

The current dry-run contract confirms the preset default: `fps=15`,
`capture_group_target_fps=15.0`, `fusion_target_fps=15.0`, and
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
| current target deficit vs 15 FPS | `12.36 FPS` | `10.48 FPS` |

Result: **FAIL** against the current Demo 2.2 `15 FPS` target. With the standard
96% pass threshold, the required filtered render rate is `14.4 FPS`; this run
measured `4.52 FPS`.

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
close to a `5 FPS` processing budget but far above the `66.7 ms` budget required
for a true `15 FPS` full pipeline. Normal EdgeTAM latency remained stable at
about `31 ms` per camera, so three serialized EdgeTAM camera passes already take
roughly `90-100 ms` before FFS, fusion, filtering, and rendering.

## Recommendation

Use `15 FPS` camera input and `15 FPS` capture-group construction for the local
pure-5090 Demo 2.2 path, but treat this as the formal target rather than an
achieved result. The current implementation cannot reach full 15 FPS with
serialized FFS plus serialized EdgeTAM plus filtered object/controller PCD.

The next optimization should focus on structural throughput, not only jitter:

- reduce or batch the serialized three-camera EdgeTAM cost;
- reduce FFS cycle time and p95/max spikes;
- stabilize capture grouping and latest-wins scheduling at the full 15 FPS target;
- reduce enhanced object filter spikes above `200 ms`;
- consider enabling the reusable EdgeTAM CUDA pixel-value slots on the
  single-owner path without switching to three-way EdgeTAM parallelism.

## Artifacts

- Profile JSON: `docs/generated/demo2_2_async_filter_15fps_input_profile.json`
- Profile Markdown: `docs/generated/demo2_2_async_filter_15fps_input_profile.md`
