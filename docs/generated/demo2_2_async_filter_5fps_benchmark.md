# Demo 2.2 Async Filter 5 FPS Benchmark

Status: historical hardware benchmark run completed; FPS target not reached.
Demo 2.2 now defaults to `15 FPS` capture and a `15 FPS` formal target, so
reproduce this old run with explicit `--fps 5 --fusion-target-fps 5`.

## Target

- Pure RTX 5090 Laptop local path.
- 3x RealSense RGB+IR at explicit `5 FPS`.
- Local FFS TensorRT `20-30-48 / valid_iters=4 / 848x480->864x480 / builderOptimizationLevel=5`.
- Compiled EdgeTAM with live SAM3.1 first-frame object/controller initialization.
- Raw fused semantic PCD build is separate from async latest-wins filtering.
- Render displays latest filtered fused PCD only.

## Baseline

- Demo 2.1 single-owner object+controller baseline: `4.44 FPS`.
- Baseline report: `docs/generated/demo2_1_controller_towel_single_owner_no_pin_repeat_120s.md`.

## Demo 2.2 Command

```bash
conda run --no-capture-output -n demo_2_max python demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py \
  --preset demo2.2-async-filter-5fps \
  --fps 5 \
  --fusion-target-fps 5 \
  --duration-s 120 \
  --profile-warmup-exclude-s 40 \
  --profile-json-output docs/generated/demo2_2_async_filter_5fps_profile.json \
  --debug
```

## PASS Criteria

- Warmup-excluded filtered render FPS >= `4.8`.
- `depth_source=ffs`.
- `filter_scheduler.render_filtered_only=true`.
- No native-depth or remote-FFS fallback.

## Result

| Metric | Warmup-excluded value |
| --- | ---: |
| capture group FPS | `2.66` |
| GPU owner / raw fusion FPS | `2.64` |
| filter output FPS | `2.64` |
| filtered render FPS | `2.64` |
| complete fused groups | `213 / 221` |
| complete group ratio | `0.964` |
| target deficit | `2.36 FPS` |

Result: **FAIL** against the `4.8 FPS` pass threshold.

The run used the intended pure-local path: local FFS, single-owner GPU scheduling,
compiled EdgeTAM, async latest-wins filtering, and filtered-only render. The
measured bottleneck class is `upstream_supply`.

## Latency Summary

| Stage | Median | p90 | p95 | Max |
| --- | ---: | ---: | ---: | ---: |
| GPU owner total | `177.91 ms` | `206.76 ms` | `216.46 ms` | `397.69 ms` |
| FFS cycle | `73.50 ms` | `91.57 ms` | `95.38 ms` | `302.33 ms` |
| EdgeTAM cycle | `102.20 ms` | `116.45 ms` | `123.63 ms` | `150.69 ms` |
| raw fusion | `7.75 ms` | `9.32 ms` | `9.93 ms` | `13.11 ms` |
| async filter total | `34.30 ms` | `38.08 ms` | `40.07 ms` | `236.96 ms` |
| render | `0.41 ms` | `0.62 ms` | `1.83 ms` | `4.32 ms` |

The async split worked mechanically: raw fusion and render are cheap, and normal
filter latency is about `34 ms`. There are occasional enhanced object filter
spikes above `200 ms`, but the aggregate FPS is mostly limited by upstream
capture/group/GPU-owner supply, not Open3D render.

## Artifacts

- Profile JSON: `docs/generated/demo2_2_async_filter_5fps_profile.json`
- Profile Markdown: `docs/generated/demo2_2_async_filter_5fps_profile.md`
