# Demo 2.2 Staged Parallel 5 FPS Benchmark

Status: historical hardware benchmark run completed; FPS target not reached.
Demo 2.2 now defaults to `15 FPS` capture and a `15 FPS` formal target, so
reproduce this old run with explicit `--fps 5 --fusion-target-fps 5`.

## Target

- Pure RTX 5090 Laptop local path.
- 3x RealSense RGB+IR at explicit `5 FPS`.
- Local FFS TensorRT `20-30-48 / valid_iters=4 / 848x480->864x480 / builderOptimizationLevel=5`.
- Staged GPU schedule:
  - FFS cam0 -> cam1 -> cam2 sequentially.
  - EdgeTAM cam0/cam1/cam2 in parallel on per-camera CUDA streams.
- Compiled EdgeTAM with live SAM3.1 first-frame object/controller initialization.
- FFS input staging uses pinned CPU memory plus persistent CUDA input buffers.
- EdgeTAM pixel-value staging uses pinned CPU memory plus reusable CUDA device slots.
- Raw fused semantic PCD build remains separate from async latest-wins filtering.
- Render displays latest filtered fused PCD only.

## Command

```bash
conda run --no-capture-output -n demo_2_max python demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py \
  --preset demo2.2-staged-parallel-5fps \
  --fps 5 \
  --fusion-target-fps 5 \
  --duration-s 120 \
  --profile-warmup-exclude-s 40 \
  --profile-json-output docs/generated/demo2_2_staged_parallel_5fps_profile.json \
  --debug
```

## PASS Criteria

- Warmup-excluded filtered render FPS >= `4.8`.
- `depth_source=ffs`.
- `gpu_pipeline=staged`.
- `staged_order=ffs-then-parallel-edgetam`.
- `filter_scheduler.render_filtered_only=true`.
- No native-depth or remote-FFS fallback.

## Result

| Metric | Warmup-excluded value |
| --- | ---: |
| capture group FPS | `2.74` |
| raw fusion FPS | `2.71` |
| filter output FPS | `2.71` |
| filtered render FPS | `2.70` |
| complete fused groups | `218 / 223` |
| complete group ratio | `0.978` |
| target deficit | `2.30 FPS` |

Result: **FAIL** against the `4.8 FPS` pass threshold.

The implementation did run the intended staged-parallel shape, but it did not
improve throughput. The measured bottleneck class is still `upstream_supply`.

## Comparison

| Run | GPU schedule | Filter | Render FPS after warmup |
| --- | --- | --- | ---: |
| Demo 2.1 object+controller baseline | single-owner | synchronous fused path | `4.44` |
| Demo 2.2 async filter | single-owner FFS then EdgeTAM | async latest-wins | `2.64` |
| Demo 2.2 staged parallel | FFS sequential, EdgeTAM 3-way parallel | async latest-wins | `2.70` |

The staged-parallel version is only slightly above the previous async-filter
run and remains far below the older Demo 2.1 single-owner baseline. It should
not become the default production/demo schedule.

## Latency Summary

| Stage | Median | p90 | p95 | Max |
| --- | ---: | ---: | ---: | ---: |
| GPU owner total | `230.54 ms` | `249.00 ms` | `257.61 ms` | `406.55 ms` |
| FFS cycle | `87.44 ms` | `93.90 ms` | `98.39 ms` | `267.79 ms` |
| staged EdgeTAM wall | `143.80 ms` | `161.04 ms` | `163.79 ms` | `179.86 ms` |
| staged EdgeTAM model-sum | `374.17 ms` | `415.69 ms` | `433.02 ms` | `484.52 ms` |
| staged EdgeTAM parallel efficiency | `2.60x` | `2.68x` | `2.69x` | `2.76x` |
| raw fusion | `8.08 ms` | `9.59 ms` | `10.44 ms` | `12.81 ms` |
| async filter total | `34.61 ms` | `37.60 ms` | `39.58 ms` | `216.51 ms` |
| render | `0.39 ms` | `0.57 ms` | `1.04 ms` | `4.59 ms` |

Per-camera EdgeTAM model latency in this run was about `125-126 ms` median per
camera when all three streams ran together. In the previous single-owner async
run, each EdgeTAM camera was about `31 ms` median. The parallel stage did overlap
work, but it made each EdgeTAM invocation much slower on the same RTX 5090 GPU.

## Conclusion

The requested `sequential + parallel` schedule is implemented as
`demo2.2-staged-parallel-5fps`, and the memory-resident staging pieces are in
place. The hardware result shows that three concurrent compiled EdgeTAM streams
are not the right optimization for this workload on one RTX 5090 Laptop GPU.

The next optimization should keep EdgeTAM serialized or micro-batched instead of
running three independent compiled streams concurrently. FFS CUDA buffer reuse
and EdgeTAM reusable pixel-value slots are still useful and can be retained.

## Artifacts

- Profile JSON: `docs/generated/demo2_2_staged_parallel_5fps_profile.json`
- Profile Markdown: `docs/generated/demo2_2_staged_parallel_5fps_profile.md`
