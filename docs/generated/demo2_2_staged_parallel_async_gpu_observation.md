# Demo 2.2 Staged Async GPU Observation

Status: profiled on RTX 5090 Laptop WSL.

## Command

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py \
  --experimental-staged-parallel \
  --duration-s 80 \
  --warmup-s 50 \
  --profile-json-output docs/generated/demo2_2_staged_parallel_async_gpu_profile.json \
  --debug
```

GPU sampling:

```bash
nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,memory.used,temperature.gpu \
  --format=csv \
  -lms 200 \
  > docs/generated/gpu_samples/demo2_2_staged_parallel_gpu_util_samples.csv
```

## Contract

This is the experimental staged async path:

```text
capture: 3 cameras @ 15 FPS
FFS: batch=1, sequential cam0 -> cam1 -> cam2
EdgeTAM: replicated model topology, cam0/cam1/cam2 parallel
GPU gate: off
filter: async latest-wins, render filtered only
```

This is not the current default Demo 2.2 path. The default remains single-owner
with FFS TensorRT batch=3.

## Result

| Metric | Value |
| --- | ---: |
| render FPS after warmup | `3.46` |
| raw fusion FPS after warmup | `3.46` |
| filter output FPS after warmup | `3.46` |
| capture group FPS after warmup | `13.23` |
| complete group ratio after warmup | `0.257` |
| GPU owner median | `283.66 ms` |
| FFS cycle median | `98.41 ms` |
| EdgeTAM cam0 median | `158.50 ms` |
| EdgeTAM cam1 median | `160.64 ms` |
| EdgeTAM cam2 median | `161.69 ms` |
| filter median | `42.43 ms` |
| raw fusion median | `9.90 ms` |
| time to first rendered group | `37.64 s` |

## GPU Utilization

For the estimated formal window from `t=50s` to `t=80s`:

| Metric | mean | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPU util | `28.89%` | `29.00%` | `32.00%` | `33.00%` | `38.00%` |
| memory util | `7.66%` | `8.00%` | `10.00%` | `10.00%` | `13.00%` |
| power draw, cleaned | `112.82 W` | `111.68 W` | `137.38 W` | `147.29 W` | `166.46 W` |
| GPU temperature | `60.72 C` | `60.50 C` | `65.00 C` | `65.00 C` | `66.00 C` |

WSL `nvidia-smi` caveat: this run produced some invalid power spikes outside
the formal window and reported `memory.used=0 MiB`, so memory-used values are
not trustworthy. The cleaned utilization and power samples are still useful as
coarse evidence that the GPU is not continuously saturated.

## Interpretation

The staged async path works functionally, but it is not the best current path.
It reaches only about `3.46 FPS` filtered render throughput, below both the
`5 FPS` practical target and the `15 FPS` camera input target.

The GPU does not appear to be full. During the formal window, utilization is
mostly around `29%` with p95 around `33%`, and power is bursty rather than
consistently pinned. The bottleneck is therefore not raw RTX 5090 compute
capacity. It is the staged execution shape:

```text
FFS batch=1 sequential work
+ three replicated HF EdgeTAM video trackers running in parallel
+ Python/session/kernel-launch overhead
+ occasional FFS/filter tail spikes
```

The parallel EdgeTAM stage overlaps three camera trackers, but each tracker is
still small and host-heavy. The result is lower pipeline FPS than the default
single-owner FFS batch=3 path, despite lower apparent GPU utilization.

## Recommendation

Keep staged parallel as an experiment/debug path, not default. For Demo 2.2,
the better baseline remains:

```text
single-owner GPU path
FFS TensorRT batch=3
shared HF EdgeTAM model with three independent camera sessions
async latest-wins filter
```

If we want true GPU saturation later, the next useful step is not more Python
threading. It would need lower-overhead EdgeTAM video tracking, such as a
batched memory/decoder scheduler or a non-HF runtime for the video tracker.
