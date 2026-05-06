# Demo 2 Depth Source FPS Comparison

Date: 2026-05-05

## Goal

Compare current local 5090 Laptop Demo 2 FPS for:

1. formal FFS-derived depth:
   `20-30-48`, `valid_iters=4`, `848x480 -> 864x480`,
   `builderOptimizationLevel=5`, `--depth-source ffs`
2. fallback/debug native depth:
   `--depth-source realsense`

Both runs used the same tracking path:

```text
D455 serial: 239222300412
profile: 848x480@60
init: SAM3.1 first-frame prompt
track mode: object-only
object prompt: stuffed animal
EdgeTAM: HF EdgeTAMVideo
compile mode: vision-reduce-overhead
dtype: bfloat16
pcd mode: masked
render mode: none
profile CUDA events: true
```

This comparison intentionally used `--render-mode none` to isolate capture,
tracking, depth, and masked PCD from WSLg/Open3D render overhead.

## Raw Logs

```text
docs/generated/demo2_depth_source_compare_ffs_no_render.txt
docs/generated/demo2_depth_source_compare_realsense_no_render.txt
```

## Commands

Formal FFS-derived path:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2/realtime_masked_edgetam_pcd.py \
  --serial 239222300412 \
  --profile 848x480 \
  --fps 60 \
  --depth-source ffs \
  --init-mode sam31-first-frame \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --pcd-mode masked \
  --render-mode none \
  --compile-mode vision-reduce-overhead \
  --dtype bfloat16 \
  --depth-min-m 0.2 \
  --depth-max-m 1.5 \
  --pcd-max-points 60000 \
  --pcd-color-mode rgb \
  --duration-s 60 \
  --debug \
  --profile-cuda-events
```

Fallback/debug native RealSense path:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2/realtime_masked_edgetam_pcd.py \
  --serial 239222300412 \
  --profile 848x480 \
  --fps 60 \
  --depth-source realsense \
  --init-mode sam31-first-frame \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --pcd-mode masked \
  --render-mode none \
  --compile-mode vision-reduce-overhead \
  --dtype bfloat16 \
  --depth-min-m 0.2 \
  --depth-max-m 1.5 \
  --pcd-max-points 60000 \
  --pcd-color-mode rgb \
  --duration-s 60 \
  --debug \
  --profile-cuda-events
```

## Stable Median Results

Stable rows are debug rows with nonzero `pcd_fps` and nonzero object points;
the first nonzero transition row is excluded.

| depth path | formal status | samples | capture FPS | seg FPS | pcd FPS | EdgeTAM CUDA event ms | FFS ms | FFS align ms | PCD total ms | e2e latency ms | object pts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FFS 20-30-48 iters4 864x480 opt5 + compiled EdgeTAM | official quality path | 40 | 53.70 | 25.20 | 25.20 | 35.21 | 21.53 | 6.04 | 29.58 | 84.22 | 27,525 |
| RealSense native depth + compiled EdgeTAM | fallback/debug only | 25 | 58.70 | 48.80 | 48.90 | 16.05 | 0.00 | 0.00 | 2.88 | 33.90 | 26,623 |

## Interpretation

The RealSense fallback path is about `1.94x` faster in no-render PCD FPS:

```text
48.90 FPS / 25.20 FPS = 1.94x
```

It is faster because it removes the local FFS TensorRT workload. The EdgeTAM
CUDA-event model time drops from `35.21 ms` with local FFS enabled to `16.05 ms`
with native RealSense depth. This confirms the previous profiling conclusion:
local FFS and EdgeTAM contend for the same 5090 Laptop GPU.

However, the RealSense result is not an official Demo 2 quality result. Formal
Demo 2 output must be FFS-derived. The native RealSense path remains useful for
debugging UI, camera, SAM3.1 first-frame init, EdgeTAM tracking, and Open3D
render without the local FFS GPU load.

Current recommendation:

```text
professor-facing quality baseline:
  use --depth-source ffs
  expect about 25 FPS no-render on this run

fast fallback/debug:
  use --depth-source realsense
  expect about 49 FPS no-render on this run

speed + FFS semantics:
  continue remote/sparse FFS main-path work so PCD remains FFS-derived
  while moving FFS compute off the local 5090 GPU
```
