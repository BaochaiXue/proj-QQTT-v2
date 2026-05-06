# Demo 2 Local FFS Professor Speed Polish

This pass prepares the local RTX 5090 Laptop Demo 2 path for a low-FPS but
usable professor-facing realtime demo.

The formal path remains:

- depth source: FFS-derived depth
- FFS contract: `20-30-48`, `valid_iters=4`, `848x480 -> pad 864x480`,
  `builderOptimizationLevel=5`
- tracker: HF EdgeTAMVideo frame-by-frame streaming
- EdgeTAM compile mode: `vision-reduce-overhead`
- dtype: `bfloat16`
- no native RealSense depth as the formal output

## Code Changes

- Added `--demo-preset local-ffs-professor`.
- The preset is accepted only with `--depth-source ffs`.
- The preset keeps the formal FFS and EdgeTAM contract unchanged.
- The preset lowers default render pressure:
  - `--pcd-max-points`: `60000 -> 20000`
  - `--point-size`: `2.0 -> 2.5`
  - `--latency-target-ms`: `80 -> 120`
- Explicit CLI values still override the preset defaults.
- Object-only tracking now skips the empty controller PCD backprojection path.

## Tomorrow Command

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2/realtime_masked_edgetam_pcd.py \
  --serial 239222300412 \
  --profile 848x480 \
  --fps 60 \
  --depth-source ffs \
  --demo-preset local-ffs-professor \
  --init-mode sam31-first-frame \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --pcd-mode masked \
  --render-mode pointcloud \
  --compile-mode vision-reduce-overhead \
  --dtype bfloat16 \
  --depth-min-m 0.2 \
  --depth-max-m 1.5 \
  --pcd-color-mode rgb \
  --debug \
  --profile-cuda-events
```

If WSLg/Open3D render is the visible bottleneck, add:

```bash
--render-every-n 2
```

That keeps the formal FFS-derived compute path intact and only reduces display
update pressure.

## No-Render Compute Profile

Command:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2/realtime_masked_edgetam_pcd.py \
  --serial 239222300412 \
  --profile 848x480 \
  --fps 60 \
  --depth-source ffs \
  --demo-preset local-ffs-professor \
  --init-mode sam31-first-frame \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --pcd-mode masked \
  --render-mode none \
  --compile-mode vision-reduce-overhead \
  --dtype bfloat16 \
  --depth-min-m 0.2 \
  --depth-max-m 1.5 \
  --pcd-color-mode rgb \
  --duration-s 45 \
  --debug \
  --profile-cuda-events
```

Raw log:

- `docs/generated/demo2_local_ffs_professor_preset_no_render_20260505.txt`

Stable median results:

| Metric | Median | P90 |
| --- | ---: | ---: |
| capture FPS | 53.70 | 56.22 |
| segmentation FPS | 25.00 | 25.90 |
| PCD FPS | 25.20 | 25.80 |
| EdgeTAM mask total | 35.74 ms | 42.60 ms |
| EdgeTAM CUDA event model | 34.60 ms | 40.93 ms |
| FFS | 21.31 ms | 24.45 ms |
| FFS align | 6.18 ms | 7.27 ms |
| PCD total | 29.48 ms | 33.28 ms |
| E2E latency | 84.41 ms | 96.55 ms |
| object points | 20.0k | 20.0k |

## Interpretation

The local formal FFS path remains a roughly 25 FPS quality-mode demo on this
laptop when FFS, HF EdgeTAM, and masked PCD all run together on the RTX 5090
Laptop. The preset is not meant to hide that compute limit. It makes the demo
more stable for Open3D/WSLg by bounding the displayed masked point cloud and by
removing avoidable object-only PCD work.

Native RealSense depth remains a fast fallback/debug mode only; it is not the
professor-facing formal Demo 2 output.

## Pointcloud Render Smoke

Command:

```bash
./demo_v2/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2/realtime_masked_edgetam_pcd.py \
  --serial 239222300412 \
  --profile 848x480 \
  --fps 60 \
  --depth-source ffs \
  --demo-preset local-ffs-professor \
  --init-mode sam31-first-frame \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --pcd-mode masked \
  --render-mode pointcloud \
  --compile-mode vision-reduce-overhead \
  --dtype bfloat16 \
  --depth-min-m 0.2 \
  --depth-max-m 1.5 \
  --pcd-color-mode rgb \
  --duration-s 30 \
  --debug \
  --profile-cuda-events
```

Raw log:

- `docs/generated/demo2_local_ffs_professor_preset_pointcloud_20260505.txt`

Stable median results:

| Metric | Median | P90 |
| --- | ---: | ---: |
| capture FPS | 54.20 | 57.26 |
| segmentation FPS | 23.80 | 24.54 |
| PCD FPS | 23.70 | 24.54 |
| render FPS | 23.70 | 24.57 |
| EdgeTAM mask total | 40.97 ms | 43.96 ms |
| EdgeTAM CUDA event model | 39.25 ms | 42.32 ms |
| FFS | 22.62 ms | 23.46 ms |
| FFS align | 6.60 ms | 8.47 ms |
| PCD total | 31.08 ms | 33.36 ms |
| render update | 0.03 ms | 0.06 ms |
| E2E latency | 89.50 ms | 108.49 ms |
| object points | 20.0k | 20.0k |

The pointcloud run stayed in the same low-to-mid 20 FPS band as the compute
path. In this run, Open3D update time was not the bottleneck; the dominant cost
remained local GPU contention between FFS and EdgeTAM plus masked PCD assembly.

## 2026-05-06 Professor Demo Profiling

This run repeats the formal local RTX 5090 Laptop path for the professor-facing
demo:

- `--depth-source ffs`
- `--demo-preset local-ffs-professor`
- `--track-mode object-only`
- `--compile-mode vision-reduce-overhead`
- `--dtype bfloat16`
- `--pcd-max-points 20000` from the preset

Raw logs:

- `docs/generated/demo2_local_ffs_professor_no_render_profile_20260506.txt`
- `docs/generated/demo2_local_ffs_professor_pointcloud_profile_20260506.txt`

No-render stable median results:

| Metric | Median | P90 |
| --- | ---: | ---: |
| capture FPS | 53.70 | 56.80 |
| segmentation FPS | 25.65 | 26.65 |
| PCD FPS | 25.80 | 26.55 |
| EdgeTAM mask total | 36.20 ms | 39.61 ms |
| EdgeTAM CUDA event model | 34.59 ms | 38.13 ms |
| FFS | 21.03 ms | 22.38 ms |
| FFS align | 6.20 ms | 8.10 ms |
| PCD total | 29.02 ms | 31.85 ms |
| E2E latency | 82.03 ms | 99.87 ms |
| object points | 20.0k | 20.0k |

Pointcloud stable median results:

| Metric | Median | P90 |
| --- | ---: | ---: |
| capture FPS | 55.70 | 58.80 |
| segmentation FPS | 22.00 | 23.68 |
| PCD FPS | 22.00 | 23.72 |
| render FPS | 22.00 | 23.56 |
| EdgeTAM mask total | 43.27 ms | 49.22 ms |
| EdgeTAM CUDA event model | 41.29 ms | 47.38 ms |
| FFS | 21.64 ms | 23.75 ms |
| FFS align | 6.39 ms | 9.64 ms |
| PCD total | 30.63 ms | 35.46 ms |
| render update | 0.03 ms | 0.05 ms |
| E2E latency | 93.50 ms | 108.10 ms |
| object points | 20.0k | 20.0k |

Interpretation:

- The no-render compute path is currently about `25.8 FPS`.
- The live pointcloud demo path is currently about `22.0 FPS`.
- Open3D update time is not the limiting factor in this run.
- The bottleneck remains local GPU contention between FFS and HF EdgeTAM,
  visible in the higher EdgeTAM CUDA event time during the pointcloud run.
