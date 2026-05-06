# Demo 2.1 Shared FFS + Per-Camera EdgeTAM Validation

Date: 2026-05-06

## Contract

- `frame_by_frame_streaming=true`
- `offline_video_input_used=false`
- official depth source: `ffs`
- FFS checkpoint: `20-30-48`
- FFS valid iters: `4`
- FFS input shape: `480x864`
- FFS capture shape: `480x848`
- FFS padding policy: `pad_width_848_to_864`
- TensorRT builder optimization level: `5`
- EdgeTAM backend: HF `EdgeTAMVideo`
- EdgeTAM compile mode: `vision-reduce-overhead`
- professor-safe GPU gate: serialized, `max_concurrent=1`
- object filter: `enhanced-pt`
- controller filter: `pt-filter`
- object/controller union before filter: `false`

## Runtime Design

FFS worker:

- one shared worker
- one runner/context owner
- cam0/cam1/cam2 depth generation is sequential

EdgeTAM:

- three per-camera workers
- one streaming session per camera
- `obj_id=1 controller`
- `obj_id=2 object`

Fusion:

- strict `group_id` matching
- object fused cloud uses `enhanced-pt`
- controller fused cloud uses `pt-filter`
- object and controller are never unioned before filtering

GPU gate:

- one shared `GpuInferenceGate`
- `professor-safe` serializes shared FFS and EdgeTAM model forward
- debug and summary record gate wait for FFS and each EdgeTAM camera worker

## Presets

| Preset | Profile | Target | Render default | Intended use |
| --- | --- | ---: | --- | --- |
| `professor-safe` | `848x480@30` | 2 FPS | pointcloud | current no-hand object-only low-FPS FFS-quality fused demo |
| `visual-5fps` | `848x480@30` | 5 FPS | pointcloud | quality-preserving visual candidate, FFS depth + enhanced-PT, GPU gate max_concurrent=2 |
| `climb-5` | `848x480@30` | 5 FPS | none | headless performance climb |
| `climb-10` | `848x480@30` | 10 FPS | none | diagnostic stress test |
| `diagnostics` | `848x480@30` | 2 FPS | none | capture/EdgeTAM/FFS isolation |

The professor-safe preset prioritizes semantic correctness and startup stability over frame rate.

## Validation Commands

```bash
python -m py_compile demo_v2_1/realtime_three_view_masked_fused_pcd.py

python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --dry-run \
  --preset professor-safe \
  --track-mode controller-object \
  --ffs-worker-mode shared \
  --edgetam-worker-mode per-camera

python -m unittest -v \
  tests.test_demo_v2_1_three_view_fused_pcd_smoke \
  tests.test_check_all_smoke

conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py
```

## Status

The live runtime code path is implemented with lazy hardware/model imports so deterministic checks can validate the CLI and fusion contract without cameras.

## Hardware Smoke

WSL-5090 hardware smoke was run with three attached D455 cameras, old `calibrate.pkl`, `object-only`, `render-mode=none`, and `profile=848x480`.

The first 60 FPS attempt confirmed the three camera startup path but one D455 detached from WSL under load, so the startup smoke was rerun at 30 FPS while keeping the official FFS shape contract:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --track-mode object-only \
  --depth-source ffs \
  --init-mode sam31-first-frame \
  --object-prompt "stuffed animal" \
  --ffs-worker-mode shared \
  --ffs-schedule strict3-latest \
  --edgetam-worker-mode per-camera \
  --edgetam-model-topology replicated \
  --fusion-target-fps 2 \
  --profile 848x480 \
  --fps 30 \
  --render-mode none \
  --duration-s 90 \
  --debug \
  --profile-cuda-events
```

Summary:

- summary: `result/demo2_1_three_view_fused_pcd/session_20260506_010718_summary.json`
- capture group FPS: `1.997`
- shared FFS cycle FPS: `1.949`
- fusion FPS: `2.019`
- latest group id: `169`
- object points: `10022`
- controller points: `0`
- capture timeout skips: `2`

This validates that the three-view object-only live path can start and produce fused FFS-derived object PCD packets. The old calibration is acceptable only for startup testing; fused geometry correctness still needs a fresh calibration run.

### Serialized GPU Gate Hardware Smoke

After adding the shared GPU gate, a longer professor-safe object-only headless smoke was run:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset professor-safe \
  --track-mode object-only \
  --render-mode none \
  --duration-s 120 \
  --debug \
  --profile-cuda-events
```

Summary:

- summary: `result/demo2_1_three_view_fused_pcd/session_20260506_013134_summary.json`
- capture group FPS: `2.001`
- shared FFS cycle FPS: `2.004`
- fusion FPS: `1.470` over the full run, with steady debug output near the 2 FPS target after warmup
- complete fusion groups: `130`
- fusion timeout groups: `86`
- missing cam0 mask groups: `86`
- capture timeout skips: `1`
- object points: `9987`
- controller points: `0`
- GPU gate wait median:
  - FFS: `0.003 ms`
  - EdgeTAM cam0: `117.632 ms`
  - EdgeTAM cam1: `113.592 ms`
  - EdgeTAM cam2: `115.441 ms`

This validates that the serialized gate path can produce strict group-matched fused object PCD packets for all three cameras. The lower full-run fusion average is dominated by cold-start and missing-mask timeouts; the run should be at least 120 seconds when used as a hardware smoke. Controller-object should only be used when a hand is visible or saved three-camera hand masks are available.

## Current Run Table

| Run | Capture group FPS | FFS cycle FPS | EdgeTAM FPS | Fusion FPS | Render FPS | Notes |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| object-only target2 headless | 1.997 | 1.949 | ~2 FPS/cam after warmup | 2.019 | 0.0 | old calibration, no controller visible |
| object-only professor-safe serialized gate 120s | 2.001 | 2.004 | ~2 FPS/cam after warmup | 1.470 full-run, ~2 steady | 0.0 | 130 complete groups, 86 cam0 mask timeouts, old calibration |
| controller-object professor-safe | TBD | TBD | TBD | TBD | TBD | use only when a hand is visible or saved hand masks are available |
| controller-object climb-5 | TBD | TBD | TBD | TBD | TBD | profiling only |

## Visualization Smoke

A WSLg/Open3D pointcloud render smoke was run with the current no-hand object-only professor-safe path:

```bash
QQTT_WSLG_OPEN3D_FAST_EXIT=0 \
  ./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset professor-safe \
  --object-prompt "stuffed animal" \
  --duration-s 90 \
  --debug \
  --profile-cuda-events
```

The Open3D pointcloud window started, rendered fused object PCD packets, and exited with code `0`. During steady runtime, debug output reported `render_fps` around `1.9` to `2.1`, matching the professor-safe 2 FPS target. This first visualization check was log-based; the GUI fast-exit path now writes the normal session summary before exiting.

A second climb run tested whether the same visualization path can reach 5 FPS:

```bash
QQTT_WSLG_OPEN3D_FAST_EXIT=1 \
  ./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset climb-5 \
  --render-mode pointcloud \
  --object-prompt "stuffed animal" \
  --duration-s 75 \
  --debug \
  --profile-cuda-events \
  > docs/generated/demo2_1_visual_climb5_render_smoke.log 2>&1
```

Parsed steady-state metrics from `docs/generated/demo2_1_visual_climb5_render_smoke.log`:

| Metric | Median | Mean | Notes |
| --- | ---: | ---: | --- |
| capture group FPS | `5.00` | `4.95` | capture/group builder reaches target |
| shared FFS cycle FPS | `5.42` | `5.34` | FFS worker is not the primary 5 FPS blocker |
| EdgeTAM FPS cam0/cam1/cam2 | `4.99 / 3.36 / 4.99` | `4.76 / 3.40 / 4.69` | cam1 often waits behind the serialized GPU gate |
| fusion FPS | `2.48` | `2.52` | below 5 FPS target |
| render FPS | `2.48` | `2.53` | tracks fusion FPS, not an independent Open3D bottleneck |
| FFS cycle ms | `103.90` | `125.90` | occasional gate/wait spikes |
| fusion ms | `44.90` | `51.63` | includes semantic PCD postprocess |
| filter ms | `37.30` | `43.70` | enhanced object filter dominates fusion time; occasional ~230 ms spikes |
| object points | `9981` | `9981` | after object cap/filter |
| EdgeTAM model ms cam0/cam1/cam2 | `32.30 / 33.20 / 31.40` | `34.86 / 32.99 / 31.60` | compiled model compute is not the only limiter |
| GPU gate wait cam0/cam1/cam2 | `66.30 / 345.20 / 228.00` | `116.58 / 260.01 / 204.81` | serialized gate scheduling is the largest climb-5 limiter |

Conclusion:

- Professor-safe visualization succeeds at the intended low-FPS 2 FPS target.
- The current pointcloud renderer can display fused packets at the rate produced by fusion.
- For a 5 FPS visual demo, the current median render rate is short by about `2.5 FPS`, requiring roughly a `2x` throughput improvement.
- The main blockers are serialized GPU gate wait for the three EdgeTAM workers and semantic PCD postprocess cost, not WSLg/Open3D itself.

## 5 FPS Visual Candidate

The quality-preserving hypothesis was tested by keeping the exact same PCD quality path and only allowing two GPU inference workers through the gate at a time:

```bash
QQTT_WSLG_OPEN3D_FAST_EXIT=1 \
  ./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset climb-5 \
  --render-mode pointcloud \
  --object-prompt "stuffed animal" \
  --gpu-gate-mode limited \
  --gpu-gate-max-concurrent 2 \
  --duration-s 90 \
  --debug \
  --profile-cuda-events \
  > docs/generated/demo2_1_visual_climb5_render_gate2.log 2>&1
```

Parsed steady-state metrics from `docs/generated/demo2_1_visual_climb5_render_gate2.log`:

| Metric | Median | Mean | Notes |
| --- | ---: | ---: | --- |
| capture group FPS | `4.99` | `4.90` | group builder stays near target |
| shared FFS cycle FPS | `4.97` | `4.99` | FFS remains near 5 FPS |
| EdgeTAM FPS cam0/cam1/cam2 | `5.03 / 5.04 / 5.01` | `4.98 / 4.96 / 4.93` | gate2 removes the cam1 starvation seen with serialized gate |
| fusion FPS | `4.86` | `4.69` | near the 5 FPS target |
| render FPS | `4.85` | `4.69` | tracks fusion; p90 render FPS was `5.19` |
| FFS cycle ms | `134.70` | `139.72` | FFS is slower than the serialized run but still supplies target-rate groups |
| fusion ms | `42.80` | `48.38` | includes object enhanced-PT |
| filter ms | `36.20` | `40.94` | unchanged enhanced object filter; occasional spike remains |
| object points | `9982` | `8298` | intermittent point-count drops still need visual review |
| EdgeTAM model ms cam0/cam1/cam2 | `62.80 / 63.00 / 60.50` | `63.89 / 66.15 / 61.12` | model kernels slow under gate2 contention, but total throughput improves |
| GPU gate wait ffs/cam0/cam1/cam2 | `0.00 / 89.30 / 59.50 / 51.70` | `0.06 / 88.95 / 62.98 / 54.12` | much lower than serialized gate waits |

Conclusion:

- The 5 FPS candidate does not change the PCD quality contract: FFS-derived depth, object `enhanced-pt`, controller `pt-filter`, and no native RealSense fallback.
- `gpu_gate_max_concurrent=2` is the first quality-preserving change that brings WSLg/Open3D visualization close to 5 FPS.
- Remaining risk is stability, not renderer capacity: filter spikes and intermittent low object point counts still need visual inspection before replacing `professor-safe`.
- This candidate is now exposed as the `visual-5fps` preset.

## visual-5fps Profiling Schema

`realtime_three_view_masked_fused_pcd.py` now supports explicit profiling flags that are disabled by default:

```bash
./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset visual-5fps \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --duration-s 120 \
  --debug \
  --profile-pipeline \
  --profile-filter \
  --profile-visualization \
  --profile-gpu-gate \
  --profile-warmup-exclude-s 20 \
  --profile-json-output docs/generated/demo2_1_visual5fps_profile_object_only.json
```

The JSON profile records one row per `group_id` with:

- capture frame sequence and group build timing
- per-camera EdgeTAM gate wait, model, postprocess, and total timing
- shared FFS gate wait, cycle timing, per-camera FFS and align timing
- fusion wait, raw PCD build timing, object enhanced-PT timing, controller PT-filter timing, point counts
- Open3D geometry update and render callback timing

The companion Markdown report summarizes full-run and warmup-excluded medians, p90, p95, max values, target FPS deficit, bottleneck class, and top slowest object enhanced-PT groups.

A 60s object-only smoke profile was run after adding this schema:

```bash
QQTT_WSLG_OPEN3D_FAST_EXIT=1 \
  ./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset visual-5fps \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --duration-s 60 \
  --debug \
  --profile-pipeline \
  --profile-filter \
  --profile-visualization \
  --profile-gpu-gate \
  --profile-warmup-exclude-s 20 \
  --profile-json-output docs/generated/demo2_1_visual5fps_profile_object_only.json \
  > docs/generated/demo2_1_visual5fps_profile_object_only.log 2>&1
```

Outputs:

- `docs/generated/demo2_1_visual5fps_profile_object_only.json`
- `docs/generated/demo2_1_visual5fps_profile_object_only.md`
- `docs/generated/demo2_1_visual5fps_profile_object_only.log`

Warmup-excluded summary from this smoke:

| Metric | Value |
| --- | ---: |
| render FPS | `4.56` |
| fusion FPS | `4.56` |
| target deficit | `0.44 FPS` |
| bottleneck class | `upstream_supply` |
| FFS cycle median / p95 | `123.31 / 195.44 ms` |
| EdgeTAM model median cam0/cam1/cam2 | `61.96 / 58.67 / 58.67 ms` |
| EdgeTAM gate wait median cam0/cam1/cam2 | `50.80 / 58.92 / 0.00 ms` |
| object enhanced-PT median / p95 / max | `33.78 / 41.36 / 44.80 ms` |
| Open3D render callback median / p95 | `0.51 / 3.27 ms` |
| Open3D object geometry update median / p95 | `0.03 / 2.24 ms` |

Interpretation:

- Open3D/WSLg is not the current limiter in this profile; the render callback is sub-4ms at p95.
- This run did not reproduce the earlier ~230ms enhanced-PT spike; object enhanced-PT p95 stayed near `41 ms`.
- The remaining deficit is upstream supply: FFS/EdgeTAM scheduling and missing mask groups during startup/initialization.
- The last steady debug window reached about `5.0 FPS`, so a longer 120s profile with a larger warmup exclusion is needed before changing quality settings.

Recommended current-lab professor-facing command when no hand/controller is visible:

```bash
./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset professor-safe \
  --object-prompt "stuffed animal" \
  --duration-s 120 \
  --debug
```

5 FPS visual candidate command:

```bash
./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset visual-5fps \
  --object-prompt "stuffed animal" \
  --duration-s 120 \
  --debug \
  --profile-cuda-events
```

Controller-object command for when a hand is visible:

```bash
./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset professor-safe \
  --track-mode controller-object \
  --controller-prompt "hand" \
  --object-prompt "stuffed animal" \
  --duration-s 120 \
  --debug
```
