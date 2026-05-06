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
| `professor-safe` | `848x480@30` | 2 FPS | pointcloud | low-FPS FFS-quality fused demo |
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

Recommended current-lab professor-facing command when no hand/controller is visible:

```bash
./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset professor-safe \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --duration-s 120 \
  --debug
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
