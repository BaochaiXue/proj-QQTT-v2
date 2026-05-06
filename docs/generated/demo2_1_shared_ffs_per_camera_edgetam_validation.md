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

## Validation Commands

```bash
python -m py_compile demo_v2_1/realtime_three_view_masked_fused_pcd.py

python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --dry-run \
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
