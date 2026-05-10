# 2026-05-10 Demo 2.2 15 FPS Camera Input Rerun

Status: completed.

## Goal

Rerun the Demo 2.2 single-owner async-filter path with RealSense input raised
from `5 FPS` to `15 FPS`, while keeping the fused/render target at `5 FPS`.

The purpose is to test whether the previous `2.64 FPS` result was limited by
three-camera coherent group supply rather than GPU compute.

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

## Validation

- PASS: dry-run contract kept `gpu_pipeline=single-owner`, `fps=15`, and
  `fusion_target_fps=5.0`.
- PASS: 120s hardware profile completed.
- PASS: generated benchmark note comparing:
  - previous Demo 2.2 async-filter `5 FPS` camera input profile
  - new Demo 2.2 async-filter `15 FPS` camera input profile

## Result

- Previous 5 FPS camera input render FPS: `2.64`.
- New 15 FPS camera input render FPS: `4.52`.
- Complete fused group ratio improved from `0.964` to `0.992`.
- Capture skew median improved from `43.00 ms` to `19.12 ms`.
- Demo 2.2 pass threshold remains `4.8 FPS`; result is still FAIL, but only by
  `0.28 FPS`.
