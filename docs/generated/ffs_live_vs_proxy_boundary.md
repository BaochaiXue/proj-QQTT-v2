# FFS Live vs Proxy Boundary

- Date: `2026-05-01`
- Scope: current status label for FFS performance reports

## Status Table

| Result family | Current status | Config | Measured result | Source |
| --- | --- | --- | --- | --- |
| Live PyTorch 3-camera | RED: not realtime | `20-30-48`, `valid_iters=4`, `scale=0.5`, `max_disp=192`, 3 D455 cameras | about `22.6` aggregate FFS FPS, about `7.5` FPS per camera | `docs/generated/ffs_live_3cam_benchmark_validation.md` |
| Static replay / TensorRT proxy | Target basically reached | `20-30-48`, `valid_iters=4`, real `848x480` inputs padded to `864x480`, `builderOptimizationLevel=5` | level-5 TRT static-image table records `14.00 ms` mean, about `71.43 FPS` for the single-pair proxy measurement | `result/ffs_trt_static_rounds_848x480_pad864_builderopt5_rtx5090_laptop_20260428/report.md` |
| Concurrent 3-view static replay proxy | Offline proxy only | concurrent `3` subprocess workers over static aligned rounds | best `20-30-48 / scale=0.5 / valid_iters=4` two-stage FP16 result recorded `35.308` overall mean FPS | `docs/generated/ffs_static_replay_matrix_concurrent3view_validation.md` |

## Reporting Rule

Use the live PyTorch 3-camera benchmark for live realtime claims. Use the static replay / TensorRT proxy reports for proxy throughput and artifact-default claims. Do not combine the proxy numbers with the live PyTorch result or describe the PyTorch live path as realtime.

## Current Operator Wording

- "Live PyTorch 3-camera FFS is not realtime on the RTX 5090 laptop; best measured `scale=0.5` reached about `22.6` aggregate FFS FPS and about `7.5` FPS per camera."
- "`20-30-48 / valid_iters=4 / 848x480 -> 864x480 / builderOptimizationLevel=5` is the current TensorRT static replay / proxy target and has basically reached that proxy goal."
