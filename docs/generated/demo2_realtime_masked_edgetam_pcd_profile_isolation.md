# Demo 2 Realtime Masked EdgeTAM PCD Profiling Isolation

Date: 2026-05-03

Hardware/runtime path:

- D455 serial: `239222300412`
- Capture: `848x480@60`
- EdgeTAM: `yonigozlan/EdgeTAM-hf`, `bfloat16`, `vision-reduce-overhead`
- Default live sync policy: `profile_sync_enabled=0`
- CUDA event model timing: enabled for EdgeTAM isolation runs

## Commands

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2/realtime_masked_edgetam_pcd.py \
  --serial 239222300412 --profile 848x480 --fps 60 \
  --depth-source none --track-mode none --pcd-mode none --render-mode none \
  --duration-s 8 --debug \
  2>&1 | tee docs/generated/demo2_profile_A_capture_only.txt

conda run --no-capture-output -n demo_2_max \
  python demo_v2/realtime_masked_edgetam_pcd.py \
  --serial 239222300412 --profile 848x480 --fps 60 \
  --depth-source none --init-mode sam31-first-frame \
  --track-mode object-only --object-prompt "stuffed animal" \
  --pcd-mode none --render-mode none \
  --compile-mode vision-reduce-overhead --dtype bfloat16 \
  --duration-s 45 --debug --profile-cuda-events \
  2>&1 | tee docs/generated/demo2_profile_B_edgetam_only.txt

conda run --no-capture-output -n demo_2_max \
  python demo_v2/realtime_masked_edgetam_pcd.py \
  --serial 239222300412 --profile 848x480 --fps 60 \
  --depth-source ffs --track-mode none --pcd-mode none --render-mode none \
  --duration-s 20 --debug \
  2>&1 | tee docs/generated/demo2_profile_C_ffs_only.txt

conda run --no-capture-output -n demo_2_max \
  python demo_v2/realtime_masked_edgetam_pcd.py \
  --serial 239222300412 --profile 848x480 --fps 60 \
  --depth-source ffs --init-mode sam31-first-frame \
  --track-mode object-only --object-prompt "stuffed animal" \
  --pcd-mode masked --render-mode none \
  --compile-mode vision-reduce-overhead --dtype bfloat16 \
  --depth-min-m 0.2 --depth-max-m 1.5 \
  --pcd-max-points 60000 --pcd-color-mode rgb \
  --duration-s 60 --debug --profile-cuda-events \
  2>&1 | tee docs/generated/demo2_profile_F_edgetam_ffs_pcd_headless.txt
```

## Stable Medians

| Run | capture FPS | seg FPS | depth/pcd FPS | wall model ms | CUDA event model ms | FFS ms | FFS align ms | PCD ms | e2e ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A capture only | 63.95 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 9.08 |
| B EdgeTAM only | 61.60 | 55.00 | 0.00 | 14.06 | 13.99 | 0.00 | 0.00 | 0.00 | 23.26 |
| C FFS only | 58.85 | 0.00 | 32.25 | 0.00 | 0.00 | 18.04 | 12.00 | 0.00 | 39.84 |
| F EdgeTAM + FFS + PCD, no render | 58.30 | 25.70 | 25.70 | 33.54 | 35.94 | 19.66 | 6.20 | 27.85 | 76.66 |

PCD breakdown in run F:

| Metric | Median ms |
| --- | ---: |
| mask/depth intersection | 0.42 |
| selected pixel extraction | 0.64 |
| backproject | 0.42 |
| color gather | 0.22 |

## Interpretation

EdgeTAM-only returns to the expected fast path: wall model median `14.06 ms`
and CUDA-event model median `13.99 ms`. This confirms Demo 2 is using the
compiled HF EdgeTAM streaming path correctly when FFS is not running.

With FFS depth and masked PCD enabled, EdgeTAM model timing rises to
`33.54 ms` wall / `35.94 ms` CUDA event. Since wall and CUDA-event timing rise
together, this is not just Python wall-clock or device-wide synchronize
pollution. It is real GPU contention between EdgeTAM and the FFS/PCD path.

The masked PCD CPU work itself is small. The `pcd_ms` number mostly includes
FFS inference and color alignment, not just point selection/backprojection.
