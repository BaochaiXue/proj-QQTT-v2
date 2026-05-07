# Demo 2 Remote FFS Real-IR Depth Benchmark

Date: 2026-05-07

## Goal

Stop treating synthetic echo as a remote FFS performance handoff. Add a WSL-side
client benchmark that captures real RealSense D455 IR left/right frames, sends
them through the existing remote FFS protocol, receives real depth, and reports
RTT/server/depth/payload metrics.

## Scope

- Extend `services/ffs_remote/ffs_depth_client.py` with
  `--real-ir-depth-benchmark`.
- Capture real `rs.stream.infrared` 1/2 `Y8` frames, plus color stream metadata
  for color-aligned depth requests.
- Reuse `FfsRemoteDepthClient.request_depth_color_m(...)` and existing request
  compression / return type settings.
- Save first returned depth `.npy` and preview image when requested.
- Update generated remote FFS reports with the new synthetic-vs-real-IR
  validation boundary.

## Non-Goals

- Do not change Demo 2 FFS model settings or EdgeTAM behavior.
- Do not start or manage the Ubuntu-4090 server from WSL.
- Do not use synthetic echo as a formal remote FFS pass/fail criterion.
- Do not change formal recording/alignment code.

## Validation

- `python -m py_compile services/ffs_remote/ffs_depth_client.py`
- `conda run --no-capture-output -n SAM21-max python -m unittest -v tests.test_realtime_single_camera_pointcloud_smoke`
- Optional hardware run once the Ubuntu-4090 strict real-depth server is ready:
  `--real-ir-depth-benchmark --serial 239222300412 --compress lz4 --return-type depth_u16`
