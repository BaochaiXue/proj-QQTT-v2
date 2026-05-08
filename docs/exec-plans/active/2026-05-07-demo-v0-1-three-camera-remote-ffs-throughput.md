# Demo v0.1 Three-Camera Remote FFS Throughput Benchmark

## Goal

Measure the transport and remote FFS ceiling without SAM3.1, EdgeTAM, masks, PCD
filtering, or Open3D in the loop.

The benchmark should answer:

```text
Can WSL-5090 send three real RealSense IR pairs at 15 FPS/camera,
can Ubuntu-4090 run FFS TensorRT and return full depth,
and what aggregate completed camera-FPS does the current protocol sustain?
```

## Scope

- Add a benchmark-only mode to `services/ffs_remote/ffs_depth_client.py`:
  `--three-camera-real-ir-depth-benchmark`.
- Capture real IR1/IR2 Y8 plus color calibration metadata from up to three local
  WSL-5090 RealSense D455 cameras.
- Send one full-depth request per camera per group.
- Use independent REQ socket workers for asynchronous client-side send/receive.
- Measure per-camera FPS, aggregate camera-FPS, complete group FPS, request and
  response KB, RTT, server FFS, server align, server total, and queue pressure.

## Non-Goals

- Do not use synthetic IR as a formal result.
- Do not require masks or SAM3.1.
- Do not change Demo 2 runtime behavior.
- Do not change the Ubuntu-4090 server implementation in this change.
- Do not claim the current REP server is fully asynchronous; this benchmark
  measures async client pressure against the current server.

## Required Server Mode

Ubuntu-4090 must run:

```text
return_type = depth_u16
response_compression = lz4
```

`masked_uv_depth` is not valid for this v0.1 test because it requires a mask and
returns sparse depth.

## Target

```text
single camera target: >=45 FPS
three camera target: >=15 FPS/camera, aggregate >=45 camera-FPS
group target: >=15 complete 3-camera groups/s
```

## Validation

- `python -m py_compile services/ffs_remote/ffs_depth_client.py`
- `conda run --no-capture-output -n SAM21-max python -m unittest -v tests.test_realtime_single_camera_pointcloud_smoke`
- Hardware benchmark when 4090 is in `depth_u16/lz4` mode:
  `--three-camera-real-ir-depth-benchmark --fps 15 --inflight 6 --return-type depth_u16 --compress lz4`
