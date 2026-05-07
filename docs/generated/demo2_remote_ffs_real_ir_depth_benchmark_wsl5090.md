# Demo 2 Remote FFS Real-IR Depth Benchmark

Date: 2026-05-07

## Status

Implemented and ran WSL-5090 client support for the formal remote FFS benchmark:

```text
input_source = real RealSense D455 IR left/right
request = compressed IR pair over ZeroMQ multipart frames
server = Ubuntu-4090 strict TensorRT FFS depth server
response = real full-frame depth_u16 or depth_float_m
```

This replaces the previous synthetic echo performance handoff. Synthetic echo is
now only TCP/protocol sanity.

2026-05-07 real-IR depth result:

```text
PASS: WSL-5090 captured real D455 IR left/right frames, sent them to the
Ubuntu-4090 strict TensorRT FFS server, received real nonzero depth_u16, and
saved first-depth artifacts.
```

## Code Path

Updated:

```text
services/ffs_remote/ffs_depth_client.py
```

New CLI mode:

```text
--real-ir-depth-benchmark
```

Important flags:

```text
--serial <D455 serial>
--profile 848x480
--fps 30
--compress none|lz4|zstd|png
--return-type depth_u16|depth_float_m
--save-first-depth-preview
--output-dir docs/generated
```

The mode opens RealSense streams:

```text
rs.stream.infrared index 1, Y8
rs.stream.infrared index 2, Y8
rs.stream.color, BGR8
```

It extracts:

```text
K_ir_left
K_color
T_ir_left_to_color
ir_baseline_m
depth_scale_m_per_unit
```

and sends them through `FfsRemoteDepthClient.request_depth_color_m(...)`.

## First Formal Benchmark Command

Run after Ubuntu-4090 has stopped echo-only mode and started the strict FFS
server with:

```text
--return depth_u16
--compress lz4
```

WSL-5090 command:

```bash
cd /home/zhangxinjie/proj-QQTT-v2

conda run --no-capture-output -n demo_2_max \
  python services/ffs_remote/ffs_depth_client.py \
  --endpoint tcp://192.168.0.162:7001 \
  --real-ir-depth-benchmark \
  --serial 239222300412 \
  --profile 848x480 \
  --fps 30 \
  --duration-s 20 \
  --timeout-ms 5000 \
  --compress lz4 \
  --return-type depth_u16 \
  --save-first-depth-preview \
  --debug
```

If `192.168.0.162` is not the active route, use:

```text
tcp://128.59.19.35:7001
```

## Run Results

Environment note:

```text
The first lz4 run failed before network send because demo_2_max did not have
the Python lz4 package. Installed lz4==4.4.5 in demo_2_max, then reran the
same real-IR benchmark successfully.
```

Both endpoints were tested with:

```text
input_source = real RealSense IR left/right
serial = 239222300412
profile = 848x480
fps = 30
request_compression = lz4
response_compression = lz4
return_type = depth_u16
```

| Endpoint | Sent | OK | Failed | Reply FPS | RTT p50 / p90 / p95 ms | Server FFS / Align / Total p50 ms | Request KB | Response KB | Depth Nonzero Mean | First Depth Artifacts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `tcp://128.59.19.35:7001` | 181 | 181 | 0 | 8.40 | 99.56 / 105.32 / 107.91 | 13.34 / 4.76 / 19.46 | 619.63 | 332.40 | 362725.07 | `docs/generated/demo2_real_ir_remote_depth_20260507_134437_frame000000_depth_m.npy`, `docs/generated/demo2_real_ir_remote_depth_20260507_134437_frame000000_depth_preview.png` |
| `tcp://192.168.0.162:7001` | 255 | 255 | 0 | 11.83 | 68.99 / 94.29 / 139.74 | 13.54 / 4.38 / 19.37 | 620.26 | 332.53 | 362996.32 | `docs/generated/demo2_real_ir_remote_depth_20260507_134511_frame000000_depth_m.npy`, `docs/generated/demo2_real_ir_remote_depth_20260507_134511_frame000000_depth_preview.png` |

Decision:

```text
Best measured endpoint for the real-IR lz4/lz4 depth benchmark:
  tcp://192.168.0.162:7001

Reason:
  Higher reply FPS and lower p50 RTT in the real payload benchmark, despite
  earlier ping results suggesting the other route.

Formal handoff:
  pass for real IR -> remote FFS TensorRT -> real depth_u16 semantics.

Current bottleneck:
  not server FFS compute. The server total median is about 19 ms, while the
  end-to-end RTT median is about 69-100 ms depending on endpoint.
```

## Metrics Printed

The summary line prints:

```text
sent
ok
failed
capture_miss
reply_fps
rtt_ms_p50 / p90 / p95
server_ffs_ms_p50
server_align_ms_p50
server_total_ms_p50
request_kb_mean
response_kb_mean
depth_nonzero_count_mean
request_compression
response_compression
return_type
depth_shapes
first_depth_npy_path
first_depth_preview_path
```

## Pass Criteria

Formal remote FFS handoff should be judged from real-IR remote-depth only:

```text
input_source = real RealSense IR
return_type = depth_u16 or depth_float_m
server_status = ok
depth_nonzero_count_mean > 0
failed = 0
reply_fps >= target
rtt_ms_p50 stable
server_ffs_ms_p50 stable
server_align_ms_p50 stable
request/response KB recorded
first depth preview/npy saved
```

## Validation

```text
python -m py_compile services/ffs_remote/ffs_depth_client.py
conda run --no-capture-output -n SAM21-max python -m unittest -v tests.test_realtime_single_camera_pointcloud_smoke
```

Result:

```text
py_compile: pass
unittest: pass, 56 tests, 3 skipped
```
