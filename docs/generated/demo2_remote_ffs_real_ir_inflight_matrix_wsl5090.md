# Demo 2 Remote FFS Real-IR Inflight Matrix

Date: 2026-05-07

## Target

```text
single camera realtime: 45 FPS
three camera realtime: 15 FPS per camera, aggregate 45 camera-FPS
```

Semantic requirement:

```text
input_source = real RealSense D455 IR left/right Y8 from WSL-5090
server = Ubuntu-4090 strict FFS TensorRT server
output = real full-frame depth_u16
synthetic echo = protocol sanity only, not formal performance evidence
```

## Method

`services/ffs_remote/ffs_depth_client.py` now supports benchmark-only
multi-inflight real-IR requests:

```text
--inflight N
--drop-stale-replies
```

Implementation:

```text
N independent ZeroMQ REQ sockets
each socket has at most one in-flight request
main thread captures real D455 IR frames
worker sockets submit real IR -> remote FFS -> real depth_u16
out-of-order older replies can be counted as stale latest-wins drops
```

This does not change Demo 2 runtime behavior yet. It is a transport benchmark
to determine whether full `depth_u16` can meet realtime throughput.

## Fixed Settings

```text
endpoint = tcp://192.168.0.162:7001
serial = 239222300412
profile = 848x480
target_capture_fps = 60
duration = 30 s
request_compression = lz4
response_compression = lz4
return_type = depth_u16
timeout_ms = 5000
```

## Results

| Inflight | Submitted FPS | Completed FPS | Latest Reply FPS | OK / Failed | Stale Replies | Timeout | RTT p50 / p90 / p95 ms | Server Total p50 ms | Request KB | Response KB | Depth Nonzero Mean | Verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 9.40 | 9.40 | 9.40 | 300 / 0 | 0 | 0 | 85.10 / 159.82 / 175.67 | 19.16 | 615.96 | 330.53 | 363619.64 | too slow |
| 2 | 12.47 | 12.47 | 9.51 | 396 / 0 | 94 | 0 | 119.66 / 202.56 / 247.51 | 18.93 | 616.04 | 330.81 | 363618.01 | slight completed FPS gain, worse latency |
| 4 | 14.50 | 14.50 | 8.05 | 465 / 0 | 207 | 0 | 188.40 / 346.24 / 379.96 | 18.91 | 615.76 | 331.41 | 364004.32 | still far below target |
| 8 | 14.82 | 14.82 | 7.74 | 471 / 0 | 225 | 0 | 396.48 / 630.57 / 729.42 | 19.00 | 615.74 | 330.99 | 363922.74 | queueing dominates |

## Decision

```text
Full depth_u16 realtime target: fail
Best completed FPS observed: 14.82 FPS at inflight=8
Target: 45 FPS
Gap: about 3.0x
```

Multi-inflight hides some single-flight RTT, but the full-depth path saturates
around `14-15 completed FPS` and latency becomes unacceptable. Server compute is
still not the bottleneck: server total median stays around `19 ms` while RTT
grows from `85 ms` to almost `400 ms` as in-flight depth requests increase.

Next required path:

```text
Switch realtime remote FFS hot path to masked_uv_depth / ROI depth.
Keep full depth_u16 as semantic correctness and baseline mode.
```

