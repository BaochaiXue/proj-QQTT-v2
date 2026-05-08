# Demo v0.2 Single-Camera Replay Async Matrix

Date: 2026-05-08

## Setup

```text
Client: WSL Ubuntu RTX 5090 Laptop
Server: Ubuntu RTX 4090
Endpoint: tcp://192.168.0.162:7002
Server mode: staged
Server workers: decode=1, gpu=1, encode=1
Return: full depth_u16
Compression: lz4
Input source: cam0 from real IR replay
Source replay dir: result/demo_v0_2_data_ir_triplet_replay_848x480_still_object_round8
```

No masks, SAM3.1, EdgeTAM, PCD, or rendering were used.

## Results

| Inflight | Camera-depth FPS | E2E p50 ms | Server total p50 ms | Decode p50 ms | FFS stage p50 ms | Encode p50 ms | Request KB | Response KB | Timeouts |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.65 | 92.70 | 20.19 | 0.53 | 18.20 | 1.22 | 640.81 | 304.08 | 0 |
| 2 | 7.86 | 221.88 | 20.26 | 0.39 | 18.36 | 1.21 | 640.82 | 304.06 | 0 |
| 4 | 8.79 | 422.50 | 20.10 | 0.45 | 18.14 | 1.20 | 640.81 | 304.06 | 0 |
| 8 | 8.23 | 916.35 | 20.14 | 0.40 | 18.21 | 1.21 | 640.81 | 304.08 | 0 |
| 16 | 6.71 | 2442.89 | 20.24 | 0.40 | 18.28 | 1.20 | 640.82 | 304.06 | 0 |
| 32 | 7.46 | 3929.44 | 20.08 | 0.40 | 18.14 | 1.20 | 640.82 | 304.07 | 0 |

## Decision

Single-camera full-depth replay does not meet the Demo v0.2 target:

```text
Target: >=45 camera-depth-FPS
Best observed: 8.79 camera-depth-FPS
```

The server-side FFS path is fast enough for the single-camera compute target:

```text
server total p50 ~= 20 ms
FFS stage p50 ~= 18 ms
```

The full-depth transport path is the bottleneck. Raising in-flight depth did
not raise throughput and increased latency from roughly 93 ms to multiple
seconds.

