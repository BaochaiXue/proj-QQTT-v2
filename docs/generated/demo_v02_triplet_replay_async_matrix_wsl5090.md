# Demo v0.2 Triplet Replay Async Matrix

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
Input source: existing real IR triplet replay, no synthetic IR
Source case: data_collect/both_30_still_object_round8_20260428
Replay dir: result/demo_v0_2_data_ir_triplet_replay_848x480_still_object_round8
Replay frames: 21 common three-camera IR frame ids
```

No masks, SAM3.1, EdgeTAM, PCD, or rendering were used.

## Results

| Inflight | Kit FPS | Camera-depth FPS | E2E p50 ms | Server total p50 ms | Decode p50 ms | FFS stage p50 ms | Encode p50 ms | Request KB | Response KB | Timeouts |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.10 | 9.30 | 257.39 | 62.82 | 0.97 | 58.04 | 3.61 | 1907.15 | 1198.66 | 0 |
| 2 | 3.14 | 9.43 | 598.68 | 62.62 | 0.98 | 57.77 | 3.68 | 1907.14 | 1198.72 | 0 |
| 3 | 2.69 | 8.07 | 1107.75 | 62.77 | 0.98 | 58.03 | 3.62 | 1907.16 | 1198.70 | 0 |
| 6 | 2.26 | 6.79 | 2523.19 | 63.32 | 0.96 | 58.48 | 3.63 | 1907.14 | 1198.58 | 0 |
| 9 | 2.30 | 6.90 | 3962.11 | 62.16 | 1.05 | 57.09 | 3.67 | 1907.16 | 1198.60 | 0 |
| 12 | 2.57 | 7.71 | 4394.76 | 62.28 | 1.01 | 57.15 | 3.66 | 1907.14 | 1198.72 | 0 |
| 16 | 2.48 | 7.44 | 6038.11 | 62.03 | 0.96 | 57.09 | 3.68 | 1907.15 | 1198.66 | 3 |
| 24 | 1.92 | 5.76 | 11454.59 | 62.31 | 1.01 | 57.15 | 3.65 | 1907.15 | 1198.60 | 14 |
| 32 | 2.24 | 6.72 | 8985.23 | 61.99 | 0.97 | 57.10 | 3.64 | 1907.15 | 1198.70 | 22 |

## Decision

Triplet full-depth replay does not meet the Demo v0.2 target:

```text
Target: >=15 kit-FPS / >=45 camera-depth-FPS
Best observed: 3.14 kit-FPS / 9.43 camera-depth-FPS
```

The 4090 server-side staged pipeline is not the bottleneck in this run:

```text
decode p50 ~= 1 ms
FFS stage p50 ~= 57-58 ms per triplet
encode p50 ~= 3.6 ms
server total p50 ~= 62-63 ms
```

Increasing in-flight requests did not buy throughput. It mostly increased
end-to-end latency and eventually caused timeouts. The limiting path is the
large full-depth payload transport/client-side send-receive path, not FFS
TensorRT compute.

