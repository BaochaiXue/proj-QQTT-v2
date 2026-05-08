# Demo v0.2 100-Kit Replay Profile

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
Input source: existing real IR triplets copied/cycled to 100 kits
Source case: data_collect/both_30_still_object_round8_20260428
Replay dir: result/demo_v0_2_data_ir_triplet_replay_100kits_848x480
Replay size: 100 triplet kits
Target send rate: 15 kit-FPS
```

No masks, SAM3.1, EdgeTAM, PCD, or rendering were used.

## Replay Preparation

The source case had 21 common three-camera IR frame ids. Demo v0.2 copied and
cycled those real IR triplets into a 100-kit replay directory.

```text
source_unique_frame_count = 21
replay_frame_count = 100
target_kit_fps = 15
```

## 100-Kit Results

| Inflight | Submitted | Completed | Kit FPS | Camera-depth FPS | E2E mean | E2E p90 | E2E p99 | E2E max | Server total mean | FFS stage mean | Request KB | Response KB | Timeouts |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 100 | 100 | 2.99 | 8.98 | 280.84 | 370.85 | 563.35 | 596.13 | 63.28 | 58.34 | 1907.15 | 1198.71 | 0 |
| 2 | 100 | 100 | 2.44 | 7.32 | 761.59 | 1000.99 | 1529.15 | 1718.58 | 63.13 | 58.22 | 1907.15 | 1198.71 | 0 |

All timing values are milliseconds except FPS and KB columns.

## Stage Timing Detail

Inflight 1:

```text
kit_e2e_ms: min=177.20 mean=280.84 p90=370.85 p99=563.35 max=596.13
server_total_ms: min=58.90 mean=63.28 p90=70.36 p99=75.59 max=82.19
server_decode_ms: min=0.87 mean=1.08 p90=1.44 p99=1.64 max=1.71
server_ffs_stage_ms: min=54.28 mean=58.34 p90=65.36 p99=70.91 max=77.52
server_encode_ms: min=3.39 mean=3.74 p90=4.20 p99=4.59 max=4.61
server_ffs_ms_per_camera: min=12.28 mean=13.15 p90=13.90 p99=17.09 max=22.58
server_align_ms_per_camera: min=4.31 mean=5.36 p90=7.62 p99=10.67 max=13.06
```

Inflight 2:

```text
kit_e2e_ms: min=396.16 mean=761.59 p90=1000.99 p99=1529.15 max=1718.58
server_total_ms: min=58.72 mean=63.13 p90=69.56 p99=74.68 max=78.10
server_decode_ms: min=0.86 mean=1.07 p90=1.41 p99=1.51 max=1.59
server_ffs_stage_ms: min=54.10 mean=58.22 p90=64.35 p99=69.97 max=73.02
server_encode_ms: min=3.55 mean=3.71 p90=4.01 p99=4.74 max=4.86
server_ffs_ms_per_camera: min=12.21 mean=13.02 p90=13.88 p99=15.35 max=16.40
server_align_ms_per_camera: min=4.39 mean=5.45 p90=7.96 p99=10.16 max=11.56
```

## Decision

Full-depth Demo v0.2 remains a semantic/capacity baseline, not a realtime
solution for the current transport path.

```text
Target: 15 kit-FPS / 45 camera-depth-FPS
Best 100-kit result: 2.99 kit-FPS / 8.98 camera-depth-FPS
```

The server-side staged pipeline is stable and fast:

```text
decode mean ~= 1 ms
FFS triplet stage mean ~= 58 ms
encode mean ~= 3.7 ms
server total mean ~= 63 ms
```

The limiting factor is still end-to-end full-depth payload transport/client
send-receive behavior. Raising in-flight from 1 to 2 increased latency and
reduced throughput in this 100-kit run.

