# Demo v0.2 Async Remote FFS WSL-5090 Implementation

Date: 2026-05-07

Update: 2026-05-08

## Summary

Demo v0.2 is implemented as an independent full-depth capacity benchmark.

```text
No mask
No SAM3.1
No EdgeTAM
No PCD
No render
```

The WSL-5090 client can:

```text
record real three-camera IR triplets
replay recorded triplets to an async server
run live three-camera triplet requests
run single-camera replay/live requests
measure async inflight throughput and latency
report server stage timing for staged server runs
prepare a 100-kit replay dataset from existing real IR cases
run fixed-size 100-kit profiling with min/mean/max/p90/p99 summaries
```

## Files

```text
demo_v0_2/README.md
demo_v0_2/async_remote_ffs_triplet_client.py
services/ffs_remote/async_protocol_v02.py
services/ffs_remote/ffs_depth_async_server_v02.py
tests/test_demo_v02_async_protocol.py
```

## Protocol

```text
protocol = qqtt_demo_v0_2_async_remote_ffs
socket pattern = DEALER client -> ROUTER server
request = header + left/right IR for each camera
reply = header + depth_u16 for each camera
compression = lz4
return_type = depth_u16
```

## Targets

```text
single camera target: >=45 camera-depth-FPS
three camera target: >=15 kit-FPS
aggregate target: >=45 camera-depth-FPS
```

## Key Metrics

```text
completed_kit_fps
completed_camera_depth_fps
kit_e2e_ms_p50/p95
server_total_ms_p50/p95
server_decode_ms_p50/p95
server_ffs_stage_ms_p50/p95
server_encode_ms_p50/p95
server_router_queue_ms_p50
server_ffs_queue_ms_p50
server_encode_queue_ms_p50
server_ffs_ms_per_camera_p50/p95
server_align_ms_per_camera_p50/p95
request_kb_mean
response_kb_mean
inflight_mean/max
per_camera_completed_fps
latency/stage min/mean/p50/p90/p95/p99/max
```

## Validation

```text
python -m py_compile demo_v0_2/async_remote_ffs_triplet_client.py services/ffs_remote/async_protocol_v02.py services/ffs_remote/ffs_depth_async_server_v02.py
conda run --no-capture-output -n SAM21-max python -m unittest -v tests.test_demo_v02_async_protocol
conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py
```

Result:

```text
py_compile: pass
demo_2_max protocol tests: pass, 5 tests
SAM21-max protocol tests: pass, lz4 payload roundtrip skipped because lz4 is not installed there
check_all quick: pass, 132 tests
git diff --check: pass
```

## 2026-05-08 Staged Pipeline Addition

The v0.2 server now supports:

```text
--server-pipeline-mode fused-worker
--server-pipeline-mode staged
```

`staged` separates receive/decode, FFS, and reply encode/compression into
distinct queues. This directly tests whether throughput follows the slowest
stage while latency follows the sum of the overlapped stages plus queueing.

## 2026-05-08 100-Kit Replay Profiling Addition

The v0.2 client now supports:

```text
--mode prepare-replay-from-case
--source-case <case with ir_left/ir_right metadata>
--replay-frame-count 100
--max-submit-kits 100
```

This copies/cycles real IR triplets from `data/` or `data_collect/` into a
100-kit replay directory. The benchmark can then submit exactly 100 kits at
`--target-kit-fps 15`, simulating a 15 FPS three-camera IR video without live
camera attachment.
