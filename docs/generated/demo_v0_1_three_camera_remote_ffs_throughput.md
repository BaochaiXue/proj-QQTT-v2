# Demo v0.1 Three-Camera Remote FFS Throughput

Date: 2026-05-07

## Purpose

Demo v0.1 isolates remote FFS transport and server throughput from SAM3.1,
EdgeTAM, masks, PCD filtering, and rendering.

The benchmark sends real IR from three local WSL-5090 RealSense cameras to the
Ubuntu-4090 FFS TensorRT server and receives full `depth_u16` replies.

## Evaluation

The testing idea is correct for the network/remote-depth question:

```text
WSL-5090:
  capture real IR left/right from cam0/cam1/cam2
  send 3 requests per group asynchronously
  receive depth replies asynchronously

Ubuntu-4090:
  run FFS TensorRT per request
  return full depth_u16
```

This does not need a mask. Masked returns are useful for final masked PCD
realtime optimization, but they hide the full-depth transport ceiling. Demo v0.1
therefore uses full `depth_u16/lz4`.

Important limitation:

```text
current server socket = REP
current server processing = one request at a time
current WSL client = async pressure via multiple independent REQ sockets
```

So Demo v0.1 measures current end-to-end throughput under asynchronous client
pressure. It is not yet a full ROUTER/DEALER asynchronous server design.

## Targets

```text
single camera: >=45 FPS
three cameras: >=15 FPS/camera
aggregate: >=45 camera-FPS
complete group target: >=15 complete 3-camera groups/s
```

## Required 4090 Server Mode

Ubuntu-4090 must run full depth:

```text
return_type = depth_u16
response_compression = lz4
```

If the server is still in `masked_uv_depth/lz4`, Demo v0.1 is invalid because
the server will force sparse returns and requires a mask.

### 4090 command

```bash
cd /home/xinjie/proj-QQTT-v2
export CUDA_VISIBLE_DEVICES=1

kill "$(cat /tmp/qqtt_ffs_strict_masked_uv_depth_lz4_7001.pid)" || true
kill "$(cat /tmp/qqtt_ffs_strict_depth_u16_lz4_7001.pid)" || true
sleep 2

nohup conda run --no-capture-output -n demo_2_max \
  python services/ffs_remote/ffs_depth_server.py \
  --bind tcp://0.0.0.0:7001 \
  --ffs-repo /home/xinjie/Fast-FoundationStereo \
  --ffs-trt-model-dir data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864 \
  --return depth_u16 \
  --compress lz4 \
  --warmup 20 \
  --debug \
  --strict-engine-contract \
  --required-model 20-30-48 \
  --required-valid-iters 4 \
  --required-height 480 \
  --required-width 864 \
  --required-builder-optimization-level 5 \
  --required-max-disp 192 \
  > /tmp/qqtt_ffs_strict_depth_u16_lz4_7001.log 2>&1 &

echo $! > /tmp/qqtt_ffs_strict_depth_u16_lz4_7001.pid
sleep 8
tail -n 120 /tmp/qqtt_ffs_strict_depth_u16_lz4_7001.log
ss -ltnp | grep 7001 || true
```

## WSL-5090 Command

This runs three local cameras at 15 groups/s, sending 3 full-depth requests per
group. `--inflight 6` is a first pass; test `3/6/9/12` if the server is stable.

```bash
cd /home/zhangxinjie/proj-QQTT-v2

conda run --no-capture-output -n demo_2_max \
  python services/ffs_remote/ffs_depth_client.py \
  --endpoint tcp://192.168.0.162:7001 \
  --three-camera-real-ir-depth-benchmark \
  --profile 848x480 \
  --fps 15 \
  --duration-s 30 \
  --timeout-ms 5000 \
  --compress lz4 \
  --return-type depth_u16 \
  --inflight 6 \
  --drop-stale-replies \
  --save-first-depth-preview \
  --debug
```

Optional explicit serial order:

```bash
--serials 239222300412 239222300781 239222303506
```

## Reported Metrics

The client summary line is:

```text
[ffs-remote-demo-v0.1-summary] ...
```

Key fields:

```text
camera_count
target_per_camera_fps
target_aggregate_camera_fps
complete_group_fps
aggregate_completed_fps
per_camera_completed_fps
rtt_ms_p50/p95
server_ffs_ms_p50
server_align_ms_p50
server_total_ms_p50
request_kb_mean
response_kb_mean
mbps_payload
depth_nonzero_count_mean
return_type
depth_shapes
```

Pass condition:

```text
complete_group_fps >= 15
aggregate_completed_fps >= 45
per_camera_completed_fps >= 15 for all cameras
return_type = depth_u16
depth_shapes includes 480x848
failed = 0 or near 0
```

## Decision Boundary

If Demo v0.1 fails below 45 aggregate camera-FPS while server
`server_total_ms_p50` remains low, the bottleneck is transport/protocol pressure,
not FFS compute. The next protocol step is ROUTER/DEALER or another explicit
async server design.
