# Demo v0.2: Full Async Remote FFS Throughput

Demo v0.2 is a capacity benchmark for remote FFS only.

It does not run:

```text
SAM3.1
EdgeTAM
masks
PCD filtering
Open3D rendering
```

It measures:

```text
WSL-5090 real RealSense IR
-> async DEALER client
-> Ubuntu-4090 ROUTER async FFS server
-> depth_u16 replies
```

The server has two pipeline modes:

```text
fused-worker:
  decode/decompress request -> FFS -> encode/compress reply inside one worker

staged:
  ROUTER receive -> decode/decompress worker -> FFS worker -> encode/compress worker -> ROUTER send
```

Use `staged` to test whether throughput follows `1000 / max(stage_ms)`
while end-to-end latency follows the sum of the overlapped stages.

## Targets

```text
single camera: >=45 camera-depth-FPS
three cameras: >=15 kit-FPS
aggregate: >=45 camera-depth-FPS
```

Latency may rise when `--max-inflight` increases. This benchmark explicitly
tests whether throughput can be bought with latency.

## 4090 Server

Run on Ubuntu-4090. Keep Demo 2 port `7001` untouched; v0.2 uses `7002`.

```bash
cd /home/xinjie/proj-QQTT-v2
export CUDA_VISIBLE_DEVICES=1

conda run --no-capture-output -n demo_2_max \
  python services/ffs_remote/ffs_depth_async_server_v02.py \
  --bind tcp://0.0.0.0:7002 \
  --ffs-repo /home/xinjie/Fast-FoundationStereo \
  --ffs-trt-model-dir data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864 \
  --return depth_u16 \
  --compress lz4 \
  --server-pipeline-mode staged \
  --decode-workers 1 \
  --gpu-workers 1 \
  --encode-workers 1 \
  --max-queue 32 \
  --warmup 20 \
  --debug \
  --strict-engine-contract \
  --required-model 20-30-48 \
  --required-valid-iters 4 \
  --required-height 480 \
  --required-width 864 \
  --required-builder-optimization-level 5 \
  --required-max-disp 192
```

## WSL Record

Run on WSL-5090 with three RealSense cameras.

```bash
cd /home/zhangxinjie/proj-QQTT-v2

conda run --no-capture-output -n demo_2_max \
  python demo_v0_2/async_remote_ffs_triplet_client.py \
  --mode triplet-live \
  --serials 239222300412,239222300781,239222303506 \
  --profile 848x480 \
  --camera-fps 30 \
  --record-dir result/demo_v0_2_real_ir_triplet_record_848x480 \
  --record-duration-s 30 \
  --no-send \
  --debug
```

## WSL Prepare 100-Kit Replay From Existing Data

If the cameras are not attached, build a 100-kit replay set from an existing
case that already contains real three-camera IR left/right frames:

```bash
cd /home/zhangxinjie/proj-QQTT-v2

conda run --no-capture-output -n demo_2_max \
  python demo_v0_2/async_remote_ffs_triplet_client.py \
  --mode prepare-replay-from-case \
  --source-case data_collect/both_30_still_object_round8_20260428 \
  --replay-dir result/demo_v0_2_data_ir_triplet_replay_100kits_848x480 \
  --replay-frame-count 100 \
  --target-kit-fps 15 \
  --debug
```

This copies/cycles real IR triplets into:

```text
result/demo_v0_2_data_ir_triplet_replay_100kits_848x480/
  metadata.json
  cam0/left/000000.png
  cam0/right/000000.png
  ...
```

It still uses real IR frames; it does not synthesize image contents.

## WSL Replay Matrix

```bash
cd /home/zhangxinjie/proj-QQTT-v2

for N in 1 2 3 6 9 12 16; do
  conda run --no-capture-output -n demo_2_max \
    python demo_v0_2/async_remote_ffs_triplet_client.py \
    --mode triplet-replay \
    --replay-dir result/demo_v0_2_data_ir_triplet_replay_100kits_848x480 \
    --endpoint tcp://192.168.0.162:7002 \
    --target-kit-fps 15 \
    --compression lz4 \
    --return-type depth_u16 \
    --max-inflight "$N" \
    --max-submit-kits 100 \
    --drop-stale-replies \
    --duration-s 90 \
    --save-first-depth-preview \
    --debug
done
```

## WSL Live Triplet

Only run after replay passes.

```bash
BEST_N=6

conda run --no-capture-output -n demo_2_max \
  python demo_v0_2/async_remote_ffs_triplet_client.py \
  --mode triplet-live \
  --serials 239222300412,239222300781,239222303506 \
  --profile 848x480 \
  --camera-fps 30 \
  --endpoint tcp://192.168.0.162:7002 \
  --target-kit-fps 15 \
  --compression lz4 \
  --return-type depth_u16 \
  --max-inflight "$BEST_N" \
  --drop-stale-replies \
  --duration-s 60 \
  --save-first-depth-preview \
  --debug
```

## Summary Line

The WSL client prints:

```text
[demo-v0.2-summary] ...
```

Important fields for the staged pipeline:

```text
server_decode_ms_p50/p95
server_ffs_stage_ms_p50/p95
server_encode_ms_p50/p95
server_router_queue_ms_p50
server_ffs_queue_ms_p50
server_encode_queue_ms_p50
kit_e2e_ms_p50/p95
completed_kit_fps
completed_camera_depth_fps
```

For fixed-size 100-kit profiling, use:

```text
--max-submit-kits 100
--target-kit-fps 15
```

The summary includes min/mean/p50/p90/p95/p99/max for:

```text
kit_e2e_ms
server_total_ms
server_decode_ms
server_ffs_stage_ms
server_encode_ms
server_ffs_ms_per_camera
server_align_ms_per_camera
```

Pass:

```text
completed_kit_fps >= 15
completed_camera_depth_fps >= 45
failed = 0
```
