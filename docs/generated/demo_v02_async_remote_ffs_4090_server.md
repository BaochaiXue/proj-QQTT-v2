# Demo v0.2 Async Remote FFS 4090 Server

Date: 2026-05-07

Update: 2026-05-08

## Role

Ubuntu-4090 is server-only:

```text
No RealSense
No pyrealsense2 capture
No SAM3.1
No EdgeTAM
No UI
```

It runs a ROUTER socket on port `7002` and owns the FFS TensorRT runner.

Two server pipeline modes are available:

```text
fused-worker:
  decode/decompress -> FFS -> encode/compress in one worker

staged:
  ROUTER receive -> decode/decompress worker -> FFS worker -> encode/compress worker -> ROUTER send
```

## Command

```bash
cd /home/xinjie/proj-QQTT-v2
export CUDA_VISIBLE_DEVICES=1

nohup conda run --no-capture-output -n demo_2_max \
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
  --required-max-disp 192 \
  > /tmp/qqtt_demo_v02_async_ffs_server_7002.log 2>&1 &

echo $! > /tmp/qqtt_demo_v02_async_ffs_server_7002.pid
```

## Notes

Default `--gpu-workers 1` keeps one FFS TensorRT runner/context owned by one
thread. Higher worker counts create additional runner instances and should be
treated as a separate ablation.

Use the staged mode to test the throughput model:

```text
kit throughput ~= 1000 / max(decode_ms, ffs_stage_ms, encode_ms)
kit latency ~= decode_ms + queueing + ffs_stage_ms + queueing + encode_ms + network
```

The reply header includes `server_stage_ms`, and the WSL client summary prints
stage p50/p95 fields.
