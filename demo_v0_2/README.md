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
  --gpu-workers 1 \
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

## WSL Replay Matrix

```bash
cd /home/zhangxinjie/proj-QQTT-v2

for N in 1 2 3 6 9 12 16; do
  conda run --no-capture-output -n demo_2_max \
    python demo_v0_2/async_remote_ffs_triplet_client.py \
    --mode triplet-replay \
    --replay-dir result/demo_v0_2_real_ir_triplet_record_848x480 \
    --endpoint tcp://192.168.0.162:7002 \
    --target-kit-fps 15 \
    --compression lz4 \
    --return-type depth_u16 \
    --max-inflight "$N" \
    --drop-stale-replies \
    --duration-s 60 \
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

Pass:

```text
completed_kit_fps >= 15
completed_camera_depth_fps >= 45
failed = 0
```
