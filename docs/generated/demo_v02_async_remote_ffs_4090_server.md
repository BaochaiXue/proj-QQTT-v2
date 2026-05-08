# Demo v0.2 Async Remote FFS 4090 Server

Date: 2026-05-07

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
  --required-max-disp 192 \
  > /tmp/qqtt_demo_v02_async_ffs_server_7002.log 2>&1 &

echo $! > /tmp/qqtt_demo_v02_async_ffs_server_7002.pid
```

## Notes

Default `--gpu-workers 1` keeps one FFS TensorRT runner/context owned by one
thread. Higher worker counts create additional runner instances and should be
treated as a separate ablation.
