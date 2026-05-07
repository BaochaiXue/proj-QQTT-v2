# Demo 2 Remote FFS 4090 Real Depth Server

Date: 2026-05-07

## Scope

This report records the Ubuntu RTX 4090 side of the Demo 2 remote FFS handoff.
Synthetic echo is treated only as TCP/protocol sanity. The active service is a
real FFS TensorRT depth server that expects real RealSense IR left/right frames
from the WSL-5090 client.

This 4090 host has no camera role in the experiment. It does not open
RealSense, does not run EdgeTAM/SAM/UI, and does not run the real-IR client
benchmark.

## Server

- machine: Native Ubuntu RTX 4090
- role: FFS TensorRT server only, no camera
- GPU: RTX 4090
- driver: 570.211.01
- CUDA_VISIBLE_DEVICES: 1
- visible PyTorch device: 0, NVIDIA GeForce RTX 4090, capability (8, 9)
- torch: 2.11.0+cu128
- torch CUDA: 12.8
- TensorRT: 10.16.1.11
- pyzmq: 27.1.0
- lz4: 4.4.5
- zstandard: 0.25.0
- bind: tcp://0.0.0.0:7001
- current preferred endpoint for real payload matrix: tcp://192.168.0.162:7001
- secondary endpoint for comparison: tcp://128.59.19.35:7001
- ufw: inactive

`nvidia-smi` snapshot after server startup:

```text
0, NVIDIA GeForce RTX 4090, 570.211.01, 6243 MiB / 24564 MiB, 60 %
1, NVIDIA GeForce RTX 4090, 570.211.01, 6033 MiB / 24564 MiB, 40 %
```

## FFS Contract

- Fast-FoundationStereo repo: `/home/xinjie/Fast-FoundationStereo`
- engine path: `data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864`
- return_type: `depth_u16`
- response_compression: `lz4`
- strict contract: pass
- required model: `20-30-48`
- required valid_iters: 4
- required capture size: 480x848
- required engine size: 480x864
- required builderOptimizationLevel: 5
- required max_disp: 192
- warmup: 20 lazy iterations on first real request

## Active Process

- server PID: 550803
- PID file: `/tmp/qqtt_ffs_strict_depth_u16_lz4_7001.pid`
- log file: `/tmp/qqtt_ffs_strict_depth_u16_lz4_7001.log`
- listen status: pass, `0.0.0.0:7001`
- current variant: response `lz4`
- real request status: pass, server has processed real-depth requests

Startup command:

```bash
cd /home/xinjie/proj-QQTT-v2
export CUDA_VISIBLE_DEVICES=1
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate demo_2_max
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
  --required-max-disp 192
```

Startup log tail:

```text
compression imports ok
[ffs-remote-server] {"bind": "tcp://0.0.0.0:7001", "compress": "lz4", "echo_only": false, "engine_contract": {"ffs_contract_builder_optimization_level": 5, "ffs_contract_capture_height": 480, "ffs_contract_capture_width": 848, "ffs_contract_engine_dir": "data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864", "ffs_contract_engine_height": 480, "ffs_contract_engine_width": 864, "ffs_contract_max_disp": 192, "ffs_contract_model": "20-30-48", "ffs_contract_padding_policy": "480x848->pad_to_480x864", "ffs_contract_strict": true, "ffs_contract_valid_iters": 4}, "ffs_repo": "/home/xinjie/Fast-FoundationStereo", "ffs_trt_model_dir": "data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864", "return_type": "depth_u16", "warmup": 20}
```

Live request log tail observed after startup:

```text
[ffs-remote-server] frame_id=247 status=ok return=depth_u16 shape=(480, 848) ffs_ms=14.09 align_ms=4.07 total_ms=19.32
[ffs-remote-server] frame_id=248 status=ok return=depth_u16 shape=(480, 848) ffs_ms=14.06 align_ms=5.23 total_ms=20.50
[ffs-remote-server] frame_id=249 status=ok return=depth_u16 shape=(480, 848) ffs_ms=13.33 align_ms=8.21 total_ms=24.32
[ffs-remote-server] frame_id=250 status=ok return=depth_u16 shape=(480, 848) ffs_ms=14.66 align_ms=7.35 total_ms=23.67
[ffs-remote-server] frame_id=251 status=ok return=depth_u16 shape=(480, 848) ffs_ms=13.14 align_ms=4.17 total_ms=18.74
[ffs-remote-server] frame_id=252 status=ok return=depth_u16 shape=(480, 848) ffs_ms=14.09 align_ms=4.71 total_ms=20.28
[ffs-remote-server] frame_id=253 status=ok return=depth_u16 shape=(480, 848) ffs_ms=13.30 align_ms=4.12 total_ms=19.01
[ffs-remote-server] frame_id=254 status=ok return=depth_u16 shape=(480, 848) ffs_ms=13.76 align_ms=5.73 total_ms=21.56
```

Current interpretation:

- semantic path: pass, real client requests reached the server and returned real `depth_u16`
- return shape: `(480, 848)`
- observed FFS time: mostly about 13-15 ms
- observed align time: mostly about 4-8 ms
- observed server total time: mostly about 18-24 ms
- bottleneck under investigation: network / serialization / compression / REQ-REP roundtrip, not 4090 TensorRT FFS

## Validation Commands

```bash
python -m py_compile services/ffs_remote/ffs_depth_client.py services/ffs_remote/ffs_depth_server.py
python -m unittest -v \
  tests.test_realtime_single_camera_pointcloud_smoke.RealtimeSingleCameraPointCloudSmokeTest.test_ffs_remote_client_cli_help_and_echo_benchmark_summary \
  tests.test_realtime_single_camera_pointcloud_smoke.RealtimeSingleCameraPointCloudSmokeTest.test_ffs_remote_client_real_ir_benchmark_summary_and_artifacts
python scripts/harness/check_all.py
```

Results:

- py_compile: pass
- focused remote client tests: pass
- quick deterministic harness: pass, 132 tests
- `git diff --check`: pass

Not run on 4090:

- no `pyrealsense2` import or capture
- no RealSense IR stream startup
- no EdgeTAM/SAM/UI process
- no synthetic echo result used as formal handoff evidence

## iperf3 Server

Installed and started for WSL-5090 network baseline:

```text
pid: 750173
command: iperf3 -s -p 5201
log: /tmp/qqtt_iperf3_5201.log
listen: *:5201
```

WSL-5090 can run:

```bash
iperf3 -c 192.168.0.162 -p 5201 -t 20
iperf3 -c 192.168.0.162 -p 5201 -t 20 -R
iperf3 -c 192.168.0.162 -p 5201 -t 20 -P 4

iperf3 -c 128.59.19.35 -p 5201 -t 20
iperf3 -c 128.59.19.35 -p 5201 -t 20 -R
iperf3 -c 128.59.19.35 -p 5201 -t 20 -P 4
```

## Compression Variants

Do not run these until the WSL-5090 client is ready for the matching matrix
row. Each variant keeps the same strict FFS engine contract and changes only the
server response compression.

Shared command template:

```bash
cd /home/xinjie/proj-QQTT-v2
export CUDA_VISIBLE_DEVICES=1

kill "$(cat /tmp/qqtt_ffs_strict_depth_u16_lz4_7001.pid 2>/dev/null)" 2>/dev/null || true
kill "$(cat /tmp/qqtt_ffs_strict_depth_u16_png_7001.pid 2>/dev/null)" 2>/dev/null || true
kill "$(cat /tmp/qqtt_ffs_strict_depth_u16_none_7001.pid 2>/dev/null)" 2>/dev/null || true
sleep 2

setsid bash -lc '
  cd /home/xinjie/proj-QQTT-v2
  export CUDA_VISIBLE_DEVICES=1
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate demo_2_max
  exec python services/ffs_remote/ffs_depth_server.py \
    --bind tcp://0.0.0.0:7001 \
    --ffs-repo /home/xinjie/Fast-FoundationStereo \
    --ffs-trt-model-dir data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864 \
    --return depth_u16 \
    --compress COMPRESSION \
    --warmup 20 \
    --debug \
    --strict-engine-contract \
    --required-model 20-30-48 \
    --required-valid-iters 4 \
    --required-height 480 \
    --required-width 864 \
    --required-builder-optimization-level 5 \
    --required-max-disp 192
' > LOG_PATH 2>&1 < /dev/null &
echo $! > PID_PATH
sleep 8
tail -n 120 LOG_PATH
ss -ltnp | grep 7001 || true
```

Variant substitutions:

| Variant | `COMPRESSION` | `LOG_PATH` | `PID_PATH` |
| --- | --- | --- | --- |
| lz4 response | `lz4` | `/tmp/qqtt_ffs_strict_depth_u16_lz4_7001.log` | `/tmp/qqtt_ffs_strict_depth_u16_lz4_7001.pid` |
| png response | `png` | `/tmp/qqtt_ffs_strict_depth_u16_png_7001.log` | `/tmp/qqtt_ffs_strict_depth_u16_png_7001.pid` |
| none response | `none` | `/tmp/qqtt_ffs_strict_depth_u16_none_7001.log` | `/tmp/qqtt_ffs_strict_depth_u16_none_7001.pid` |

## Next Client Command

Run this from WSL-5090. This is the first formal handoff benchmark because it
uses real RealSense IR and requests real depth from the 4090 server:

```bash
conda run --no-capture-output -n demo_2_max \
  python services/ffs_remote/ffs_depth_client.py \
  --endpoint tcp://192.168.0.162:7001 \
  --real-ir-depth-benchmark \
  --serial 239222300412 \
  --profile 848x480 \
  --fps 30 \
  --duration-s 20 \
  --timeout-ms 5000 \
  --compress lz4 \
  --return-type depth_u16 \
  --save-first-depth-preview \
  --debug
```

Expected server-side request logs will include `frame_id`, `status=ok`,
`return=depth_u16`, `ffs_ms`, `align_ms`, and `total_ms`.
