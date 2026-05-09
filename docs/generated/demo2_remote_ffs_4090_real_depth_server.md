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

Targets for all follow-up remote FFS reports:

- single-camera realtime: 45 FPS
- three-camera realtime: 15 FPS per camera, aggregate 45 camera-FPS
- semantic requirement: real RealSense IR left/right input and real server depth output
- synthetic echo: protocol sanity only, never formal performance evidence
- full `depth_u16`: semantic correctness / baseline mode
- `masked_uv_depth`: realtime hot-path candidate if full-depth transport cannot reach target

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

`nvidia-smi` snapshot before the masked server switch:

```text
0, NVIDIA GeForce RTX 4090, 570.211.01, 6137 MiB / 24564 MiB, 72 %
1, NVIDIA GeForce RTX 4090, 570.211.01, 4354 MiB / 24564 MiB, 31 %
```

## FFS Contract

- Fast-FoundationStereo repo: `/home/xinjie/Fast-FoundationStereo`
- engine path: `data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864`
- return_type: currently `masked_uv_depth`; previous full-depth baseline used `depth_u16`
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

- server PID: 860860
- PID file: `/tmp/qqtt_ffs_strict_masked_uv_depth_lz4_7001.pid`
- log file: `/tmp/qqtt_ffs_strict_masked_uv_depth_lz4_7001.log`
- listen status: pass, `0.0.0.0:7001`
- current variant: `masked_uv_depth` with response `lz4`
- real request status: pass, server has processed real-depth requests
- current performance decision: full `depth_u16/lz4` semantic pass, full-depth realtime fail, masked realtime candidate active

Previous full-depth semantic baseline command:

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
- observed server total time: earlier about 18-24 ms, latest lz4/full-depth run about 23-28 ms
- bottleneck under investigation: network / serialization / compression / REQ-REP roundtrip, not 4090 TensorRT FFS
- current WSL conclusion for full-depth transport: `lz4/lz4` is best observed full-depth baseline
- best completed FPS for full `depth_u16/lz4`: 14.82
- full-depth realtime target status: not pass for single-camera 45 FPS or three-camera 15 FPS/camera
- next official realtime candidate: `masked_uv_depth/lz4` on `tcp://192.168.0.162:7001`

## Active masked_uv_depth Server

Switched on 2026-05-07 to the realtime hot-path candidate:

```text
pid: 860860
return_type: masked_uv_depth
response_compression: lz4
bind: tcp://0.0.0.0:7001
pid_file: /tmp/qqtt_ffs_strict_masked_uv_depth_lz4_7001.pid
log_file: /tmp/qqtt_ffs_strict_masked_uv_depth_lz4_7001.log
```

Startup log:

```text
[ffs-remote-server] {"bind": "tcp://0.0.0.0:7001", "compress": "lz4", "echo_only": false, "engine_contract": {"ffs_contract_builder_optimization_level": 5, "ffs_contract_capture_height": 480, "ffs_contract_capture_width": 848, "ffs_contract_engine_dir": "data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864", "ffs_contract_engine_height": 480, "ffs_contract_engine_width": 864, "ffs_contract_max_disp": 192, "ffs_contract_model": "20-30-48", "ffs_contract_padding_policy": "480x848->pad_to_480x864", "ffs_contract_strict": true, "ffs_contract_valid_iters": 4}, "ffs_repo": "/home/xinjie/Fast-FoundationStereo", "ffs_trt_model_dir": "data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864", "return_type": "masked_uv_depth", "warmup": 20}
```

Initial masked request log observed after the switch:

```text
[ffs-remote-server] lazy_warmup count=20 elapsed_ms=1354.42
[ffs-remote-server] frame_id=0 status=ok return=masked_uv_depth shape=(0, 4) ffs_ms=12.76 align_ms=10.88 total_ms=1378.98
[ffs-remote-server] frame_id=1 status=ok return=masked_uv_depth shape=(0, 4) ffs_ms=12.59 align_ms=4.12 total_ms=17.15
[ffs-remote-server] frame_id=2 status=ok return=masked_uv_depth shape=(0, 4) ffs_ms=13.42 align_ms=3.87 total_ms=17.70
[ffs-remote-server] frame_id=3 status=ok return=masked_uv_depth shape=(0, 4) ffs_ms=12.47 align_ms=4.08 total_ms=17.32
[ffs-remote-server] frame_id=4 status=ok return=masked_uv_depth shape=(0, 4) ffs_ms=12.46 align_ms=3.92 total_ms=16.74
```

The `shape=(0, 4)` response is a valid empty sparse payload, but it is not a
PCD success condition. WSL-5090 should confirm that object masks are non-empty
and are being attached to remote requests before interpreting masked PCD
throughput as valid.

Current prompt-corrected wait state:

- waiting for WSL-5090 to rerun with current-scene prompts:
  `object-prompt="stuff toy"` and `controller-prompt="rag"`
- current server return mode remains `masked_uv_depth/lz4`
- recent empty sparse responses, `shape=(0, 4)`, are attributed to an
  empty/missing WSL mask from stale first-frame SAM3.1 init prompts, not an FFS
  server failure
- server readiness for the corrected prompt run: pass, `0.0.0.0:7001`
  listening with strict 4090 TensorRT engine contract

No RealSense, `pyrealsense2`, EdgeTAM, SAM, or UI process was started on the
4090. The 4090 remains an FFS TensorRT server only.

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

Installed and restarted with explicit IPv4 bind for WSL-5090 network baseline:

```text
pid: 805787
command: iperf3 -s -B 0.0.0.0 -p 5201 -D --logfile /tmp/qqtt_iperf3_5201.log
log: /tmp/qqtt_iperf3_5201.log
listen: 0.0.0.0:5201
local nc check: pass, `nc -vz 127.0.0.1 5201`
note: `nc` is not a valid iperf3 handshake, so the final server was restarted after the nc probe with iperf3 daemon mode
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

## masked_uv_depth Restart Template

Use this if the current masked server must be restarted. It keeps the same
strict FFS TensorRT engine and returns sparse/ROI `masked_uv_depth`.

```bash
cd /home/xinjie/proj-QQTT-v2
export CUDA_VISIBLE_DEVICES=1

kill "$(cat /tmp/qqtt_ffs_strict_depth_u16_lz4_7001.pid 2>/dev/null)" 2>/dev/null || true
kill "$(cat /tmp/qqtt_ffs_strict_masked_uv_depth_lz4_7001.pid 2>/dev/null)" 2>/dev/null || true
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
    --return masked_uv_depth \
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
' > /tmp/qqtt_ffs_strict_masked_uv_depth_lz4_7001.log 2>&1 < /dev/null &
echo $! > /tmp/qqtt_ffs_strict_masked_uv_depth_lz4_7001.pid
sleep 8
tail -n 120 /tmp/qqtt_ffs_strict_masked_uv_depth_lz4_7001.log
ss -ltnp | grep 7001 || true
```

## Next WSL-5090 Steps

The 4090 server is now ready for WSL-5090 to run:

1. `masked_uv_depth/lz4` no-render inflight matrix against
   `tcp://192.168.0.162:7001`.
2. If single-camera no-render reaches 45 FPS, run the rendered single-camera
   masked test with the best inflight value.
3. If single-camera masked no-render passes, run the three-camera 15 FPS/camera
   proxy using real IR replay.
