# Demo v0.2 Async Remote FFS 4090 Server

Timestamp: 2026-05-08T14:57:13-04:00

## Role

- Machine: Native Ubuntu dual RTX 4090
- Role: server only
- Camera: none
- No `pyrealsense2`, SAM3.1, EdgeTAM, UI, mask, or PCD work on this host
- Existing Demo 2 server on `7001` left untouched

## Target

- Experiment: Demo v0.2 full async remote FFS throughput test
- Single-camera target: 45 camera-depth-FPS
- Three-camera target: 15 kit-FPS, aggregate 45 camera-depth-FPS
- Request input: real RealSense IR left/right from WSL-5090
- Server output: real `depth_u16`
- Socket pattern: ROUTER async server on port `7002`
- Compression: `lz4`

## Source

- Worktree: `/home/xinjie/proj-QQTT-v2-demo-v02`
- Source ref: `origin/main`
- Commit: `1e3e1f3 feat: implement asynchronous remote FFS client and server for demo v0.2 with associated documentation and tests`
- Original repo with generated reports: `/home/xinjie/proj-QQTT-v2`

The original repo worktree has local generated-doc changes, so Demo v0.2 was
run from a clean detached `origin/main` worktree. `git pull --ff-only` was not
applicable in the detached worktree, so after `git fetch origin` the worktree
was updated with `git checkout --detach origin/main`.

## Validation

Files verified in the Demo v0.2 worktree:

```text
services/ffs_remote/async_protocol_v02.py
services/ffs_remote/ffs_depth_async_server_v02.py
demo_v0_2/async_remote_ffs_triplet_client.py
demo_v0_2/README.md
tests/test_demo_v02_async_protocol.py
```

Commands:

```bash
python -m py_compile \
  demo_v0_2/async_remote_ffs_triplet_client.py \
  services/ffs_remote/async_protocol_v02.py \
  services/ffs_remote/ffs_depth_async_server_v02.py

conda run --no-capture-output -n demo_2_max \
  python -m unittest -v tests.test_demo_v02_async_protocol

git diff --check
```

Results:

```text
py_compile: pass
tests.test_demo_v02_async_protocol: pass, 5 tests
git diff --check in v0.2 worktree: pass
```

## Active Staged Server

- PID: `3883045`
- PID file: `/tmp/qqtt_demo_v02_async_ffs_server_7002.pid`
- Log: `/tmp/qqtt_demo_v02_async_ffs_server_7002.log`
- Bind: `tcp://0.0.0.0:7002`
- Listen status: pass
- CUDA_VISIBLE_DEVICES: `1`
- Pipeline mode: `staged`
- Decode workers: `1`
- GPU workers: `1`
- Encode workers: `1`
- Max queue: `32`
- Return: `depth_u16`
- Compression: `lz4`
- Fast-FoundationStereo repo: `/home/xinjie/Fast-FoundationStereo`
- Engine: `/home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864`

Startup command used:

```bash
cd /home/xinjie/proj-QQTT-v2-demo-v02
export CUDA_VISIBLE_DEVICES=1

setsid bash -lc '
  cd /home/xinjie/proj-QQTT-v2-demo-v02
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate demo_2_max
  export CUDA_VISIBLE_DEVICES=1
  exec python services/ffs_remote/ffs_depth_async_server_v02.py \
    --bind tcp://0.0.0.0:7002 \
    --ffs-repo /home/xinjie/Fast-FoundationStereo \
    --ffs-trt-model-dir /home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864 \
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
' > /tmp/qqtt_demo_v02_async_ffs_server_7002.log 2>&1 &
echo $! > /tmp/qqtt_demo_v02_async_ffs_server_7002.pid
```

Startup log:

```text
[demo-v0.2-async-server] {"bind": "tcp://0.0.0.0:7002", "compress": "lz4", "decode_workers": 1, "encode_workers": 1, "engine_contract": {"ffs_contract_builder_optimization_level": 5, "ffs_contract_capture_height": 480, "ffs_contract_capture_width": 848, "ffs_contract_engine_dir": "/home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864", "ffs_contract_engine_height": 480, "ffs_contract_engine_width": 864, "ffs_contract_max_disp": 192, "ffs_contract_model": "20-30-48", "ffs_contract_padding_policy": "480x848->pad_to_480x864", "ffs_contract_strict": true, "ffs_contract_valid_iters": 4}, "ffs_repo": "/home/xinjie/Fast-FoundationStereo", "ffs_trt_model_dir": "/home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864", "gpu_workers": 1, "max_queue": 32, "protocol": "qqtt_demo_v0_2_async_remote_ffs", "return_type": "depth_u16", "server_pipeline_mode": "staged"}
```

Port checks:

```text
7001: existing Demo 2 server still listening, PID 860860
7002: Demo v0.2 staged async server listening, PID 3883045
5201: iperf3 still listening, PID 805787
```

GPU snapshot:

```text
GPU 1 includes PID 3883045, python, about 1134 MiB
GPU 1 also includes existing 7001 server PID 860860
```

## Notes

- A first restart attempt used `pkill -f "ffs_depth_async_server_v02.py.*7002"`
  and matched the calling shell, so it was interrupted after stopping the old
  server. The final staged server was started without broad `pkill`.
- The current server log contains older malformed-request errors from probes
  against the previous non-staged server. The active staged startup line is the
  one shown above.
- Server lazy warmup is enabled with `--warmup 20`, so the first real request can
  have elevated latency and should be reported separately from steady state.

## Decision

- Demo v0.2 4090 staged async server status: ready
- WSL endpoint: `tcp://192.168.0.162:7002`
- Next WSL step: run real-IR triplet record, then triplet replay and single replay async matrices
