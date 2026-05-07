# Demo 2 remote FFS 4090 server / 5090 client validation

Run time: 2026-05-07 12:39 EDT

## Roles

WSL-5090 remains the RealSense / SAM3.1 / HF EdgeTAM / UI client.
Ubuntu-4090 is intended to be the FFS TensorRT depth service.

Current tested endpoint:

```text
tcp://100.66.203.123:7001
```

The old LAN endpoint is still not reachable from WSL:

```text
tcp://10.201.169.250:7001
```

## WSL-5090 sanity

Command:

```bash
conda run --no-capture-output -n demo_2_max python - <<'PY'
import torch, zmq
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("pyzmq", zmq.__version__)
PY
```

Result:

```text
torch 2.11.0+cu130 cuda 13.0
cuda available True
device NVIDIA GeForce RTX 5090 Laptop GPU
pyzmq 27.1.0
```

RealSense devices visible in WSL:

```text
devices 3
Intel RealSense D455 239222303506
Intel RealSense D455 239222300412
Intel RealSense D455 239222300781
```

## TCP reachability

Command:

```bash
python - <<'PY'
import socket
for host in ["100.66.203.123", "10.201.169.250"]:
    s = socket.socket()
    s.settimeout(3)
    try:
        s.connect((host, 7001))
        print("tcp_connect_ok", host, 7001)
    except Exception as e:
        print("tcp_connect_fail", host, 7001, type(e).__name__, e)
    finally:
        s.close()
PY
```

Result:

```text
tcp_connect_ok 100.66.203.123 7001
tcp_connect_fail 10.201.169.250 7001 TimeoutError timed out
```

## Echo benchmark: full 848x480 payload

Command:

```bash
conda run --no-capture-output -n demo_2_max \
  python services/ffs_remote/ffs_depth_client.py \
  --endpoint tcp://100.66.203.123:7001 \
  --echo-benchmark \
  --profile 848x480 \
  --fps 30 \
  --duration-s 20 \
  --timeout-ms 5000 \
  --debug \
  2>&1 | tee docs/generated/demo2_remote_ffs_echo_tailscale_current_20260507.txt
```

Summary:

```text
duration_s=20.05
sent=133
ok=133
failed=0
reply_fps=6.63
rtt_ms_p50=147.08
rtt_ms_p90=158.04
rtt_ms_p95=165.84
server_total_ms_p50=0.86
request_kb_mean=795.71
response_kb_mean=795.21
mbps_payload=86.43
```

Interpretation:

```text
TCP connectivity is working, but full 848x480 raw request/response throughput is
still far below the formal remote FFS realtime target. This path does not meet
the echo pass line of reply_fps ~= 30 and rtt_p50 < 15 ms.
```

## Echo benchmark: sparse masked_uv_depth probe

Command:

```bash
conda run --no-capture-output -n demo_2_max \
  python services/ffs_remote/ffs_depth_client.py \
  --endpoint tcp://100.66.203.123:7001 \
  --echo-benchmark \
  --profile 848x480 \
  --fps 30 \
  --duration-s 20 \
  --timeout-ms 5000 \
  --return-type masked_uv_depth \
  --mask-fraction 0.06 \
  --debug \
  2>&1 | tee docs/generated/demo2_remote_ffs_echo_tailscale_current_masked_uv_depth_006_20260507.txt
```

Result:

```text
status=error
FfsRemoteProtocolError: request expected 3 parts, got 4
duration_s=20.07
sent=184
ok=0
failed=184
```

Interpretation:

```text
The currently running Ubuntu-4090 echo server is an older echo-only protocol
that accepts only metadata + left IR + right IR. It does not accept the mask
payload needed by masked_uv_depth sparse echo / real sparse remote FFS.
```

## Decision

Current remote formal path status:

```text
FAIL / blocked.
```

Reasons:

```text
1. The only reachable endpoint is tcp://100.66.203.123:7001; the LAN endpoint is
   still unreachable from WSL.
2. Full 848x480 raw echo over the reachable endpoint is only 6.63 FPS with
   rtt_ms_p50 ~= 147 ms.
3. Sparse masked_uv_depth echo cannot be tested until the Ubuntu-4090 server is
   updated/restarted with the mask-aware protocol.
```

Do not run professor-facing `--depth-source ffs_remote` from this endpoint until:

```text
1. Ubuntu-4090 restarts the current repo's ffs_depth_server.py.
2. Echo sparse masked_uv_depth accepts a mask payload.
3. Full or sparse echo throughput is high enough for the target demo FPS.
```

Next Ubuntu-4090 server command to test after repo update:

```bash
cd /home/xinjie/proj-QQTT-v2
export CUDA_VISIBLE_DEVICES=0

conda run --no-capture-output -n demo_2_max \
  python services/ffs_remote/ffs_depth_server.py \
  --bind tcp://0.0.0.0:7001 \
  --ffs-repo /home/xinjie/Fast-FoundationStereo \
  --ffs-trt-model-dir data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864 \
  --return masked_uv_depth \
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

