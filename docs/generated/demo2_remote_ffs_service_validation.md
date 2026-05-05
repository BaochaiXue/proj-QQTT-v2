# Demo 2 Remote FFS Service Validation

Date: 2026-05-05

## What Changed

Demo 2 now has a remote FFS depth source:

```text
--depth-source ffs_remote
```

The local machine keeps:

```text
RealSense capture
SAM3.1 first-frame init
HF EdgeTAM streaming masks
masked PCD / render / UI
```

The remote GPU machine runs:

```text
services/ffs_remote/ffs_depth_server.py
```

This is service offload. It does not expose the remote GPU as a local CUDA
device.

Both client and server environments need `pyzmq`.

Important engine note: the DORM-4090 server should use an FFS TensorRT engine
built or at least validated on the 4090 machine. Do not assume a serialized
TensorRT engine produced on the local RTX 5090 Laptop will run on the RTX 4090;
without explicit TensorRT hardware compatibility settings, engines are not
generally portable across GPU architectures.

## Protocol

Transport is ZeroMQ multipart REQ/REP:

```text
request part 0: JSON metadata
request part 1: IR left uint8 bytes
request part 2: IR right uint8 bytes

response part 0: JSON metadata
response part 1: color-aligned depth bytes
```

The first version supports `--ffs-remote-max-inflight 1` only. The PCD worker
requests depth for the current mask packet's `seq`; if the request times out or
returns a mismatched `frame_id`, that frame is skipped rather than combining old
depth with new masks.

## Server Command

Expected 4090-local engine target:

```text
model: 20-30-48
valid_iters: 4
input: 848x480 padded to 864x480
builderOptimizationLevel: 5
```

```bash
conda run --no-capture-output -n demo_2_max \
  python services/ffs_remote/ffs_depth_server.py \
  --bind tcp://0.0.0.0:7001 \
  --ffs-repo ../Fast-FoundationStereo \
  --ffs-trt-model-dir data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864 \
  --return depth_u16 \
  --warmup 20 \
  --debug
```

`--warmup` is lazy: the server runs warmup iterations with the first real
request's IR pair and calibration, because the server does not know the camera
intrinsics before the first request arrives.

## Echo-Only Network Check

Remote server:

```bash
conda run --no-capture-output -n demo_2_max \
  python services/ffs_remote/ffs_depth_server.py \
  --bind tcp://0.0.0.0:7001 \
  --echo-only \
  --debug
```

Local client:

```bash
conda run --no-capture-output -n demo_2_max \
  python services/ffs_remote/ffs_depth_client.py \
  --endpoint tcp://<remote_tailscale_ip>:7001 \
  --echo-benchmark \
  --profile 848x480 \
  --fps 30 \
  --duration-s 20 \
  --debug
```

Passing echo-only means request/reply works, RTT is stable enough for a live
depth path, and the payload size matches the planned `848x480` IR-pair plus
`depth_u16` reply path.

## Client Command

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2/realtime_masked_edgetam_pcd.py \
  --serial 239222300412 \
  --profile 848x480 \
  --fps 60 \
  --depth-source ffs_remote \
  --ffs-remote-endpoint tcp://<remote_tailscale_ip>:7001 \
  --ffs-remote-max-inflight 1 \
  --ffs-remote-return depth_u16 \
  --init-mode sam31-first-frame \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --pcd-mode masked \
  --render-mode none \
  --compile-mode vision-reduce-overhead \
  --dtype bfloat16 \
  --debug \
  --profile-cuda-events \
  --duration-s 60
```

## Expected Readout

Success means local EdgeTAM timing moves back toward the EdgeTAM-only baseline
while depth timing moves into the remote/network fields:

```text
cuda_event_model_ms: should be closer to local EdgeTAM-only timing
remote_rtt_ms: network + server round trip
remote_server_total_ms: remote FFS + align + server overhead
ffs_ms / ffs_align_ms: server-reported FFS and align timings
```

If `remote_rtt_ms` is high or unstable, check the Tailscale path first. A direct
connection is expected for useful realtime behavior; DERP relay is likely only
usable for quality preview.

## Validation

Implemented deterministic coverage:

```text
protocol request/response roundtrip
client uint16 depth decode
Demo 2 CLI exposes ffs_remote args
Demo 2 ffs_remote validation skips local FFS engine checks
```

Local environment note:

```text
demo_2_max: installed pyzmq==27.1.0 on 2026-05-05
```

Echo-only smoke:

```text
server: services/ffs_remote/ffs_depth_server.py --bind tcp://127.0.0.1:7011 --echo-only --debug
client: FfsRemoteDepthClient(endpoint="tcp://127.0.0.1:7011", timeout_ms=1000)
result: frame_id=7, depth_shape=(2, 3), depth_sum=0.0, rtt_ms=9.41
```

Echo benchmark CLI smoke:

```text
server: services/ffs_remote/ffs_depth_server.py --bind tcp://127.0.0.1:7012 --echo-only --debug
client: services/ffs_remote/ffs_depth_client.py --endpoint tcp://127.0.0.1:7012 --echo-benchmark --profile 848x480 --fps 30 --duration-s 2 --debug
result: sent=60, ok=60, failed=0, reply_fps=30.00, rtt_ms_p50=2.67, rtt_ms_p95=3.51, payload_mbps=390.89
```

Pending two-machine result table:

| mode | seg FPS | pcd FPS | render FPS | EdgeTAM cuda event ms | remote RTT ms | remote server total ms | timeout/drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| local FFS quality mode | TBD | TBD | TBD | TBD | n/a | n/a | TBD |
| remote FFS no-render | TBD | TBD | n/a | TBD | TBD | TBD | TBD |
| remote FFS pointcloud render | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
