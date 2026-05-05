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

```bash
conda run --no-capture-output -n demo_2_max \
  python services/ffs_remote/ffs_depth_server.py \
  --bind tcp://0.0.0.0:7001 \
  --ffs-repo ../Fast-FoundationStereo \
  --ffs-trt-model-dir data/experiments/ffs_trt_static_rounds_848x480_pad864_builderopt5_rtx5090_laptop_20260428/engines/model_20-30-48_iters_4_res_480x864 \
  --return depth_u16 \
  --warmup 20 \
  --debug
```

`--warmup` is lazy: the server runs warmup iterations with the first real
request's IR pair and calibration, because the server does not know the camera
intrinsics before the first request arrives.

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
