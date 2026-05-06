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

## LAN Endpoint Attempt

Date: 2026-05-05

Target endpoint:

```text
tcp://10.201.169.250:7001
```

WSL-5090 TCP sanity check:

```text
tcp_connect_failed TimeoutError('timed out')
```

WSL-5090 ping:

```text
3 packets transmitted, 0 received, 100% packet loss
```

Windows host `Test-NetConnection`:

```text
ComputerName     : 10.201.169.250
RemoteAddress    : 10.201.169.250
TcpTestSucceeded : False
PingSucceeded    : False
InterfaceAlias   : Wi-Fi
SourceAddress    : 192.168.0.166
```

Interpretation:

```text
The WSL host and Windows host cannot reach the UBUNTU-4090 LAN IP. The cross-machine
echo benchmark was not run because the TCP endpoint is unreachable from LOCAL-5090.
Next steps are to verify UBUNTU-4090 is on a reachable LAN/VPN path, allow tcp/7001,
or enable Tailscale and use the Tailscale IP.
```

UBUNTU-4090 side confirmation:

```text
FFS echo server is still listening on 0.0.0.0:7001, PID 33024.
UBUNTU-4090 LAN IP is 10.201.169.250/24.
LOCAL-5090 Windows/WSL source address is 192.168.0.166.
These hosts are not on a mutually reachable LAN path, so tcp://10.201.169.250:7001
fails before ZeroMQ or Demo 2 code is involved.
tailscale command is not installed on UBUNTU-4090, and tailscaled is inactive.
UFW is active; tcp/7001 has not yet been confirmed open because the pkexec
authorization for ufw allow 7001/tcp did not complete.
```

Recommended next connection path:

```text
1. Put both machines on the same reachable LAN, or
2. Install/start Tailscale on UBUNTU-4090 and use the 4090 Tailscale IPv4 endpoint.
```

Expected next WSL-5090 command after Tailscale is available:

```bash
conda run --no-capture-output -n demo_2_max \
  python services/ffs_remote/ffs_depth_client.py \
  --endpoint tcp://<4090_TAILSCALE_IP>:7001 \
  --echo-benchmark \
  --profile 848x480 \
  --fps 30 \
  --duration-s 20 \
  --debug
```

## Userspace Tailscale Endpoint Attempt

Date: 2026-05-05

UBUNTU-4090 reported userspace Tailscale status:

```text
4090 Tailscale IP: 100.66.203.123
TCP serve configured:
  tcp://100.66.203.123:7001 -> tcp://127.0.0.1:7001
```

WSL-5090 echo benchmark command:

```bash
conda run --no-capture-output -n demo_2_max \
  python services/ffs_remote/ffs_depth_client.py \
  --endpoint tcp://100.66.203.123:7001 \
  --echo-benchmark \
  --profile 848x480 \
  --fps 30 \
  --duration-s 20 \
  --debug
```

Result:

```text
sent=248
ok=0
failed=248
error=Again: Resource temporarily unavailable
```

WSL-5090 TCP sanity check:

```text
tcp_connect_failed TimeoutError('timed out')
ping: 3 transmitted, 0 received, 100% packet loss
ip route get 100.66.203.123:
  100.66.203.123 via 192.168.0.1 dev eth1 src 192.168.0.166
```

Windows host `Test-NetConnection`:

```text
ComputerName     : 100.66.203.123
RemoteAddress    : 100.66.203.123
TcpTestSucceeded : False
PingSucceeded    : False
InterfaceAlias   : Wi-Fi
SourceAddress    : 192.168.0.166
```

Interpretation:

```text
The 4090 userspace Tailscale endpoint exists, but LOCAL-5090 is not currently
routing 100.66.203.123 through Tailscale. WSL routes the 100.x address through
the normal LAN gateway, and Windows does not have a tailscale command available.
The echo benchmark failed before ZeroMQ received any reply. LOCAL-5090 also needs
to join the tailnet, or UBUNTU-4090 must expose the service through a network path
reachable from 192.168.0.166.
```

## Windows Tailscale Join And WSL Echo

Date: 2026-05-05

Windows Tailscale status:

```text
100.93.16.124   xinjiezhang           windows
100.66.203.123  ubuntu-4090-qqtt-ffs  linux
```

Windows `tailscale ping` reached UBUNTU-4090, mostly via DERP(nyc), with samples:

```text
11 ms, 17 ms, 9 ms, 77 ms via DERP(nyc)
119 ms via 172.85.116.88:56372
```

Windows `Test-NetConnection`:

```text
ComputerName     : 100.66.203.123
RemoteAddress    : 100.66.203.123
RemotePort       : 7001
InterfaceAlias   : Tailscale
SourceAddress    : 100.93.16.124
TcpTestSucceeded : True
```

WSL route after Windows Tailscale login:

```text
100.66.203.123 dev eth4 src 100.93.16.124
```

WSL TCP sanity check:

```text
tcp_connect_ok 100.66.203.123 7001 ms 14.96
```

Small-payload ZeroMQ echo:

```bash
conda run --no-capture-output -n demo_2_max \
  python services/ffs_remote/ffs_depth_client.py \
  --endpoint tcp://100.66.203.123:7001 \
  --echo-benchmark \
  --profile 4x3 \
  --fps 1 \
  --duration-s 5 \
  --timeout-ms 2000 \
  --debug
```

Result:

```text
sent=5
ok=5
failed=0
reply_fps=1.00
rtt_ms_p50=9.63
rtt_ms_p95=28.69
```

Large-payload `848x480` echo with Demo 2 default `80 ms` timeout:

```text
sent=247
ok=0
failed=247
error=Again: Resource temporarily unavailable
```

Large-payload `848x480` echo at 5 FPS with 5000 ms timeout:

```text
sent=40
ok=40
failed=0
reply_fps=5.00
rtt_ms_p50=147.95
rtt_ms_p90=155.39
rtt_ms_p95=158.78
request_kb_mean=795.38
response_kb_mean=795.21
payload_mbps=65.15
```

Large-payload `848x480` echo with target 30 FPS and 5000 ms timeout:

```text
sent=63
ok=63
failed=0
reply_fps=6.18
rtt_ms_p50=147.10
rtt_ms_p90=167.08
rtt_ms_p95=238.24
request_kb_mean=795.38
response_kb_mean=795.21
payload_mbps=80.55
```

Interpretation:

```text
Windows Tailscale made the endpoint reachable from WSL. ZeroMQ works through the
userspace Tailscale TCP serve path. However, the current path appears bandwidth
and latency limited for full 848x480 raw IR-pair plus depth_u16 responses:
effective throughput is about 6 FPS for ~1.55 MB round trips, and RTT is around
147 ms. This path is useful for protocol/functional validation, but it is not
fast enough for Demo 2 realtime remote FFS at 30 FPS unless the Tailscale path
becomes direct/faster or the payload is reduced/compressed.
```

## Current Network Conclusion

Date: 2026-05-05

The direct LAN endpoint is not available from LOCAL-5090, and the userspace
Tailscale path is reachable but payload-limited:

```text
LAN IP direct path: unavailable
Tailscale path: reachable, mostly DERP/relay-class throughput
848x480 raw IR pair + depth_u16 round trip: about 1.55 MB/frame
observed payload throughput: about 80.55 Mbps
observed reply FPS: about 6.18 FPS
```

Interpretation:

```text
Full-frame raw remote FFS is not a 30 FPS realtime depth source on this link.
The bottleneck is the transport payload budget, not ZeroMQ message structure.
This does not make native RealSense depth an acceptable formal Demo 2 output.
Formal Demo 2 output must remain FFS-derived. Native RealSense depth with remote
FFS refresh is fallback/debug only.
```

Formal Demo 2 quality paths:

```text
1. Local FFS quality baseline:
   --depth-source ffs
   FFS engine: 20-30-48, valid_iters=4, 848x480->864x480, builderOpt5

2. Remote FFS exact/sparse path:
   --depth-source ffs_remote
   --ffs-remote-return depth_u16|masked_uv_depth|masked_xyz
   The rendered PCD must be built from the returned FFS-derived result.
   No silent fallback to native RealSense depth is allowed.

3. Fallback/debug:
   --depth-source realsense --enable-remote-ffs-quality
   Useful for UI/network debugging only, not professor-facing quality output.
```

## Payload Reduction Plan

Implemented experimental knobs:

```text
--ffs-remote-compress none|zstd|lz4|png
--return depth_u16|depth_float_m|masked_uv_depth|masked_xyz
--mask-fraction for synthetic sparse echo benchmarks
```

Demo 2 fallback/debug quality refresh mode:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2/realtime_masked_edgetam_pcd.py \
  --serial 239222300412 \
  --profile 848x480 \
  --fps 60 \
  --depth-source realsense \
  --enable-remote-ffs-quality \
  --remote-ffs-quality-endpoint tcp://100.66.203.123:7001 \
  --remote-ffs-quality-return masked_uv_depth \
  --remote-ffs-quality-compress none \
  --remote-ffs-quality-interval-ms 200 \
  --init-mode sam31-first-frame \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --pcd-mode masked \
  --render-mode pointcloud \
  --compile-mode vision-reduce-overhead \
  --dtype bfloat16 \
  --pcd-color-mode rgb \
  --debug \
  --profile-cuda-events
```

This mode keeps realtime PCD on local native RealSense depth. Remote FFS is an
asynchronous comparison side channel; the HUD/debug stream reports remote
quality FPS, age, RTT, server time, request KB, and response KB. It is not the
formal Demo 2 FFS-quality output.

Formal remote sparse FFS main path:

```bash
# Remote UBUNTU-4090 server
conda run --no-capture-output -n demo_2_max \
  python services/ffs_remote/ffs_depth_server.py \
  --bind tcp://0.0.0.0:7001 \
  --ffs-repo ../Fast-FoundationStereo \
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

# Local WSL-5090 client
conda run --no-capture-output -n demo_2_max \
  python demo_v2/realtime_masked_edgetam_pcd.py \
  --serial 239222300412 \
  --profile 848x480 \
  --fps 60 \
  --depth-source ffs_remote \
  --ffs-remote-endpoint tcp://100.66.203.123:7001 \
  --ffs-remote-max-inflight 1 \
  --ffs-remote-timeout-ms 5000 \
  --ffs-remote-return masked_uv_depth \
  --init-mode sam31-first-frame \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --pcd-mode masked \
  --render-mode pointcloud \
  --compile-mode vision-reduce-overhead \
  --dtype bfloat16 \
  --depth-min-m 0.2 \
  --depth-max-m 1.5 \
  --pcd-max-points 60000 \
  --pcd-color-mode rgb \
  --debug \
  --profile-cuda-events
```

## Tailscale Direct Echo Retest

Date: 2026-05-05

UBUNTU-4090 reported the tailnet path had moved from DERP/relay to direct UDP:

```text
4090 Tailscale IP: 100.66.203.123
Windows peer:      100.93.16.124
tailscale ping:    direct via 128.59.18.108:41644 in 6-9 ms
```

WSL-5090 local network sanity checks:

```text
ip route get 100.66.203.123:
  100.66.203.123 dev eth4 src 100.93.16.124

tcp connect to 100.66.203.123:7001:
  7.20-11.49 ms, ok

ping 100.66.203.123:
  3/3 received, rtt avg 6.93 ms
```

Default 80 ms realtime deadline echo:

```bash
conda run --no-capture-output -n demo_2_max \
  python services/ffs_remote/ffs_depth_client.py \
  --endpoint tcp://100.66.203.123:7001 \
  --echo-benchmark \
  --profile 848x480 \
  --fps 30 \
  --duration-s 20 \
  2>&1 | tee docs/generated/demo2_remote_ffs_echo_tailscale_direct_20260505.txt
```

Result:

```text
duration_s=20.08
sent=248
ok=0
failed=248
reply_fps=0.00
```

Long-timeout throughput echo:

```bash
conda run --no-capture-output -n demo_2_max \
  python services/ffs_remote/ffs_depth_client.py \
  --endpoint tcp://100.66.203.123:7001 \
  --echo-benchmark \
  --profile 848x480 \
  --fps 30 \
  --duration-s 20 \
  --timeout-ms 5000 \
  2>&1 | tee docs/generated/demo2_remote_ffs_echo_tailscale_direct_timeout5000_20260505.txt
```

Result:

```text
duration_s=20.06
sent=124
ok=124
failed=0
reply_fps=6.18
rtt_ms_p50=159.50
rtt_ms_p90=164.71
rtt_ms_p95=170.01
server_total_ms_p50=1.03
request_kb_mean=795.71
response_kb_mean=795.21
mbps_payload=80.55
```

Interpretation:

```text
Direct Tailscale fixed the small-packet route/latency problem, but full-frame
848x480 raw echo throughput is still about 80.55 Mbps and 6.18 FPS. The 80 ms
Demo 2 deadline is not met for raw depth_u16 full-frame remote FFS.

The bottleneck is now payload throughput / userspace Tailscale serve forwarding
or the network path's effective large-payload bandwidth, not basic reachability.
Do not start formal full-frame depth_u16 remote FFS expecting realtime behavior
on this path. Continue with sparse return, compression, low-res FFS, or a
system tailscaled/direct routing setup before using remote FFS as the main
quality path.
```

Tracking table:

| mode | RTT p50 | reply FPS | request KB | response KB | use |
| --- | ---: | ---: | ---: | ---: | --- |
| raw depth_u16, DERP/relay-era baseline | 147 ms | 6.18 | 795 | 795 | not realtime on current link |
| raw depth_u16, direct Tailscale, 80 ms timeout | n/a | 0.00 | n/a | n/a | misses realtime deadline |
| raw depth_u16, direct Tailscale, 5000 ms timeout | 159.50 ms | 6.18 | 795.71 | 795.21 | not realtime; throughput still 80.55 Mbps |
| remote FFS quality refresh | TBD | TBD | TBD | TBD | fallback/debug only |
| masked_uv_depth | TBD | TBD | TBD | TBD | formal only as main FFS-derived depth path |
| compressed zstd/lz4/png | TBD | TBD | TBD | TBD | payload reduction |
| low-res FFS | TBD | TBD | TBD | TBD | possible realtime probe |

Local implementation checks:

```text
protocol sparse request/response roundtrip: pass
local echo-only masked_uv_depth smoke, 4x3, mask_fraction=0.5:
  reply_fps=5.00
  rtt_ms_p50=1.01
  request_kb_mean=0.83
  response_kb_mean=0.96
  sparse_points_mean=7.00
  strict_engine_contract=true
remote sparse main path code: implemented
check_harness_catalog: pass
check_all quick: pass
```
