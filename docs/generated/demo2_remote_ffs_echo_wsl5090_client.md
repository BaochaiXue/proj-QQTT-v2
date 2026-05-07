# Demo 2 Remote FFS LAN Echo Client Probe

Date: 2026-05-07

## Context

Machine:

```text
WSL Ubuntu RTX 5090 Laptop
repo=/home/zhangxinjie/proj-QQTT-v2
env=demo_2_max
task=Phase 2 LAN echo client only
```

The Ubuntu-4090 side was assumed to be running the echo-only server on port
`7001`. This probe did not start RealSense, EdgeTAM, Open3D, or any FFS server.

## Commands

Primary endpoint:

```bash
conda run --no-capture-output -n demo_2_max \
  python services/ffs_remote/ffs_depth_client.py \
  --endpoint tcp://192.168.0.162:7001 \
  --echo-benchmark \
  --profile 848x480 \
  --fps 30 \
  --duration-s 20 \
  --timeout-ms 5000 \
  --debug
```

Fallback endpoint, tested because the primary endpoint did not meet the pass
line:

```bash
conda run --no-capture-output -n demo_2_max \
  python services/ffs_remote/ffs_depth_client.py \
  --endpoint tcp://128.59.19.35:7001 \
  --echo-benchmark \
  --profile 848x480 \
  --fps 30 \
  --duration-s 20 \
  --timeout-ms 5000 \
  --debug
```

## Echo Results

| Endpoint | Sent | OK | Failed | Reply FPS | RTT p50 | RTT p90 | RTT p95 | Request KB mean | Response KB mean | Payload Mbps | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `tcp://192.168.0.162:7001` | 207 | 207 | 0 | 10.34 | 84.74 ms | 133.94 ms | 200.01 ms | 795.71 | 795.88 | 134.84 | fail throughput/latency |
| `tcp://128.59.19.35:7001` | 150 | 150 | 0 | 7.47 | 131.14 ms | 138.88 ms | 143.11 ms | 795.71 | 795.88 | 97.46 | fail throughput/latency |

Pass line:

```text
ok = 600
failed = 0
reply_fps ~= 30
rtt_ms_p50 ideally < 10 ms on LAN
```

Both endpoints were reachable and had `failed=0`, but neither endpoint met the
30 FPS / low-latency echo pass line.

## Network Diagnostics

`192.168.0.162`:

```text
ping: 4/4 received, avg 124.805 ms, min 44.361 ms, max 173.895 ms
nc: Connection to 192.168.0.162 7001 succeeded
```

`128.59.19.35`:

```text
ping: 4/4 received, avg 3.843 ms, min 3.213 ms, max 4.506 ms
nc: Connection to 128.59.19.35 7001 succeeded
```

The `128.59.19.35` ICMP path is low latency, but the full-frame ZeroMQ echo
payload still only reached `7.47` reply FPS. This suggests the blocker is not
simple TCP reachability; it may be payload throughput, server-side echo path,
routing for large TCP payloads, or interface selection.

## Artifacts

```text
docs/generated/demo2_remote_ffs_echo_wsl5090_client_192_168_0_162.log
docs/generated/demo2_remote_ffs_echo_wsl5090_client_128_59_19_35.log
docs/generated/demo2_remote_ffs_echo_wsl5090_network_192_168_0_162.log
docs/generated/demo2_remote_ffs_echo_wsl5090_network_128_59_19_35.log
```

## Decision

```text
Phase 2 LAN echo client: FAIL for formal remote FFS handoff.
Reason: TCP connects and echo replies work, but full-frame echo throughput is
only 7-10 FPS and RTT p50 is 85-131 ms.
```

Do not ask the Ubuntu-4090 side to switch from echo-only server to strict
TensorRT FFS server yet. The next step should be to diagnose why the full-frame
echo payload is capped around `97-135 Mbps` despite TCP reachability, especially
because `128.59.19.35` has low ICMP latency.
