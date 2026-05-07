# Demo 2 Remote FFS Masked UV Depth Realtime Matrix

Date: 2026-05-07

## Target

```text
single camera realtime: >=45 FPS
three camera realtime: >=15 FPS per camera, aggregate >=45 camera-FPS
```

## Scope

This report is for the realtime hot-path candidate:

```text
WSL-5090 RealSense D455 + SAM3.1 first-frame init + HF EdgeTAM
-> object mask
-> real IR left/right + mask sent to Ubuntu-4090
-> Ubuntu-4090 FFS TensorRT
-> masked_uv_depth/lz4 sparse depth response
-> WSL-5090 masked PCD
```

Full `depth_u16/lz4` is retained as the semantic correctness baseline. It has
already failed realtime throughput:

```text
best full-depth completed FPS = 14.82
target = 45 FPS
```

## Server Probe

A short real-IR probe against `tcp://192.168.0.162:7001` showed the server has
been switched to sparse/masked mode:

```text
response_kb_mean = 0.90 KB
response_compression = lz4
depth_shape = 0x0 in the full-depth client summary
```

The `0x0` full-depth shape is expected for the client probe because the server
returned sparse `masked_uv_depth`, while the probe client only counts nonzero
full-frame depth for full-depth modes.

## Demo 2 No-Render Attempt

Command:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2/realtime_masked_edgetam_pcd.py \
  --serial 239222300412 \
  --profile 848x480 \
  --fps 60 \
  --depth-source ffs_remote \
  --ffs-remote-endpoint tcp://192.168.0.162:7001 \
  --ffs-remote-max-inflight 1 \
  --ffs-remote-timeout-ms 5000 \
  --ffs-remote-return masked_uv_depth \
  --ffs-remote-compress lz4 \
  --init-mode sam31-first-frame \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --pcd-mode masked \
  --render-mode none \
  --compile-mode vision-reduce-overhead \
  --dtype bfloat16 \
  --duration-s 45 \
  --debug \
  --profile-cuda-events
```

Result:

```text
status = fail-fast before remote PCD benchmark
reason = SAM3.1 did not produce a mask for label 'stuffed animal'
remote_rtt_ms = 0.00 in runtime logs because no mask packet reached remote FFS
pcd_fps = 0.0
```

Log path:

```text
/tmp/demo2_remote_masked_uv_depth_lz4_inflight1_no_render_5090.log
```

## Decision

```text
masked_uv_depth/lz4 server readiness: pass
Demo 2 masked no-render FPS result: not measured
blocker: live SAM3.1 object initialization failed for "stuffed animal"
```

This is not a transport failure. It is the correct no-fallback behavior for a
formal live SAM3.1 path. To run the masked realtime matrix, place a visible
object that SAM3.1 can initialize with the chosen prompt, or explicitly choose a
different prompt for the current scene.

## Current Runtime Limitation

Demo 2 runtime currently rejects:

```text
--ffs-remote-max-inflight != 1
```

The `services/ffs_remote/ffs_depth_client.py` utility has benchmark-only
multi-inflight support, but `demo_v2/realtime_masked_edgetam_pcd.py` still needs
a runtime async remote-depth worker before a real `1/2/4/8` Demo matrix can be
run.

