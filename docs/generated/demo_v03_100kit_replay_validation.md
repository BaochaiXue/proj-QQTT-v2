# Demo v0.3 100-Kit IR Triplet Replay Validation

Status: 5090 data preparation complete. 4090 local profiles and 5090-to-4090
remote profiles are still pending.

## Dataset

- source folder:
  `result/demo_v0_2_data_ir_triplet_replay_848x480_still_object_round8`
- 100-kit folder: `result/demo_v0_3_ir_triplet_100kits_848x480`
- camera count: 3
- kit count: 100
- width/height: 848x480
- capture cadence: 15 kit-FPS
- kit period: 66.6667 ms
- unique source kits: 21
- cycled: true
- folder size on 5090: 128M
- file count: 603
- preparation log:
  `demo_v03_prepare_100_ir_triplets_5090_20260508_151459.log`
- checksum manifest:
  `demo_v03_ir_triplet_100kits_sha256_5090.txt`

## Warmup

- warmup kits: 20
- measured kits: 100
- warmup included in stats: no

## 4090 Transfer

- status: pending
- attempted host: `xinjie@192.168.0.162`
- result: non-interactive SSH is not currently available from this 5090 WSL
  session; `BatchMode=yes` returned permission denied after accepting the host
  key.
- next command after SSH credentials are available:

```bash
rsync -aP --info=progress2 \
  result/demo_v0_3_ir_triplet_100kits_848x480/ \
  xinjie@192.168.0.162:/home/xinjie/proj-QQTT-v2/result/demo_v0_3_ir_triplet_100kits_848x480/
```

## 4090 Local Batch1 Profile

- measured_completed_kits:
- ffs_triplet_ms avg/min/max/p50/p90/p95/p99:
- cam0_ms avg/p90/p99:
- cam1_ms avg/p90/p99:
- cam2_ms avg/p90/p99:

## 4090 Batch3

- build:
- validate:
- profile:
- batch3 ffs_triplet_ms avg/min/max/p50/p90/p95/p99:
- speedup vs batch1:

## 5090 To 4090 Remote 15 FPS Profile

- best max-inflight:
- measured_completed_kits:
- failed:
- stale:
- completed_kit_fps_mean:
- completed_camera_depth_fps_mean:
- kit_e2e_ms avg/min/max/p50/p90/p95/p99:
- server_decode_ms avg/p90/p99:
- server_ffs_triplet_ms avg/p90/p99:
- server_postprocess_encode_ms avg/p90/p99:
- server_total_ms avg/p90/p99:
- request_kb avg/p90/p99:
- reply_kb avg/p90/p99:

## Saturated Capacity Profile

- best max-inflight:
- completed_kit_fps_mean:
- completed_camera_depth_fps_mean:
- p99 latency:
- bottleneck stage:

## Conclusion

- official server mode:
- recommended max-inflight for 15 FPS:
- capacity bottleneck:
