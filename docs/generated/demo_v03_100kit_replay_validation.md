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
- latest attempt: 2026-05-09
- result: non-interactive SSH is not currently available from this 5090 WSL
  session; `BatchMode=yes` returned permission denied for both
  `xinjie@192.168.0.162` and `zhangxinjie@192.168.0.162`.
- prepared HTTP fallback tarball:
  `/tmp/demo_v0_3_ir_triplet_100kits_848x480.tgz`
- tarball size: 126M
- tarball sha256:
  `192b07f5b5f0564ed3fa32377d833e291a5e1c0bb7422cdf30d64419b707480d`
- candidate 5090 IPs:
  `100.93.16.124`, `192.168.0.166`, `fd7a:115c:a1e0::453a:107c`
- next command after SSH credentials are available:

```bash
rsync -aP --info=progress2 \
  result/demo_v0_3_ir_triplet_100kits_848x480/ \
  xinjie@192.168.0.162:/home/xinjie/proj-QQTT-v2/result/demo_v0_3_ir_triplet_100kits_848x480/
```

HTTP fallback from 5090 when 4090 is ready:

```bash
python3 -m http.server 8799 --directory /tmp
```

Then on 4090:

```bash
curl -fL http://192.168.0.166:8799/demo_v0_3_ir_triplet_100kits_848x480.tgz \
  -o /tmp/demo_v0_3_ir_triplet_100kits_848x480.tgz
curl -fL http://192.168.0.166:8799/demo_v0_3_ir_triplet_100kits_848x480.tgz.sha256 \
  -o /tmp/demo_v0_3_ir_triplet_100kits_848x480.tgz.sha256
cd /tmp
sha256sum -c demo_v0_3_ir_triplet_100kits_848x480.tgz.sha256
mkdir -p /home/xinjie/proj-QQTT-v2/result
tar -C /home/xinjie/proj-QQTT-v2/result -xzf /tmp/demo_v0_3_ir_triplet_100kits_848x480.tgz
```

## Branch / Merge Rhythm

- current merge stance: do not merge v0.3 staged server/client to `main` yet.
- P0 transfer: pending
- P1 4090 local batch1/batch3 profile: pending
- P2 clean feature branch implementation: local 5090 implementation/tests pass
- P3 4090 7003 smoke: pending
- P4 remote 100-kit matrix: pending
- merge gate for staged server/client: P4 pass only

Required pass condition for final merge:

```text
measured_completed_kits=100
measured_failed_kits=0
measured_stale_kits=0
completed_kit_fps_mean >= 15
completed_camera_depth_fps_mean >= 45
depth_nonzero_cam0_min > 0
depth_nonzero_cam1_min > 0
depth_nonzero_cam2_min > 0
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
