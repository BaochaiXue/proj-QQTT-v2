# Demo 2 Remote FFS Real-IR Compression Matrix

Date: 2026-05-07

## Scope

This report uses the formal remote FFS path:

```text
WSL-5090 RealSense D455 real IR left/right Y8
-> ZeroMQ binary multipart request
-> Ubuntu-4090 strict FFS TensorRT server
-> real full-frame depth_u16 response
```

Synthetic echo is not used as handoff evidence.

Target:

```text
single camera realtime: 45 FPS
three camera realtime: 15 FPS per camera, aggregate 45 camera-FPS
```

Current server state during these runs:

```text
server_return_type = depth_u16
server_response_compression = lz4
endpoint = tcp://192.168.0.162:7001
serial = 239222300412
profile = 848x480@30
```

Because the Ubuntu-4090 server was not switched during this WSL run, this is a
request-compression matrix with fixed `response_compression=lz4`. Response
`png` and `none` variants still require 4090-side server restarts.

## Results

| Request Compress | Response Compress | OK / Failed | Reply FPS | RTT p50 / p90 / p95 ms | Server FFS / Align / Total p50 ms | Request KB | Response KB | Depth Nonzero Mean | First Depth Preview | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `lz4` | `lz4` | `255 / 0` | `11.83` | `68.99 / 94.29 / 139.74` | `13.54 / 4.38 / 19.37` | `620.26` | `332.53` | `362996.32` | `docs/generated/demo2_real_ir_remote_depth_20260507_134511_frame000000_depth_preview.png` | best current mode |
| `zstd` | `lz4` | `207 / 0` | `9.67` | `79.35 / 138.20 / 179.59` | `14.10 / 4.30 / 20.20` | `504.59` | `331.64` | `363110.53` | `docs/generated/demo2_real_ir_remote_depth_20260507_150051_frame000000_depth_preview.png` | smaller request, slower |
| `none` | `lz4` | `208 / 0` | `9.72` | `84.91 / 120.09 / 139.06` | `13.97 / 4.28 / 19.26` | `796.04` | `331.80` | `363656.38` | `docs/generated/demo2_real_ir_remote_depth_20260507_150130_frame000000_depth_preview.png` | raw request too large |
| `png` | `lz4` | `133 / 0` | `6.20` | `69.23 / 98.45 / 105.03` | `14.03 / 4.29 / 24.99` | `279.34` | `331.33` | `363122.93` | `docs/generated/demo2_real_ir_remote_depth_20260507_150207_frame000000_depth_preview.png` | payload small but CPU cost too high |

## Interpretation

```text
Best current transport setting:
  request_compression = lz4
  response_compression = lz4
  endpoint = tcp://192.168.0.162:7001

Server compute is not the bottleneck:
  server_total_p50 is about 19-20 ms for lz4/zstd/none request modes.

Transport remains the bottleneck:
  best RTT p50 is about 69 ms and best reply FPS is about 11.8.
```

This is a semantic pass and realtime fail for full `depth_u16`.

`zstd` reduced request size from about `620 KB` to `505 KB`, but the extra CPU
compression/decompression cost lowered throughput. `png` reduced request size
to about `279 KB`, but server/client image encode/decode overhead lowered
throughput further and increased server total time.

## Pending Response Variants

These still require Ubuntu-4090 server restarts:

```text
request lz4 / response png
request none / response none
request png / response png
```

Do not treat this report as the final full compression matrix until those
server-side response variants have been run.
