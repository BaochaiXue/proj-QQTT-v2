# Real IR Remote Depth Benchmark

## Goal

Make the remote FFS handoff validation use real RealSense IR stereo frames and real server-computed depth, instead of synthetic echo payloads.

## Scope

- Add a `--real-ir-depth-benchmark` mode to `services/ffs_remote/ffs_depth_client.py`.
- Keep `--echo-benchmark` as TCP/protocol sanity only.
- Preserve the existing `FfsRemoteDepthClient` API used by Demo 2.
- Start the 4090 strict TensorRT server with `depth_u16` and `lz4` response compression.
- Record the 4090 server startup proof under `docs/generated/`.

## Validation

- Focused remote client smoke tests.
- `python -m py_compile services/ffs_remote/ffs_depth_client.py services/ffs_remote/ffs_depth_server.py`
- `python scripts/harness/check_all.py`

## Outcome

- Added the WSL-client-only `--real-ir-depth-benchmark` path.
- Preserved 4090 as server-only: no RealSense capture, no EdgeTAM/SAM/UI.
- Started strict 4090 FFS server on `tcp://0.0.0.0:7001` with `depth_u16` and `lz4`.
- Recorded server proof in `docs/generated/demo2_remote_ffs_4090_real_depth_server.md`.
- Validation passed:
  - py_compile: pass
  - focused remote client tests: pass
  - `python scripts/harness/check_all.py`: pass, 132 tests
