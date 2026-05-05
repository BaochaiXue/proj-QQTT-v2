# Remote FFS Echo Benchmark CLI

## Goal

Make the already-added remote FFS client directly runnable for the two-machine echo-only validation step.

## Scope

- Add a CLI to `services/ffs_remote/ffs_depth_client.py`.
- Keep the existing `FfsRemoteDepthClient` API unchanged for Demo 2.
- Add tests for CLI help and the benchmark summary path with a fake client.
- Update docs with the 4090 TensorRT engine compatibility warning and echo command.

## Validation

- Focused unit tests for the remote client/protocol path.
- `python scripts/harness/check_all.py`.
