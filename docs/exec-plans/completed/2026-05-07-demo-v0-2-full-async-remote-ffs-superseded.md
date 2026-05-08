# Demo v0.2 Full Async Remote FFS

## Status

Superseded by `docs/exec-plans/active/2026-05-08-demo-v0-3-100kit-staged-remote-ffs.md`.
Do not extend Demo v0.2 as the active benchmark track. Keep this plan only as
historical context and as documentation for the existing v0.2 replay source
folder that Demo v0.3 may normalize into a fixed 100-kit dataset.

## Goal

Build an independent full-async remote FFS capacity benchmark:

```text
WSL-5090 real IR capture/replay
-> DEALER async client with max_inflight
-> Ubuntu-4090 ROUTER async server
-> FFS TensorRT depth_u16
-> DEALER async receive
```

## Scope

- Add `services/ffs_remote/async_protocol_v02.py`.
- Add `services/ffs_remote/ffs_depth_async_server_v02.py`.
- Add `demo_v0_2/async_remote_ffs_triplet_client.py`.
- Add protocol unit tests.
- Add generated documentation and commands.
- Add a `--server-pipeline-mode staged` server mode that separates
  decode/decompress, FFS, and reply encode/compress into distinct queues and
  reports stage timing fields.

## Non-Goals

- No masks.
- No SAM3.1.
- No EdgeTAM.
- No PCD/rendering.
- Do not change Demo 2 runtime.
- Keep the existing `7001` server path untouched; v0.2 uses `7002`.

## Targets

```text
single camera: >=45 camera-depth-FPS
three cameras: >=15 kit-FPS, aggregate >=45 camera-depth-FPS
```

## Pipeline Ablation

```text
fused-worker:
  one worker performs decode -> FFS -> encode for a full request

staged:
  ROUTER receive -> decode worker -> FFS worker -> encode worker -> ROUTER send
```

The staged mode tests whether throughput is limited by the slowest overlapped
stage, while latency is the sum of stage delays plus queueing.

## Validation

- `python -m py_compile demo_v0_2/async_remote_ffs_triplet_client.py services/ffs_remote/async_protocol_v02.py services/ffs_remote/ffs_depth_async_server_v02.py`
- `conda run --no-capture-output -n SAM21-max python -m unittest -v tests.test_demo_v02_async_protocol`
- `conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py`
