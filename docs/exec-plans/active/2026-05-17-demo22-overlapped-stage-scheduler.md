# Demo 2.2 Overlapped Stage Scheduler

## Goal

Add a Demo 2.2 `overlapped-stages` GPU pipeline mode that improves throughput by running FFS depth, EdgeTAM batch-vision tracking, and fusion/filter publication as separate cross-group stages.

This PR does not change packet residency, renderer behavior, FFS TensorRT binding strategy, or EdgeTAM model math. It keeps existing CPU/Numpy depth and point-cloud packets so the scheduler change can be evaluated independently.

## Current Baseline

The PR2 batch-vision checkpoint shows the single-owner GPU section is still serialized:

```text
group N:
  FFS(N)
  EdgeTAM batch vision + per-camera decode(N)
  fusion/filter(N)
```

Measured pointcloud profile:

```text
filter_fps: 5.689
gpu_owner p50: 168.104 ms
FFS p50: 72.725 ms
EdgeTAM batch/decode stage: about 90-95 ms
```

The PR3 expectation is throughput period near the max stage time, not lower per-group end-to-end latency:

```text
period ~= max(FFS stage, EdgeTAM stage, fusion/filter stage)
```

## Scope

- Add `GPU_PIPELINE_MODE_OVERLAPPED_STAGES = "overlapped-stages"`.
- Add a public Demo 2.2 flag `--experimental-overlapped-stages`.
- Add stage packet contracts and a same-group join buffer.
- Wire workers:
  - capture dispatch
  - FFS stage
  - EdgeTAM stage
  - same-group join + fusion publication
- Add profile fields for stage wall times, join counts, stale drops, and effective stage period.
- Add deterministic tests for the scheduler contract and join semantics.

## Non-Goals

- No GPU-resident PCD packet rewrite.
- No FFS CUDA Graph/static-buffer runner.
- No renderer changes.
- No EdgeTAM decoder parallelization.
- No change to the existing `staged` probe semantics.

## Correctness Constraints

- EdgeTAM is stateful; its stage uses one worker and processes group ids monotonically.
- Fusion must require `depth.group_id == mask.group_id == capture.group_id`.
- Stale groups may be dropped, but mismatched depth/mask groups must never be joined.
- The existing `staged` mode remains the old FFS-then-parallel-EdgeTAM probe.

## Validation

```bash
conda run -n demo_2_max --no-capture-output python -m py_compile \
  qqtt/demo/three_view_masked_fused_pcd_runtime.py \
  demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py

conda run -n demo_2_max --no-capture-output python -m unittest -v \
  tests.test_demo_v2_2_async_filtered_fused_pcd_smoke

conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py

git diff --check
```

Real hardware A/B after deterministic checks:

```text
A: PR2 single-owner batch vision baseline
B: PR3 --experimental-overlapped-stages
```

Success for PR3 is a valid full Demo 2.2 profile with same-group joins, no semantic mismatch, and a lower `display_packet_period_ms` than the PR2 single-owner baseline.
