# Demo 2.2 Stage Throughput Profile Metrics

## Goal

Add measurement-only Demo 2.2 profile fields that distinguish single-group
latency from pipeline throughput period before changing the scheduler.

## Scope

- Add publish-period summaries for capture, GPU owner, raw fusion, filtered
  output, render packet publication, and actual render.
- Record when a filtered display packet is published to the render buffer.
- Keep algorithms, scheduling, masks, FFS, EdgeTAM, filter, and renderer
  behavior unchanged.

## Validation

- Targeted Demo 2.2 smoke tests.
- `git diff --check`.

## PR3b Scheduler Experiment

PR3 measured worse than the PR2 batch-vision baseline because the current
`overlapped-stages` implementation is mask-gated:

```text
EdgeTAM(group N)
  -> FFS(group N)
  -> same-group join/fusion
```

Keep PR2 `single-owner + batch vision` as the stable demo/default path. Treat
PR3 as an experimental scheduler branch only.

Add explicit stage scheduler modes:

```text
mask-gated
  Current PR3 negative-control behavior.

edge-start
  Reserve the same group for FFS when EdgeTAM starts processing that group.

bounded-lookahead
  Allow FFS to process a bounded number of future groups while preserving exact
  group_id joins.
```

Required PR3b diagnostics:

```text
ffs_stage.request_s / start_s / publish_s / reason
edgetam_stage.request_s / start_s / publish_s / reason
stage_join.depth_ready_before_mask
stage_join.depth_wait_after_mask_ms
stage_join.mask_wait_after_depth_ms
stage_join.same_group_join_latency_ms
stage_pipeline.depth_ready_before_mask_ratio
stage_pipeline.mean_depth_wait_after_mask_ms
stage_pipeline.scheduler_mode
```
