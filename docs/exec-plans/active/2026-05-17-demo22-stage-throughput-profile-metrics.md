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
