# Demo 3.1/3.2 Tracker-Result-Gated Fusion

## Goal

Avoid wasting CPU fusion/filter work on depth/mask groups that will never be rendered by Demo 3.1/3.2. When tracking is enabled, RealSense/FFS depth and EdgeTAM/HF masks should publish bounded tracking inputs and cache only the pending fusion inputs. A fresh tracker result then triggers the matching PCD fusion and Open3D render.

## Constraints

- Keep rendered FPS honest: every rendered frame must be driven by a new tracker result.
- Do not reuse stale tracker output to render a new PCD frame.
- Keep the existing exact-then-nearest group matching policy when an exact PCD group was evicted or missed.
- Keep queues bounded so slow trackers cannot grow memory without limit.
- Preserve existing Demo 3.1/3.2 tracker backends, query semantics, overlay logic, and render profiling fields.

## Implementation Plan

1. Add a `pcd_fusion_trigger` contract/CLI knob with default `tracker-result`.
2. Add a bounded pending fusion bundle cache, parallel to the existing pending render packet cache.
3. In Demo 3.1/3.2, intercept completed depth+mask groups and defer expensive PCD fusion when tracker-result gating is enabled.
4. On a fresh tracker result, take the same group from the pending fusion cache, or the nearest cached group when allowed, then run fusion/filter exactly once and render it.
5. Report pending fusion queue size, drops, exact/nearest matches, and tracker-triggered fusion timings in profile snapshots.
6. Keep the legacy depth/mask-ready fusion path available via CLI for A/B profiling.

## Validation

- Unit tests cover default contract fields, dry-run output, bounded pending fusion cache, and tracker-result-triggered fusion.
- Run Demo 3.1 contract tests.
- Run `scripts/harness/check_all.py` in `demo_2_max`.
