# Demo 3.1 / 3.2 Exact Batch Render Bundle

## Goal

Ensure rendered tracker output is produced only from the same three-camera
`batch=3` group:

- depth group
- object/controller masks
- tracker query input
- tracker result
- fused PCD / marker lift inputs
- final rendered packet

All of those artifacts must share the same `group_id` in the default exact
target path.

## Problem

Recent live profiles show tracker inputs and surface anchors are published, and
the tracker child publishes results, but rendered groups remain zero. The
profile reports `missing-exact` because tracker-result-gated rendering looks in
`demo31_pending_fusion_bundles`, while the active path publishes tracker inputs
from the already-fused `fused_packet` path. That leaves the exact fusion-bundle
cache empty.

## Plan

1. Make tracker-result-gated mode cache the complete depth/mask bundle whenever
   tracker input is published from the fused-packet path.
2. Do not let the default exact path fall back to nearest groups.
3. Preserve the current explicit nearest debug mode.
4. Add focused tests that verify the fused-packet path caches a pending fusion
   bundle and that the tracker result renders only the exact matching group.

## Validation

- Focused Demo 3.1/3.2 contract tests.
- Deterministic harness quick check if time allows.
