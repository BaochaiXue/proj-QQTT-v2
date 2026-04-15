# 2026-04-14 Pointcloud Tabletop Depth Default

## Goal

Tighten the default depth filter used by the repo's general fused point-cloud generation workflows so tabletop scenes drop far-background geometry by default.

Adopt the effective filtering contract:

- valid depth must remain strictly `> 0`
- default point-cloud depth max becomes `1.5m`

for the generic fused point-cloud compare/export paths that still default to a looser full-scene range today.

## Non-Goals

- no change to `calibrate.pkl`
- no change to aligned-case depth files on disk
- no change to 2D depth panel defaults
- no change to object-centric workflows that already use their own tighter tabletop range
- no semantic change to explicit user-provided `--depth_min_m` / `--depth_max_m` overrides

## Files To Touch

- shared point-cloud workflow defaults/constants
- fused point-cloud compare / rerun / triplet workflow entrypoints
- harness CLIs that expose those defaults
- smoke tests and operator docs

## Implementation Plan

1. centralize a shared default depth range for generic fused point-cloud workflows
2. switch the general point-cloud compare, rerun compare, and triplet PLY workflows to default to that range
3. keep existing object-centric workflows unchanged where they already use stricter tabletop tuning
4. record the active depth range explicitly in workflow summaries/metadata where missing
5. update smoke coverage and operator docs to match the new default behavior

## Validation Plan

- `python -m unittest -v tests.test_visual_compare_depth_video_smoke`
- `python -m unittest -v tests.test_rerun_compare_workflow_smoke`
- `python -m unittest -v tests.test_triplet_ply_compare_workflow_smoke`
- `python scripts/harness/check_all.py`
