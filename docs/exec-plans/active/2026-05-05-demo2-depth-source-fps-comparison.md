# 2026-05-05 Demo 2 Depth Source FPS Comparison

Compare current Demo 2 FPS for two local 5090 Laptop paths under identical
HF EdgeTAM settings:

1. formal FFS-derived path:
   `20-30-48`, `valid_iters=4`, `848x480 -> 864x480`,
   `builderOptimizationLevel=5`, `--depth-source ffs`
2. fallback/debug native path:
   `--depth-source realsense`

Both runs use:

- D455 serial `239222300412`
- `848x480@60`
- live first-frame SAM3.1 init
- `--track-mode object-only`
- `--object-prompt "stuffed animal"`
- HF EdgeTAM `--compile-mode vision-reduce-overhead`
- `--dtype bfloat16`
- `--pcd-mode masked`
- `--render-mode none`
- `--profile-cuda-events`

The first comparison is no-render so the result isolates capture, EdgeTAM,
depth, and masked PCD without WSLg/Open3D render overhead.

## Validation

- Save raw logs under `docs/generated/`.
- Summarize medians in
  `docs/generated/demo2_depth_source_fps_comparison.md`.
- If files change, run `python scripts/harness/check_all.py`.
