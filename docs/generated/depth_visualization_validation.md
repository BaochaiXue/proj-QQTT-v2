# Depth Visualization Validation

Date: 2026-04-07

Environment:

- repo: `C:\Users\zhang\proj-QQTT`
- conda env: `qqtt-ffs-compat`
- native aligned case: `data/native_30_static`
- ffs aligned case: `data/ffs_30_static`

## Commands Run

Per-camera diagnostic panels:

```bash
python scripts/harness/visual_compare_depth_panels.py --aligned_root C:\Users\zhang\proj-QQTT\data --realsense_case native_30_static --ffs_case ffs_30_static --output_dir C:\Users\zhang\proj-QQTT\data\panels_native_30_static_vs_ffs_30_static --camera_ids 0 1 2 --frame_start 0 --frame_end 9 --write_mp4 --use_float_ffs_depth_when_available
```

Cross-view reprojection:

```bash
python scripts/harness/visual_compare_reprojection.py --aligned_root C:\Users\zhang\proj-QQTT\data --realsense_case native_30_static --ffs_case ffs_30_static --output_dir C:\Users\zhang\proj-QQTT\data\reprojection_native_30_static_vs_ffs_30_static --frame_start 0 --frame_end 9 --camera_pair 0,1 --camera_pair 0,2 --write_mp4 --use_float_ffs_depth_when_available
```

Fused cloud comparison:

```bash
python scripts/harness/visual_compare_depth_video.py --aligned_root C:\Users\zhang\proj-QQTT\data --realsense_case native_30_static --ffs_case ffs_30_static --output_dir C:\Users\zhang\proj-QQTT\data\comparison_native_30_static_vs_ffs_30_static_diagnostic --renderer fallback --render_mode neutral_gray_shaded --views oblique top side --write_mp4 --use_float_ffs_depth_when_available
```

2x3 tabletop-focused fused cloud comparison:

```bash
python scripts/harness/visual_compare_depth_video.py --aligned_root C:\Users\zhang\proj-QQTT\data --realsense_case native_30_static --ffs_case ffs_30_static --output_dir C:\Users\zhang\proj-QQTT\data\comparison_native_30_static_vs_ffs_30_static_grid --renderer fallback --preset tabletop_compare_2x3 --write_mp4 --use_float_ffs_depth_when_available
```

## Outputs Produced

Per-camera diagnostic panels:

- `data/panels_native_30_static_vs_ffs_30_static/camera_0/frames/*.png`
- `data/panels_native_30_static_vs_ffs_30_static/camera_1/frames/*.png`
- `data/panels_native_30_static_vs_ffs_30_static/camera_2/frames/*.png`
- `data/panels_native_30_static_vs_ffs_30_static/camera_0/panels.mp4`
- `data/panels_native_30_static_vs_ffs_30_static/camera_1/panels.mp4`
- `data/panels_native_30_static_vs_ffs_30_static/camera_2/panels.mp4`
- `data/panels_native_30_static_vs_ffs_30_static/summary.json`

Cross-view reprojection:

- `data/reprojection_native_30_static_vs_ffs_30_static/pair_0_to_1/frames/*.png`
- `data/reprojection_native_30_static_vs_ffs_30_static/pair_0_to_1/reprojection.mp4`
- `data/reprojection_native_30_static_vs_ffs_30_static/pair_0_to_2/frames/*.png`
- `data/reprojection_native_30_static_vs_ffs_30_static/pair_0_to_2/reprojection.mp4`
- `data/reprojection_native_30_static_vs_ffs_30_static/summary_metrics.json`

Fused cloud comparison:

- `data/comparison_native_30_static_vs_ffs_30_static_diagnostic/view_oblique/...`
- `data/comparison_native_30_static_vs_ffs_30_static_diagnostic/view_top/...`
- `data/comparison_native_30_static_vs_ffs_30_static_diagnostic/view_side/...`
- `data/comparison_native_30_static_vs_ffs_30_static_diagnostic/comparison_metadata.json`
- `data/comparison_native_30_static_vs_ffs_30_static_diagnostic/metrics.json`

2x3 tabletop-focused fused cloud comparison:

- `data/comparison_native_30_static_vs_ffs_30_static_grid/grid_2x3_frames/*.png`
- `data/comparison_native_30_static_vs_ffs_30_static_grid/videos/grid_2x3.mp4`
- `data/comparison_native_30_static_vs_ffs_30_static_grid/view_cam0/...`
- `data/comparison_native_30_static_vs_ffs_30_static_grid/view_cam1/...`
- `data/comparison_native_30_static_vs_ffs_30_static_grid/view_cam2/...`

## What Was Validated

- Single-camera panel generation works in two-case fallback mode.
- Panel outputs include native depth, ffs depth, absolute difference, valid-mask comparison, shaded depth, and ROI crops.
- Cross-view reprojection works in two-case fallback mode and writes per-pair frame panels plus summary metrics.
- Fused cloud comparison now supports multi-view output (`oblique`, `top`, `side`) and geometry-first render mode (`neutral_gray_shaded`).
- Fused cloud comparison now also supports:
  - real calibrated camera-pose views
  - tabletop focus
  - a single 2x3 comparison layout
  - world-space tabletop cropping before framing
  - geometry-aware camera distance scaling
  - denser splat-like fallback rendering

## Why This Workflow Is Easier To Read

- Tabletop crop prevents the full room bounds from dominating the frame.
- View distance scaling moves each camera-direction view closer to the tabletop ROI in 3D, instead of relying mainly on a 2D post-crop.
- `neutral_gray_shaded` is now the recommended default because it emphasizes geometry differences more clearly than RGB texture.
- `color_by_rgb` remains available as a reference mode when color context is helpful.

## Known Limitations

- Two-case fallback comparison is appropriate for static or near-static scenes only. It is not same-take ground truth.
- The current visualization pipeline still uses pinhole `K` matrices without explicit distortion coefficients.
- A dedicated temporal-stability script has not been added yet; static-scene stability must currently be inferred from panel videos and reprojection consistency.
