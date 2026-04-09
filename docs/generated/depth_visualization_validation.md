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

Single-frame object-centric coverage-aware side-by-side orbit comparison:

```bash
python scripts/harness/visual_compare_turntable.py --aligned_root C:\Users\zhang\proj-QQTT\data --realsense_case native_30_static --ffs_case ffs_30_static --frame_idx 0 --output_dir C:\Users\zhang\proj-QQTT\data\turntable_coverage_native_30_static_vs_ffs_30_static_frame_0000 --renderer fallback --scene_crop_mode auto_object_bbox --orbit_mode observed_hemisphere --num_orbit_steps 24 --projection_mode perspective --fps 12
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

Single-frame object-centric coverage-aware side-by-side orbit comparison:

- `data/turntable_coverage_native_30_static_vs_ffs_30_static_frame_0000/scene_overview_with_cameras.png`
- `data/turntable_coverage_native_30_static_vs_ffs_30_static_frame_0000/frames_geom/*.png`
- `data/turntable_coverage_native_30_static_vs_ffs_30_static_frame_0000/frames_rgb/*.png`
- `data/turntable_coverage_native_30_static_vs_ffs_30_static_frame_0000/frames_support/*.png`
- `data/turntable_coverage_native_30_static_vs_ffs_30_static_frame_0000/orbit_compare_geom.mp4`
- `data/turntable_coverage_native_30_static_vs_ffs_30_static_frame_0000/orbit_compare_rgb.mp4`
- `data/turntable_coverage_native_30_static_vs_ffs_30_static_frame_0000/orbit_compare_support.mp4`
- `data/turntable_coverage_native_30_static_vs_ffs_30_static_frame_0000/turntable_keyframes_geom.png`
- `data/turntable_coverage_native_30_static_vs_ffs_30_static_frame_0000/turntable_keyframes_rgb.png`
- `data/turntable_coverage_native_30_static_vs_ffs_30_static_frame_0000/turntable_keyframes_support.png`
- `data/turntable_coverage_native_30_static_vs_ffs_30_static_frame_0000/support_metrics.json`
- `data/turntable_coverage_native_30_static_vs_ffs_30_static_frame_0000/turntable_metadata.json`

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
  - the current preset uses `color_by_height` and `orthographic` projection as the most robust readable configuration
- The new turntable workflow now supports:
  - a single selected frame as the primary comparison unit
  - explicit camera-frusta visualization from real `calibrate.pkl` `c2w`
  - object-centric ROI extraction above the tabletop plane
  - optional per-camera RGB-box filtering via `--manual_image_roi_json` when professor-facing renders should suppress the tabletop and fuse only object pixels
  - an object-first selection path when `--manual_image_roi_json` is present:
    - full dense per-camera clouds are loaded first
    - per-camera object masks are derived before context subsampling
    - a seeded object union bbox drives crop / focus / orbit
    - `--max_points_per_camera` becomes the context-layer cap rather than an early object-point cap
  - a coverage-aware orbit informed by the real camera layout
  - synchronized Native vs FFS large side-by-side panels using the exact same orbit path
  - automatic triple outputs:
    - geometry diagnostic video + keyframe sheet
    - RGB reference video + keyframe sheet
    - support-count video + keyframe sheet
  - automatic merge-diagnostic outputs:
    - source-attribution overlay video + keyframe sheet
    - source-split keyframe sheet
    - mismatch residual video + keyframe sheet
  - object-debug outputs:
    - `debug/per_camera_object_mask_overlay/*.png`
    - `debug/per_camera_object_cloud/*.png`
    - `debug/fused_object_only/*`
    - `debug/fused_object_context/*`
    - `debug/compare_debug_metrics.json`
  - automatic pass1 -> pass2 object refinement when `--manual_image_roi_json` is absent:
    - pass1 coarse world ROI
    - projected per-camera coarse bbox generation
    - per-camera foreground-mask refinement
    - pass2 world ROI rebuilt from the masked per-camera object points
  - `graph_union` component closure instead of only a top-component anchored union

## Why This Workflow Is Easier To Read

- Tabletop crop prevents the full room bounds from dominating the frame.
- The new default gives each depth source a much larger panel than the prior 2x3 board.
- The default no longer pretends that unsupported backside views are equally trustworthy.
- The geometry, RGB, and support videos are generated together, so the same orbit path can be judged in all three modes without rerunning the workflow.
- The larger overview makes the real camera locations, supported arc, and current orbit position much easier to interpret.
- `neutral_gray_shaded` plus larger splats and supersampling is now the default geometry view because it emphasizes tabletop flatness more clearly than sparse white points.
- The object-first path makes it possible to diagnose whether missing detail was lost in:
  - the per-camera ROI mask
  - the object/context sampling split
  - or the final fused support pattern
- The new pass1/pass2 ROI artifacts make it possible to distinguish:
  - initial world-ROI under-segmentation
  - pixel-mask recovery of protrusions
  - final fusion/support limitations
- `source` now exposes which camera contributed which visible surface region.
- `mismatch` now exposes where the overlapping 3-view geometry disagrees instead of only how many cameras touched it.
- `support` remains complementary: it shows overlap quantity, while `source` and `mismatch` explain source identity and disagreement.

## Known Limitations

- Two-case fallback comparison is appropriate for static or near-static scenes only. It is not same-take ground truth.
- The current visualization pipeline still uses pinhole `K` matrices without explicit distortion coefficients.
- A dedicated temporal-stability script has not been added yet; static-scene stability must currently be inferred from panel videos and reprojection consistency.
- Teddy-bear-like cases can still require a tighter `--manual_image_roi_json` because rectangular 2D boxes may preserve some tabletop pixels near the feet or box base.
- Automatic projected-bbox refinement reduces but does not eliminate that limitation; manual image ROI remains the strongest override when the object sits directly on a visually similar tabletop.
- Source-attribution overlay is honest about provenance, but it is still a point/surfel view rather than a watertight mesh; overlap artifacts can therefore look thicker than they would on a meshed surface.

## Architecture Note

On 2026-04-09 the visualization implementation was refactored so the user-facing workflows above keep the same product boundary, but the internal code is now split into explicit layers for case IO, artifact writing, crop/view math, layouts, renderers, and workflow wrappers. See:

- `docs/generated/visual_stack_inventory.md`
- `docs/generated/visual_stack_refactor_validation.md`
