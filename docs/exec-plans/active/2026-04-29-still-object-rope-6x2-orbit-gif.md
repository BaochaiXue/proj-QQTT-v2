# 2026-04-29 Still Object Rope 6x2 Orbit GIF

## Goal

Create a headless 6x2 GIF panel for frame 0:

- rows: Still Object rounds 1-4, Still Rope rounds 1-2
- columns: Native Depth point cloud, FFS point cloud
- trajectory: Cam0/Cam1/Cam2 calibrated poses as orbit key nodes, start from Cam0, 360 object-facing frames
- FFS: `20-30-48`, `valid_iters=4`, `848x480` padded to `864x480`, TensorRT builder optimization level `5`
- point rendering: raw RGB point colors
- point-cloud cleanup: no PT-like postprocess

## Scope

- Reuse aligned cases and existing formal masks under each case's `sam31_masks`.
- Keep reusable rendering code in `data_process/visualization/experiments/`.
- Keep the operator command as a thin harness script under `scripts/harness/experiments/`.

## Validation

- Compile the updated workflow and new harness script.
- Run a short frame-count smoke render to inspect the 6x2 layout.
- Render the requested 360-frame GIF and verify frame count, render mode, and output paths.

## Result

- Workflow module: `data_process/visualization/experiments/still_object_orbit_gif.py`
- Harness CLI: `scripts/harness/experiments/visualize_still_object_rope_6x2_orbit_gif.py`
- Output root: `data/experiments/still_object_rope_frame0_cam0_orbit_6x2_gif_ffs203048_iter4_trt_level5`
- GIF: `still_object_rope_frame0000_cam0_orbit_6x2.gif`
- First-frame PNG: `still_object_rope_frame0000_cam0_orbit_6x2_first.png`
- Summary JSON: `summary.json`
- Rendered frames: `360`
- Canvas: `900x1438`
- Render mode: `color_by_rgb`
- Point radius: `1`
- PT-like postprocess: `false`

## Checks

- `python -m py_compile data_process/visualization/experiments/still_object_orbit_gif.py scripts/harness/experiments/visualize_still_object_orbit_gif.py scripts/harness/experiments/visualize_still_object_rope_6x2_orbit_gif.py`
- `python scripts/harness/experiments/visualize_still_object_rope_6x2_orbit_gif.py --help`
- `python scripts/harness/experiments/visualize_still_object_rope_6x2_orbit_gif.py --num_frames 6 --fps 6 --tile_width 300 --tile_height 180 --row_label_width 160 --max_points_per_variant 80000 --output_dir data/experiments/still_object_rope_frame0_cam0_orbit_6x2_gif_ffs203048_iter4_trt_level5_smoke`
- `python scripts/harness/experiments/visualize_still_object_rope_6x2_orbit_gif.py --num_frames 360 --fps 30 --start_camera_idx 0 --frame_idx 0 --output_dir data/experiments/still_object_rope_frame0_cam0_orbit_6x2_gif_ffs203048_iter4_trt_level5`
- `python scripts/harness/check_experiment_boundaries.py`
- `python scripts/harness/check_visual_architecture.py`
