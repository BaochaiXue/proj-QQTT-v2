# 2026-04-29 Still Object Orbit GIF

## Goal

Prototype a headless 1x2 GIF panel for `still_object round1 frame0`:

- left panel: masked Native Depth point cloud
- right panel: masked FFS point cloud
- FFS setting: `20-30-48`, `valid_iters=4`, `848x480` padded to `864x480`, TensorRT builder optimization level `5`
- trajectory: use Cam0/Cam1/Cam2 calibrated poses as orbit key nodes, start from Cam0, complete one 360-degree object-facing orbit

## Scope

- Reuse the aligned still-object case and existing `sam31_masks`.
- Keep implementation in experiment-only visualization/harness code.
- Avoid Open3D windows; render with the existing headless fallback point-cloud renderer.
- Produce a GIF, a first-frame PNG, and a JSON summary under `result/`.

## Validation

- Check the CLI help.
- Run a short low-resolution smoke GIF.
- Run the requested 360-frame Cam0 GIF for still-object round1 frame0.

## Result

- Workflow module: `data_process/visualization/experiments/still_object_orbit_gif.py`
- Harness CLI: `scripts/harness/experiments/visualize_still_object_orbit_gif.py`
- Output root: `result/still_object_round1_frame0_cam0_orbit_gif_ffs203048_iter4_trt_level5`
- GIF: `still_object_round1_frame0000_cam0_orbit_1x2.gif`
- First-frame PNG: `still_object_round1_frame0000_cam0_orbit_first.png`
- Summary JSON: `summary.json`
- Rendered frames: `360`
- Canvas: `960x478`
- Native render points: `32803`
- FFS render points: `88583`
- Key node relative angles: Cam0 `0.0`, Cam2 `149.0`, Cam1 `236.3`
- Render mode updated to raw RGB point colors: `color_by_rgb`, `point_radius_px=1`.

## Checks

- `python -m py_compile data_process/visualization/experiments/still_object_orbit_gif.py scripts/harness/experiments/visualize_still_object_orbit_gif.py`
- `python scripts/harness/experiments/visualize_still_object_orbit_gif.py --help`
- `python scripts/harness/experiments/visualize_still_object_orbit_gif.py --num_frames 12 --fps 12 --tile_width 320 --tile_height 240 --point_radius_px 1 --output_dir result/still_object_round1_frame0_cam0_orbit_gif_ffs203048_iter4_trt_level5_smoke_crop`
- `python scripts/harness/experiments/visualize_still_object_orbit_gif.py --num_frames 360 --fps 30 --start_camera_idx 0 --frame_idx 0 --output_dir result/still_object_round1_frame0_cam0_orbit_gif_ffs203048_iter4_trt_level5`
- `python scripts/harness/check_experiment_boundaries.py`
- `python scripts/harness/check_visual_architecture.py`
