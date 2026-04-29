# 2026-04-29 Still Object/Rope 6x2 Orbit Mask-Erode Sweep GIF

## Goal

Create a workflow that renders four labeled 6x2 orbit GIF panels:

- rows: Still Object rounds 1-4, Still Rope rounds 1-2
- columns: Native Depth point cloud and FFS point cloud
- frame: 0
- trajectory: Cam0/Cam1/Cam2 calibrated poses as key nodes, start from Cam0, 360 object-facing frames
- mask erosion: `1px`, `3px`, `5px`, `10px`
- FFS: `20-30-48`, `valid_iters=4`, `848x480` padded to `864x480`, TensorRT builder optimization level `5`
- no PT-like postprocess

## Scope

- Extend the experiment-only 6x2 orbit GIF workflow.
- Apply the selected eroded mask to both Native and FFS point-cloud columns for each GIF.
- Add a thin harness CLI that generates one output subfolder per erode value.
- Keep formal recording/alignment code unchanged.

## Validation

- Compile the workflow and harness scripts.
- Check the new harness `--help`.
- Run a low-resolution, low-frame-count smoke sweep for `1px` and `3px`.
- Run deterministic scope/visual architecture guards.

## Result

- Workflow function: `run_still_object_rope_6x2_orbit_gif_erode_sweep_workflow`
- Harness CLI: `scripts/harness/experiments/visualize_still_object_rope_6x2_orbit_erode_sweep_gif.py`
- Default output root: `data/experiments/still_object_rope_frame0_cam0_orbit_6x2_mask_erode_sweep_gif_ffs203048_iter4_trt_level5`
- Default erode sweep: `1,3,5,10`
- Output contract: one subfolder per erode value, each with a 6x2 GIF, first-frame PNG, and `summary.json`
- Smoke output root: `data/experiments/still_object_rope_frame0_cam0_orbit_6x2_mask_erode_sweep_gif_ffs203048_iter4_trt_level5_smoke`
- Full GIFs:
  - `mask_erode_01px/still_object_rope_frame0000_cam0_orbit_6x2_mask_erode_01px.gif`
  - `mask_erode_03px/still_object_rope_frame0000_cam0_orbit_6x2_mask_erode_03px.gif`
  - `mask_erode_05px/still_object_rope_frame0000_cam0_orbit_6x2_mask_erode_05px.gif`
  - `mask_erode_10px/still_object_rope_frame0000_cam0_orbit_6x2_mask_erode_10px.gif`
- Verified each full GIF has `360` frames and canvas size `900x1438`.

## Checks

- `python -m py_compile data_process/visualization/experiments/still_object_orbit_gif.py scripts/harness/experiments/visualize_still_object_rope_6x2_orbit_gif.py scripts/harness/experiments/visualize_still_object_rope_6x2_orbit_erode_sweep_gif.py`
- `python scripts/harness/experiments/visualize_still_object_rope_6x2_orbit_erode_sweep_gif.py --help`
- `python -m unittest -v tests.test_still_object_orbit_gif_smoke`
- `python scripts/harness/experiments/visualize_still_object_rope_6x2_orbit_erode_sweep_gif.py --erode_pixels 1,3 --num_frames 2 --fps 2 --tile_width 160 --tile_height 100 --row_label_width 120 --max_points_per_variant 5000 --output_root data/experiments/still_object_rope_frame0_cam0_orbit_6x2_mask_erode_sweep_gif_ffs203048_iter4_trt_level5_smoke`
- `python scripts/harness/check_experiment_boundaries.py`
- `python scripts/harness/check_visual_architecture.py`
- `python scripts/harness/check_all.py`
- `python scripts/harness/experiments/visualize_still_object_rope_6x2_orbit_erode_sweep_gif.py --erode_pixels 1,3,5,10 --num_frames 360 --fps 30 --start_camera_idx 0 --frame_idx 0 --output_root data/experiments/still_object_rope_frame0_cam0_orbit_6x2_mask_erode_sweep_gif_ffs203048_iter4_trt_level5`
- Full output verification: summary has `erode_pixels=[1,3,5,10]`, `variant_count=4`, `num_frames=360`, and `pt_like_postprocess_enabled=False`.
