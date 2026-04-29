# 2026-04-29 Still Object Rope 3x4 Erode Sweep Highlight GIF

## Goal

Adjust the no-delete enhanced PT-like removed-point GIF sweep from a 6x2 board to a 3x4 board:

- rows: three panel rows, each containing two rounds
- columns: Native Depth and FFS for the left round, then Native Depth and FFS for the right round
- cases: Still Object rounds 1-4, Still Rope rounds 1-2
- trajectory: Cam0/Cam1/Cam2 calibrated poses as key nodes, start from Cam0, 360 object-facing frames
- mask erosion values: `1px`, `3px`, `5px`, `10px`
- PT-like deletion: disabled
- enhanced PT-like trace: enabled only to mark would-delete points by source camera color
- FFS setting: `20-30-48`, `valid_iters=4`, pad to `864x480`, builder optimization level `5`

## Scope

- Keep changes inside experiment visualization workflow and harness CLI.
- Preserve the existing 6x2 layout as an explicit option.
- Write the 3x4 sweep into a separate output folder.

## Validation

- Compile the workflow and harness CLI.
- Run the still-object orbit GIF smoke test.
- Run a short real-data 3x4 smoke render.
- Render all four 360-frame 3x4 GIFs and verify frame count plus no-delete/highlight metadata.
- Run deterministic harness checks.

## Result

- Output root: `data/experiments/still_object_rope_frame0_cam0_orbit_3x4_mask_erode_sweep_highlight_gif_ffs203048_iter4_trt_level5`
- GIFs:
  - `mask_erode_01px/still_object_rope_frame0000_cam0_orbit_3x4_enhanced_pt_like_marked_mask_erode_01px.gif`
  - `mask_erode_03px/still_object_rope_frame0000_cam0_orbit_3x4_enhanced_pt_like_marked_mask_erode_03px.gif`
  - `mask_erode_05px/still_object_rope_frame0000_cam0_orbit_3x4_enhanced_pt_like_marked_mask_erode_05px.gif`
  - `mask_erode_10px/still_object_rope_frame0000_cam0_orbit_3x4_enhanced_pt_like_marked_mask_erode_10px.gif`
- Summary JSON: `summary.json` at the output root and inside each erode subdirectory.
- Rendered frames per GIF: `360`
- Canvas: `1440x802`
- Panel layout: `3x4`
- Render mode: `color_by_rgb`
- PT-like deletion: `false`
- Enhanced PT-like removed-point highlight: `true`
- Highlight colors: Cam0 magenta, Cam1 cyan, Cam2 amber.

## Checks

- `python -m py_compile data_process/visualization/experiments/still_object_orbit_gif.py scripts/harness/experiments/visualize_still_object_rope_6x2_orbit_erode_sweep_gif.py`
- `python scripts/harness/experiments/visualize_still_object_rope_6x2_orbit_erode_sweep_gif.py --help`
- `python -m unittest -v tests.test_still_object_orbit_gif_smoke`
- `python scripts/harness/experiments/visualize_still_object_rope_6x2_orbit_erode_sweep_gif.py --erode_pixels 1,3 --num_frames 2 --fps 2 --tile_width 160 --tile_height 100 --max_points_per_variant 5000 --output_root data/experiments/still_object_rope_frame0_cam0_orbit_3x4_mask_erode_sweep_highlight_gif_ffs203048_iter4_trt_level5_smoke`
- `python scripts/harness/experiments/visualize_still_object_rope_6x2_orbit_erode_sweep_gif.py --erode_pixels 1,3,5,10 --num_frames 360 --fps 30 --start_camera_idx 0 --frame_idx 0 --output_root data/experiments/still_object_rope_frame0_cam0_orbit_3x4_mask_erode_sweep_highlight_gif_ffs203048_iter4_trt_level5`
- Verified all four GIFs have `360` frames, `1440x802` canvas, `panel_layout=3x4`, `pt_like_postprocess_enabled=false`, and `enhanced_pt_like_removed_highlight_enabled=true`.
- `python scripts/harness/check_experiment_boundaries.py`
- `python scripts/harness/check_visual_architecture.py`
- `python scripts/harness/check_all.py`
