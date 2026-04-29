# 2026-04-29 Still Object Rope 6x2 Erode Sweep Highlight GIF

## Goal

Create a headless 6x2 GIF erode sweep for frame 0 that uses enhanced PT-like trace only to mark points:

- rows: Still Object rounds 1-4, Still Rope rounds 1-2
- columns: Native Depth point cloud, FFS point cloud
- trajectory: Cam0/Cam1/Cam2 calibrated poses as orbit key nodes, start from Cam0, 360 object-facing frames
- FFS: `20-30-48`, `valid_iters=4`, `848x480` padded to `864x480`, TensorRT builder optimization level `5`
- point rendering: raw RGB point colors
- mask erosion: `1px`, `3px`, `5px`, `10px`, with the active value written in the large title
- point-cloud cleanup: do not enable PT-like deletion
- highlighting: run enhanced PT-like trace to find would-be removed points, then mark them by source camera color without deleting them

## Scope

- Extend the existing 6x2 erode-sweep workflow without changing formal recording/alignment code.
- Reuse existing enhanced PhysTwin-like trace code and source-camera color convention.
- Keep outputs in a separate highlight experiment folder.

## Validation

- Compile the updated workflow and harness script.
- Run the unit smoke test for the 6x2 orbit GIF workflow.
- Run a short erode-sweep smoke render to inspect title, labels, highlight metadata, and no-delete metadata.
- Render the requested four 360-frame GIFs and verify frame count, render mode, erode values, and highlight metadata.

## Result

- Output root: `data/experiments/still_object_rope_frame0_cam0_orbit_6x2_mask_erode_sweep_highlight_gif_ffs203048_iter4_trt_level5`
- GIFs:
  - `mask_erode_01px/still_object_rope_frame0000_cam0_orbit_6x2_enhanced_pt_like_marked_mask_erode_01px.gif`
  - `mask_erode_03px/still_object_rope_frame0000_cam0_orbit_6x2_enhanced_pt_like_marked_mask_erode_03px.gif`
  - `mask_erode_05px/still_object_rope_frame0000_cam0_orbit_6x2_enhanced_pt_like_marked_mask_erode_05px.gif`
  - `mask_erode_10px/still_object_rope_frame0000_cam0_orbit_6x2_enhanced_pt_like_marked_mask_erode_10px.gif`
- First-frame PNGs are written beside each GIF.
- Summary JSON: `summary.json` at the output root and inside each erode subdirectory.
- Rendered frames per GIF: `360`
- Canvas: `900x1438`
- Render mode: `color_by_rgb`
- Point radius: `1`
- PT-like deletion: `false`
- Enhanced PT-like removed-point highlight: `true`
- Highlight colors: Cam0 magenta, Cam1 cyan, Cam2 amber.

## Checks

- `python -m py_compile data_process/visualization/experiments/still_object_orbit_gif.py scripts/harness/experiments/visualize_still_object_rope_6x2_orbit_erode_sweep_gif.py`
- `python scripts/harness/experiments/visualize_still_object_rope_6x2_orbit_erode_sweep_gif.py --help`
- `python -m unittest -v tests.test_still_object_orbit_gif_smoke`
- `python scripts/harness/experiments/visualize_still_object_rope_6x2_orbit_erode_sweep_gif.py --erode_pixels 1,3 --num_frames 2 --fps 2 --tile_width 160 --tile_height 100 --row_label_width 120 --max_points_per_variant 5000 --output_root data/experiments/still_object_rope_frame0_cam0_orbit_6x2_mask_erode_sweep_highlight_gif_ffs203048_iter4_trt_level5_smoke`
- `python scripts/harness/experiments/visualize_still_object_rope_6x2_orbit_erode_sweep_gif.py --erode_pixels 1,3,5,10 --num_frames 360 --fps 30 --start_camera_idx 0 --frame_idx 0 --output_root data/experiments/still_object_rope_frame0_cam0_orbit_6x2_mask_erode_sweep_highlight_gif_ffs203048_iter4_trt_level5`
- Verified all four GIFs have `360` frames and summary metadata reports `pt_like_postprocess_enabled=false`, `enhanced_pt_like_removed_highlight_enabled=true`.
- `python scripts/harness/check_experiment_boundaries.py`
- `python scripts/harness/check_visual_architecture.py`
- `python scripts/harness/check_all.py`
