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

Pending.

## Checks

Pending.
