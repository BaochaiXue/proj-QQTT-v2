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
