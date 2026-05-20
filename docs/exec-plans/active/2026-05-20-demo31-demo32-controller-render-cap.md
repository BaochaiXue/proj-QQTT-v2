# Demo 3.1 / 3.2 Controller Render Cap

## Goal

Limit the controller body point cloud rendered by Demo 3.1 and Demo 3.2 to a
bounded default so large controller masks do not flood the Open3D renderer.

## Requirements

- Default controller body render points must be capped at 10,000 or fewer.
- The cap is render-only and must not change tracker query masks, tracker query
  counts, control markers, or overlay marker generation.
- Preserve the existing controller render voxel downsample.
- Expose the cap through Demo 3.1 / Demo 3.2 CLI, runtime contract, and profile
  fields.
- Add tests proving the cap is applied after the render voxel stage.

## Validation

- Unit tests for shared controller filtering and Demo 3.1 / Demo 3.2 contract.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
