# Manual Validation: D455 + Fast-FoundationStereo Proof-of-Life

These checks are manual and are not run in CI.

## Hardware Inventory

- `239222303506`
- `239222300433`
- `239222300781`

All three active cameras are Intel RealSense D455 devices from the D400 family.

## Validation Steps

1. Verify the external FFS checkpoint exists:
   - `C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth`
2. Run the official FFS demo validation helper.
3. Run the single-camera probe against serial `239222303506`.
4. Confirm the raw sample directory contains:
   - `ir_left.png`
   - `ir_right.png`
   - `metadata.json`
   - optional `color.png`
5. Run FFS on the saved pair.
6. Confirm FFS outputs include:
   - `disparity_raw.npy`
   - `disp_vis.png`
   - `depth_ir_left_float_m.npy`
7. Reproject the FFS result to color coordinates.
8. Confirm conversion outputs include:
   - `depth_ir_left_float_m.npy`
   - `depth_color_aligned_float_m.npy` when color exists
   - `depth_color_aligned_u16.npy` when scale metadata exists
9. Inspect `quicklook_depth.png` for reasonable geometry and invalid regions.

## Geometry Checks

- Confirm the sample uses one D455 internal IR pair only.
- Confirm baseline is read from runtime extrinsics, not a hardcoded 95 mm.
- Confirm invalid depth is encoded as `0` in the replacement-compatible output.
- Confirm color-aligned depth is only produced when color metadata exists.

