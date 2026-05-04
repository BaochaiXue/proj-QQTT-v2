# Sloth Set 2 SAM3.1 Object/Controller PCD GIF

## Goal

Render a Sloth Set 2 fused-PCD GIF using SAM3.1 masks directly for
`stuffed animal` and `controller`, rather than HF EdgeTAM tracking masks.

## Scope

- Reuse the existing Sloth Set 2 hand/object PCD renderer.
- Add a render-only mask-root/source-label override so the output title and
  report say SAM3.1 when SAM3.1 masks are used.
- Use existing masks under
  `result/sloth_set_2_motion_ffs_hf_edgetam_object_controller_pcd/sam31_masks`.
- Do not rerun SAM3.1 or HF EdgeTAM tracking unless the existing masks are
  missing.

## Validation

- Render all available Sloth Set 2 frames.
- Verify the GIF and first-frame PNG exist and have expected dimensions.
- Run focused help/import checks for the renderer.

## Outcome

- Added render-only mask source overrides to the Sloth Set 2 hand/object PCD
  renderer.
- Rendered a 93-frame SAM3.1 object/controller fused PCD GIF.
- Used existing SAM3.1 masks from
  `result/sloth_set_2_motion_ffs_hf_edgetam_object_controller_pcd/sam31_masks`.
- Output GIF:
  `result/sloth_set_2_motion_ffs_sam31_object_controller_pcd/pcd_gif_object_enhanced_controller_pt_filter/gifs/sloth_set_2_motion_ffs_sam31_object_enhanced_controller_pt_filter_pcd.gif`
- First frame:
  `result/sloth_set_2_motion_ffs_sam31_object_controller_pcd/pcd_gif_object_enhanced_controller_pt_filter/first_frames/sloth_set_2_motion_ffs_sam31_object_enhanced_controller_pt_filter_pcd_first.png`
- Generated docs:
  `docs/generated/sloth_set_2_motion_ffs_sam31_object_controller_pcd_benchmark.md`
  and
  `docs/generated/sloth_set_2_motion_ffs_sam31_object_controller_pcd_results.json`.
