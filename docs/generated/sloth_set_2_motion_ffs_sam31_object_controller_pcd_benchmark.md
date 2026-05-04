# Sloth Set 2 SAM3.1 Object/Hands PCD Panel

## Output

- GIF: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_sam31_object_controller_pcd/pcd_gif_object_enhanced_controller_pt_filter/gifs/sloth_set_2_motion_ffs_sam31_object_enhanced_controller_pt_filter_pcd.gif`
- First frame: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_sam31_object_controller_pcd/pcd_gif_object_enhanced_controller_pt_filter/first_frames/sloth_set_2_motion_ffs_sam31_object_enhanced_controller_pt_filter_pcd_first.png`
- First-frame PLY dir: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_sam31_object_controller_pcd/pcd_gif_object_enhanced_controller_pt_filter/first_frame_ply`
- Streaming JSON: `not written; render-only mask panel`
- Frames: `93`
- Cameras: `[0, 1, 2]`

## Mask Source Contract

- `frame_by_frame_streaming=false` for this render-only SAM3.1 mask panel.
- No HF EdgeTAM tracking or propagation was run for this panel.
- `frame_source=png_loop`
- All frames are masked directly by `SAM3.1` masks.
- This is a qualitative PCD panel, not an XOR quality benchmark.
- PCD postprocess mode: `enhanced-pt`

## SAM3.1 Init Root

- Root: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_object_controller_pcd/sam31_masks`
- Render mask root: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_object_controller_pcd/sam31_masks`
- Object IDs: `1=stuffed animal, 2=controller`
- Note: PhysTwin-style controller root relabeled from the earlier object+union-hand run. Object id 2 is a single controller mask/track that contains both hands; it is not a per-hand instance id.

## PCD Postprocess

- Default mode: `enhanced-pt`
- Controller/hand override: `pt-filter`
- Controller/hand effective mode: `pt-filter`
- Radius: `0.01`
- Neighbors: `40`
- Component voxel size: `0.01`
- Keep-near-main gap: `0.0`

## Objects

| object id | label | postprocess | mean raw pts | mean output pts | min | max |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | stuffed animal | enhanced-pt | 83376.9 | 81308.2 | 78776 | 85997 |
| 2 | controller | pt-filter | 33352.3 | 32587.8 | 21093 | 43318 |

## Controller/Hand Warning

- `controller` follows the PhysTwin-style convention: all hand instances are merged into one controller mask/PCD.
- Object rows use `enhanced-pt` for cleaner presentation when that global mode is selected; controller/hand rows use the simpler `pt-filter` by default.
- If `enhanced-pt` is enabled on controller/hand rows with an explicit override, it can remove sparse fingertips, contact patches, or partial hand points that may matter for manipulation.
- Do not interpret controller output as per-hand identity. Per-hand workflows need an explicit 3D cross-view identity mapping.
