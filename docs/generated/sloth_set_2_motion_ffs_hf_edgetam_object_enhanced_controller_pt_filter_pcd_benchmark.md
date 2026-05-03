# Sloth Set 2 HF EdgeTAM Object/Hands PCD Panel

## Output

- GIF: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_object_controller_pcd/pcd_gif_object_enhanced_controller_pt_filter/gifs/sloth_set_2_motion_ffs_hf_edgetam_object_enhanced_controller_pt_filter_pcd.gif`
- First frame: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_object_controller_pcd/pcd_gif_object_enhanced_controller_pt_filter/first_frames/sloth_set_2_motion_ffs_hf_edgetam_object_enhanced_controller_pt_filter_pcd_first.png`
- First-frame PLY dir: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_object_controller_pcd/pcd_gif_object_enhanced_controller_pt_filter/first_frame_ply`
- Streaming JSON: `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_object_controller_streaming_results.json`
- Frames: `93`
- Cameras: `[0, 1, 2]`

## Streaming Contract

- `frame_by_frame_streaming=true`
- `offline_video_input_used=false`
- `frame_source=png_loop`
- Frame 0 prompt is SAM3.1 mask prompt for `2` objects.
- This is a qualitative PCD panel, not an XOR quality benchmark.
- PCD postprocess mode: `enhanced-pt`

## SAM3.1 Init Root

- Root: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_object_controller_pcd/sam31_masks`
- EdgeTAM mask root: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_object_controller_pcd/hf_edgetam_streaming_multi_object/masks/sloth_set_2_motion_ffs`
- Object IDs: `1=stuffed animal, 2=controller`
- Note: PhysTwin-style controller root relabeled from the earlier object+union-hand run. Object id 2 is a single controller mask/track that contains both hands; it is not a per-hand instance id.

## Compile

- Mode: `vision-reduce-overhead`
- Enabled: `True`
- Applied targets: `vision_encoder`
- Torch compile mode: `reduce-overhead`

## Streaming Summary

- Jobs passed: `3/3`
- First-frame median: `27.44 ms`
- Subsequent median: `18.97 ms`
- Median E2E FPS: `51.80`
- Median model-only FPS: `55.21`

## PCD Postprocess

- Default mode: `enhanced-pt`
- Controller override: `pt-filter`
- Radius: `0.01`
- Neighbors: `40`
- Component voxel size: `0.01`
- Keep-near-main gap: `0.0`

## Objects

| object id | label | postprocess | mean raw pts | mean output pts | min | max |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | stuffed animal | enhanced-pt | 83986.1 | 81804.4 | 79361 | 86550 |
| 2 | controller | pt-filter | 38537.3 | 37574.1 | 29392 | 53539 |

## Controller Warning

- `controller` follows the PhysTwin-style convention: all hand instances are merged into one controller mask/PCD.
- Prefer the simpler `pt-filter` radius cleanup for controller rows. If `enhanced-pt` is enabled on controller rows, it can remove sparse fingertips, contact patches, or partial hand points that may matter for manipulation.
- Do not interpret controller output as per-hand identity. Per-hand workflows need an explicit 3D cross-view identity mapping.

## Jobs

| cam | frames | first ms | subsequent median ms | p95 ms | e2e FPS | model FPS | failures |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 93 | 23.01 | 19.97 | 23.70 | 48.92 | 52.62 | 0 |
| 1 | 93 | 27.44 | 18.84 | 20.94 | 51.80 | 55.21 | 0 |
| 2 | 93 | 27.52 | 18.97 | 21.03 | 51.82 | 55.25 | 0 |
