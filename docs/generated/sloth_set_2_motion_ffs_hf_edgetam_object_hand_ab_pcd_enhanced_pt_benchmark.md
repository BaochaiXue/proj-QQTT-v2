# Sloth Set 2 HF EdgeTAM Object/Hands PCD Panel

## Output

- GIF: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_object_hand_ab_pcd/pcd_gif_enhanced_pt/gifs/sloth_set_2_motion_ffs_hf_edgetam_object_hand_ab_pcd_enhanced_pt.gif`
- First frame: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_object_hand_ab_pcd/pcd_gif_enhanced_pt/first_frames/sloth_set_2_motion_ffs_hf_edgetam_object_hand_ab_pcd_enhanced_pt_first.png`
- First-frame PLY dir: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_object_hand_ab_pcd/pcd_gif_enhanced_pt/first_frame_ply`
- Streaming JSON: `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_object_hand_ab_streaming_results.json`
- Frames: `93`
- Cameras: `[0, 1, 2]`

## Streaming Contract

- `frame_by_frame_streaming=true`
- `offline_video_input_used=false`
- `frame_source=png_loop`
- Frame 0 prompt is SAM3.1 mask prompt for `3` objects.
- This is a qualitative PCD panel, not an XOR quality benchmark.
- PCD postprocess mode: `enhanced-pt`

## SAM3.1 Init Root

- Root: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_object_hand_ab_pcd/sam31_masks`
- EdgeTAM mask root: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_object_hand_ab_pcd/hf_edgetam_streaming_multi_object/masks/sloth_set_2_motion_ffs`
- Object IDs: `1=stuffed animal, 2=hand A, 3=hand B`
- Note: Canonical three-object init root relabeled from the previous two-hand run. Object ids 2 and 3 are stable EdgeTAM hand track ids named hand A and hand B; they are not asserted to be physical left/right hands.

## Compile

- Mode: `vision-reduce-overhead`
- Enabled: `True`
- Applied targets: `vision_encoder`
- Torch compile mode: `reduce-overhead`

## Streaming Summary

- Jobs passed: `3/3`
- First-frame median: `36.27 ms`
- Subsequent median: `27.40 ms`
- Median E2E FPS: `35.99`
- Median model-only FPS: `37.82`

## PCD Postprocess

- Mode: `enhanced-pt`
- Radius: `0.01`
- Neighbors: `40`
- Component voxel size: `0.01`
- Keep-near-main gap: `0.0`

## Objects

| object id | label | mean raw pts | mean output pts | min | max |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | stuffed animal | 83986.1 | 81804.4 | 79361 | 86550 |
| 2 | hand A | 21254.0 | 12023.4 | 5951 | 20934 |
| 3 | hand B | 20239.2 | 14409.3 | 8745 | 21378 |

## Jobs

| cam | frames | first ms | subsequent median ms | p95 ms | e2e FPS | model FPS | failures |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 93 | 37.40 | 28.14 | 30.75 | 35.13 | 37.42 | 0 |
| 1 | 93 | 36.27 | 27.40 | 30.39 | 35.99 | 37.82 | 0 |
| 2 | 93 | 34.23 | 27.11 | 29.22 | 36.39 | 38.26 | 0 |
