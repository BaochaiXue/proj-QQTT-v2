# Sloth Set 2 HF EdgeTAM Object/Hands PCD Panel

## Output

- GIF: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_object_two_hands_pcd/pcd_gif/gifs/sloth_set_2_motion_ffs_hf_edgetam_object_two_hands_pcd.gif`
- First frame: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_object_two_hands_pcd/pcd_gif/first_frames/sloth_set_2_motion_ffs_hf_edgetam_object_two_hands_pcd_first.png`
- First-frame PLY dir: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_object_two_hands_pcd/pcd_gif/first_frame_ply`
- Streaming JSON: `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_object_two_hands_streaming_results.json`
- Frames: `93`
- Cameras: `[0, 1, 2]`

## Streaming Contract

- `frame_by_frame_streaming=true`
- `offline_video_input_used=false`
- `frame_source=png_loop`
- Frame 0 prompt is SAM3.1 mask prompt for `3` objects.
- This is a qualitative PCD panel, not an XOR quality benchmark.
- PCD postprocess mode: `none`

## SAM3.1 Init Root

- Root: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_object_two_hands_pcd/sam31_masks`
- EdgeTAM mask root: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_object_two_hands_pcd/hf_edgetam_streaming_multi_object/masks/sloth_set_2_motion_ffs`
- Object IDs: `1=stuffed animal, 2=left hand, 3=right hand`
- Note: Canonical three-object init root built from the existing stuffed animal masks and the two raw SAM3.1 hand instances; hand instances are split by frame-0 image x-centroid.

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

- Mode: `none`
- Radius: `0.01`
- Neighbors: `40`
- Component voxel size: `0.01`
- Keep-near-main gap: `0.0`

## Objects

| object id | label | mean raw pts | mean output pts | min | max |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | stuffed animal | 83986.1 | 83986.1 | 80834 | 89174 |
| 2 | left hand | 21254.0 | 21254.0 | 10499 | 33414 |
| 3 | right hand | 20239.2 | 20239.2 | 15792 | 27082 |

## Jobs

| cam | frames | first ms | subsequent median ms | p95 ms | e2e FPS | model FPS | failures |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 93 | 37.40 | 28.14 | 30.75 | 35.13 | 37.42 | 0 |
| 1 | 93 | 36.27 | 27.40 | 30.39 | 35.99 | 37.82 | 0 |
| 2 | 93 | 34.23 | 27.11 | 29.22 | 36.39 | 38.26 | 0 |
