# Sloth Set 2 HF EdgeTAM Hand/Object PCD Panel

## Output

- GIF: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd/pcd_gif/gifs/sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd.gif`
- First frame: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd/pcd_gif/first_frames/sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd_first.png`
- First-frame PLY dir: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd/pcd_gif/first_frame_ply`
- Streaming JSON: `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_hand_object_streaming_results.json`
- Frames: `93`
- Cameras: `[0, 1, 2]`

## Streaming Contract

- `frame_by_frame_streaming=true`
- `offline_video_input_used=false`
- `frame_source=png_loop`
- Frame 0 prompt is SAM3.1 mask prompt for two objects.
- This is a qualitative PCD panel, not an XOR quality benchmark.
- PCD postprocess mode: `none`

## SAM3.1 Init Root

- Root: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd/sam31_masks`
- EdgeTAM mask root: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd/hf_edgetam_streaming_multi_object/masks/sloth_set_2_motion_ffs`
- Object IDs: `1=stuffed animal`, `2=hand`
- Note: SAM3.1 multi-prompt session returned both frame-0 objects with label hand; this root merges the single-prompt stuffed animal masks with unioned hand masks without modifying the generator.

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

- Mode: `None`
- Radius: `None`
- Neighbors: `None`
- Component voxel size: `None`
- Keep-near-main gap: `None`

## Objects

| object id | label | mean raw pts | mean output pts | min | max |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | stuffed animal | 0.0 | 83986.1 | 80834 | 89174 |
| 2 | hand | 0.0 | 38537.3 | 30865 | 55119 |

## Jobs

| cam | frames | first ms | subsequent median ms | p95 ms | e2e FPS | model FPS | failures |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 93 | 23.01 | 19.97 | 23.70 | 48.92 | 52.62 | 0 |
| 1 | 93 | 27.44 | 18.84 | 20.94 | 51.80 | 55.21 | 0 |
| 2 | 93 | 27.52 | 18.97 | 21.03 | 51.82 | 55.25 | 0 |
