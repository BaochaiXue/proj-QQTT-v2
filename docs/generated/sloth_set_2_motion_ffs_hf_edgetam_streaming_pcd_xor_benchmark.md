# Sloth Set 2 HF EdgeTAM Streaming PCD XOR

## Output

- GIF: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_streaming_pcd_xor/pcd_xor/gifs/sloth_set_2_motion_ffs_hf_edgetam_streaming_pcd_xor.gif`
- first frame: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_streaming_pcd_xor/pcd_xor/first_frames/sloth_set_2_motion_ffs_hf_edgetam_streaming_pcd_xor_first.png`
- first-frame PLY dir: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_set_2_motion_ffs_hf_edgetam_streaming_pcd_xor/pcd_xor/first_frame_ply`
- case: `sloth_set_2_motion_ffs`
- frames: `93`

## Render Contract

- reference: SAM3.1 masks generated from local case frames
- candidate: HF EdgeTAMVideo streaming masks initialized by SAM3.1 frame-0 mask prompt
- EdgeTAM input contract: one PNG frame at a time; no MP4, full video path, or offline video-folder input
- rows: HF EdgeTAM streaming
- columns: cam0/cam1/cam2 original camera pinhole views
- fused PCD: three masked camera RGB point clouds fused before rendering
- overlap: RGB point color
- red: SAM3.1-only points
- cyan: EdgeTAM-only points

## Aggregate Metrics

| variant | mean 2D IoU | min 2D IoU | mean raw pIoU | mean post pIoU | mean output pts | mean ms/f | mean FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hf_edgetam_streaming_mask | 0.9783 | 0.9711 | 0.9840 | 0.9874 | 82145 | 19.32 | 51.26 |
