# sloth_base_motion_ffs Fused PCD Overlay 2x3

## Output

- GIF: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_base_motion_ffs_fused_pcd_overlay_2x3/gifs/sloth_base_motion_ffs_fused_pcd_overlay_2x3_small_edgetam_compiled.gif`
- first frame: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_base_motion_ffs_fused_pcd_overlay_2x3/first_frames/sloth_base_motion_ffs_fused_pcd_overlay_2x3_small_edgetam_compiled_first.png`
- first-frame PLY dir: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_base_motion_ffs_fused_pcd_overlay_2x3/first_frame_ply`
- case: `sloth_base_motion_ffs`
- frames: `86`

## Render Contract

- rows: SAM2.1 Small, EdgeTAM compiled
- columns: cam0/cam1/cam2 original camera pinhole views
- fused PCD: three masked camera RGB point clouds fused before rendering
- overlap: RGB point color
- red: SAM3.1-only points
- cyan: candidate-only points

## Depth

- source: `depth`
- depth scale override: `0.001`
- override used: `True`

## Aggregate Metrics

| variant | mean 2D IoU | min 2D IoU | mean raw pIoU | mean post pIoU | mean output pts | mean ms/f | mean FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 0.9801 | 0.9752 | 0.9839 | 0.9872 | 74051 | 22.98 | 43.58 |
| edgetam | 0.9768 | 0.9720 | 0.9815 | 0.9853 | 74242 | 14.15 | 70.65 |
