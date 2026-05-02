# SAM2.1 / EdgeTAM Mask Overlay 3x3

## Output

- GIF: `/home/zhangxinjie/proj-QQTT-v2/result/sam21_dynamics_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5_maskinit_stable_throughput/gifs/ffs_dynamics_round1_mask_overlay_3x3_small_tiny_edgetam_compiled.gif`
- first frame: `/home/zhangxinjie/proj-QQTT-v2/result/sam21_dynamics_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5_maskinit_stable_throughput/first_frames/ffs_dynamics_round1_mask_overlay_3x3_small_tiny_edgetam_compiled_first.png`
- case: `ffs_dynamics_round1`
- frames: `71`

## Overlay Legend

- green: candidate and SAM3.1 overlap
- red: SAM3.1 only
- cyan: candidate only

## IoU Summary

| variant | cam | mean IoU | min IoU |
| --- | ---: | ---: | ---: |
| small | 0 | 0.9784 | 0.9706 |
| small | 1 | 0.9741 | 0.9575 |
| small | 2 | 0.9884 | 0.9822 |
| **small** | **all** | **0.9803** | **0.9575** |
| tiny | 0 | 0.9793 | 0.9719 |
| tiny | 1 | 0.9720 | 0.9505 |
| tiny | 2 | 0.9883 | 0.9828 |
| **tiny** | **all** | **0.9799** | **0.9505** |
| edgetam | 0 | 0.8956 | 0.7424 |
| edgetam | 1 | 0.9716 | 0.9548 |
| edgetam | 2 | 0.9829 | 0.9757 |
| **edgetam** | **all** | **0.9500** | **0.7424** |
