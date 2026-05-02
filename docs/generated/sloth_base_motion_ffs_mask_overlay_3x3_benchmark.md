# SAM2.1 / EdgeTAM Mask Overlay 3x3

## Output

- GIF: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_base_motion_ffs_mask_overlay_3x3/gifs/sloth_base_motion_ffs_mask_overlay_3x3_small_tiny_edgetam_compiled.gif`
- first frame: `/home/zhangxinjie/proj-QQTT-v2/result/sloth_base_motion_ffs_mask_overlay_3x3/first_frames/sloth_base_motion_ffs_mask_overlay_3x3_small_tiny_edgetam_compiled_first.png`
- case: `sloth_base_motion_ffs`
- frames: `86`

## Overlay Legend

- background: black outside SAM3.1 union candidate, original RGB inside the union
- overlap: overlap keeps original RGB
- red: SAM3.1 only
- cyan: candidate only

## IoU Summary

| variant | cam | mean IoU | min IoU |
| --- | ---: | ---: | ---: |
| small | 0 | 0.9809 | 0.9762 |
| small | 1 | 0.9821 | 0.9751 |
| small | 2 | 0.9731 | 0.9588 |
| **small** | **all** | **0.9787** | **0.9588** |
| tiny | 0 | 0.9778 | 0.9735 |
| tiny | 1 | 0.9791 | 0.9710 |
| tiny | 2 | 0.9702 | 0.9546 |
| **tiny** | **all** | **0.9757** | **0.9546** |
| edgetam | 0 | 0.9786 | 0.9736 |
| edgetam | 1 | 0.9789 | 0.9715 |
| edgetam | 2 | 0.9674 | 0.9472 |
| **edgetam** | **all** | **0.9750** | **0.9472** |

## Worker Timing

| model | cam | ms/frame | FPS | frames |
| --- | ---: | ---: | ---: | ---: |
| small | 0 | 21.94 | 45.57 | 86 |
| small | 1 | 22.93 | 43.61 | 86 |
| small | 2 | 24.06 | 41.57 | 86 |
| tiny | 0 | 24.29 | 41.17 | 86 |
| tiny | 1 | 23.95 | 41.75 | 86 |
| tiny | 2 | 25.11 | 39.83 | 86 |
| edgetam | 0 | 14.09 | 70.96 | 86 |
| edgetam | 1 | 14.03 | 71.30 | 86 |
| edgetam | 2 | 14.35 | 69.70 | 86 |
