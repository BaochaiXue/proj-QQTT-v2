# EdgeTAM Dynamics Round1 3x6 Panel Benchmark

## Protocol

- case: `ffs_dynamics_round1`
- frames: `71`
- panel columns: `SAM3.1`, `SAM2.1 large/base_plus/small/tiny`, `EdgeTAM`
- EdgeTAM init: `SAM3.1 frame0 union mask` via `add_new_mask(...)`
- EdgeTAM timing: no-output propagation after warmup; model load, JPEG prep, init_state, prompt, warmup, and mask collection are excluded.
- depth override root: `/home/zhangxinjie/proj-QQTT-v2/result/sam21_dynamics_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5_maskinit_stable_throughput/ffs_depth_cache/ffs_dynamics_round1`

## Output

- GIF: `/home/zhangxinjie/proj-QQTT-v2/result/sam21_dynamics_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5_maskinit_stable_throughput/gifs/ffs_dynamics_round1_3x6_time_edgetam.gif`
- first frame: `/home/zhangxinjie/proj-QQTT-v2/result/sam21_dynamics_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5_maskinit_stable_throughput/first_frames/ffs_dynamics_round1_3x6_time_edgetam_first.png`

## EdgeTAM Speed

| cam | ms/frame | FPS | frames |
| ---: | ---: | ---: | ---: |
| 0 | 21.13 | 47.33 | 71 |
| 1 | 20.36 | 49.11 | 71 |
| 2 | 21.58 | 46.35 | 71 |
| **mean** | **21.02** | **47.57** |  |

## Mask Stability

| variant | cam | area std/mean | min IoU(frame,0) |
| --- | ---: | ---: | ---: |
| sam31 | 0 | 0.0659 | 0.0489 |
| sam31 | 1 | 0.1592 | 0.0532 |
| sam31 | 2 | 0.2996 | 0.2703 |
| large | 0 | 0.0688 | 0.0485 |
| large | 1 | 0.1623 | 0.0527 |
| large | 2 | 0.2995 | 0.2727 |
| base_plus | 0 | 0.0690 | 0.0489 |
| base_plus | 1 | 0.1604 | 0.0517 |
| base_plus | 2 | 0.3002 | 0.2726 |
| small | 0 | 0.0696 | 0.0489 |
| small | 1 | 0.1613 | 0.0518 |
| small | 2 | 0.3020 | 0.2714 |
| tiny | 0 | 0.0698 | 0.0492 |
| tiny | 1 | 0.1643 | 0.0520 |
| tiny | 2 | 0.2990 | 0.2730 |
| edgetam | 0 | 0.0807 | 0.0000 |
| edgetam | 1 | 0.1554 | 0.0539 |
| edgetam | 2 | 0.2983 | 0.2737 |
