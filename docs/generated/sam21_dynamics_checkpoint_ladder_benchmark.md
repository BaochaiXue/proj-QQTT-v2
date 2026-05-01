# SAM21 Dynamics Checkpoint Ladder Benchmark

## Environment

- python: `3.12.13`
- torch: `2.11.0+cu130`
- torch_cuda: `13.0`
- torchvision: `0.26.0+cu130`
- sam2_file: `/home/zhangxinjie/external/sam2/sam2/__init__.py`
- cuda_available: `True`
- gpu: `NVIDIA GeForce RTX 5090 Laptop GPU`

## Protocol

- case_set: `dynamics`
- frames: `None`
- SAM2.1 init mode: `mask`
- depth override roots: `True`

## Outputs

- FFS Dynamics R1: `/home/zhangxinjie/proj-QQTT-v2/data/experiments/sam21_dynamics_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5_maskinit_stable_throughput/gifs/ffs_dynamics_round1_3x5_time.gif`
- FFS Dynamics R2: `/home/zhangxinjie/proj-QQTT-v2/data/experiments/sam21_dynamics_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5_maskinit_stable_throughput/gifs/ffs_dynamics_round2_3x5_time.gif`

## SAM2.1 Speed Ladder

| checkpoint | mean ms/frame | mean FPS | samples |
| --- | ---: | ---: | ---: |
| large | 47.16 | 21.21 | 6 |
| base_plus | 26.00 | 38.47 | 6 |
| small | 19.15 | 52.23 | 6 |
| tiny | 18.11 | 55.21 | 6 |

## Stable Worker Aggregate

| checkpoint | prop-only ms/frame | prop-only FPS | sweep wall FPS incl. setup | timed frames |
| --- | ---: | ---: | ---: | ---: |
| large | 47.11 | 21.22 | 14.34 | 531 |
| base_plus | 26.07 | 38.36 | 22.68 | 531 |
| small | 19.14 | 52.25 | 28.66 | 531 |
| tiny | 18.11 | 55.22 | 28.11 | 531 |

## Mask Stability

| case | variant | cam | area std/mean | min IoU(frame,0) |
| --- | --- | ---: | ---: | ---: |
| ffs_dynamics_round1 | sam31 | 0 | 0.0659 | 0.0489 |
| ffs_dynamics_round1 | sam31 | 1 | 0.1592 | 0.0532 |
| ffs_dynamics_round1 | sam31 | 2 | 0.2996 | 0.2703 |
| ffs_dynamics_round1 | large | 0 | 0.0688 | 0.0485 |
| ffs_dynamics_round1 | large | 1 | 0.1623 | 0.0527 |
| ffs_dynamics_round1 | large | 2 | 0.2995 | 0.2727 |
| ffs_dynamics_round1 | base_plus | 0 | 0.0690 | 0.0489 |
| ffs_dynamics_round1 | base_plus | 1 | 0.1604 | 0.0517 |
| ffs_dynamics_round1 | base_plus | 2 | 0.3002 | 0.2726 |
| ffs_dynamics_round1 | small | 0 | 0.0696 | 0.0489 |
| ffs_dynamics_round1 | small | 1 | 0.1613 | 0.0518 |
| ffs_dynamics_round1 | small | 2 | 0.3020 | 0.2714 |
| ffs_dynamics_round1 | tiny | 0 | 0.0698 | 0.0492 |
| ffs_dynamics_round1 | tiny | 1 | 0.1643 | 0.0520 |
| ffs_dynamics_round1 | tiny | 2 | 0.2990 | 0.2730 |
| ffs_dynamics_round2 | sam31 | 0 | 0.3659 | 0.0000 |
| ffs_dynamics_round2 | sam31 | 1 | 0.1864 | 0.0000 |
| ffs_dynamics_round2 | sam31 | 2 | 0.5301 | 0.1506 |
| ffs_dynamics_round2 | large | 0 | 0.3704 | 0.0000 |
| ffs_dynamics_round2 | large | 1 | 0.1932 | 0.0000 |
| ffs_dynamics_round2 | large | 2 | 0.5269 | 0.1516 |
| ffs_dynamics_round2 | base_plus | 0 | 0.3691 | 0.0000 |
| ffs_dynamics_round2 | base_plus | 1 | 0.1884 | 0.0000 |
| ffs_dynamics_round2 | base_plus | 2 | 0.5299 | 0.1513 |
| ffs_dynamics_round2 | small | 0 | 0.3703 | 0.0000 |
| ffs_dynamics_round2 | small | 1 | 0.1866 | 0.0000 |
| ffs_dynamics_round2 | small | 2 | 0.5276 | 0.1510 |
| ffs_dynamics_round2 | tiny | 0 | 0.3691 | 0.0000 |
| ffs_dynamics_round2 | tiny | 1 | 0.1939 | 0.0000 |
| ffs_dynamics_round2 | tiny | 2 | 0.5254 | 0.1529 |

## Timing Contract

The reported SAM2.1 FPS is no-output propagation timing with per-step cudagraph markers after 5 warmup propagations per case/camera job in one long-lived checkpoint worker. Model load, JPEG preparation, init_state, prompt, warmup propagation, and separate mask collection are excluded.
