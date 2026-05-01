# SAM21-max Round2 Benchmark

## Environment

- python: `3.12.13`
- torch: `2.11.0+cu130`
- torch_cuda: `13.0`
- torchvision: `0.26.0+cu130`
- sam2_file: `/home/zhangxinjie/external/sam2/sam2/__init__.py`
- cuda_available: `True`
- gpu: `NVIDIA GeForce RTX 5090 Laptop GPU`

## Outputs

- Still Object R1: `/home/zhangxinjie/proj-QQTT-v2/result/sam21_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5_stable_throughput/gifs/still_object_round1_3x5_time.gif`
- Still Object R2: `/home/zhangxinjie/proj-QQTT-v2/result/sam21_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5_stable_throughput/gifs/still_object_round2_3x5_time.gif`
- Still Object R3: `/home/zhangxinjie/proj-QQTT-v2/result/sam21_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5_stable_throughput/gifs/still_object_round3_3x5_time.gif`
- Still Object R4: `/home/zhangxinjie/proj-QQTT-v2/result/sam21_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5_stable_throughput/gifs/still_object_round4_3x5_time.gif`
- Still Rope R1: `/home/zhangxinjie/proj-QQTT-v2/result/sam21_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5_stable_throughput/gifs/still_rope_round1_3x5_time.gif`
- Still Rope R2: `/home/zhangxinjie/proj-QQTT-v2/result/sam21_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5_stable_throughput/gifs/still_rope_round2_3x5_time.gif`

## SAM2.1 Speed Ladder

| checkpoint | mean ms/frame | mean FPS | samples |
| --- | ---: | ---: | ---: |
| large | 39.83 | 25.11 | 18 |
| base_plus | 21.82 | 45.82 | 18 |
| small | 15.66 | 63.85 | 18 |
| tiny | 15.29 | 65.39 | 18 |

## Stable Worker Aggregate

| checkpoint | prop-only ms/frame | prop-only FPS | sweep wall FPS incl. setup | timed frames |
| --- | ---: | ---: | ---: | ---: |
| large | 39.83 | 25.11 | 17.47 | 540 |
| base_plus | 21.82 | 45.82 | 26.22 | 540 |
| small | 15.66 | 63.85 | 31.68 | 540 |
| tiny | 15.29 | 65.39 | 30.91 | 540 |

## Mask Stability

| case | variant | cam | area std/mean | min IoU(frame,0) |
| --- | --- | ---: | ---: | ---: |
| still_object_round1 | sam31 | 0 | 0.0007 | 0.9914 |
| still_object_round1 | sam31 | 1 | 0.0022 | 0.9850 |
| still_object_round1 | sam31 | 2 | 0.0047 | 0.9815 |
| still_object_round1 | large | 0 | 0.0038 | 0.9895 |
| still_object_round1 | large | 1 | 0.0041 | 0.9888 |
| still_object_round1 | large | 2 | 0.0045 | 0.9761 |
| still_object_round1 | base_plus | 0 | 0.0007 | 0.9945 |
| still_object_round1 | base_plus | 1 | 0.0011 | 0.9913 |
| still_object_round1 | base_plus | 2 | 0.0017 | 0.9877 |
| still_object_round1 | small | 0 | 0.0015 | 0.9921 |
| still_object_round1 | small | 1 | 0.0025 | 0.9877 |
| still_object_round1 | small | 2 | 0.0019 | 0.9844 |
| still_object_round1 | tiny | 0 | 0.0006 | 0.9929 |
| still_object_round1 | tiny | 1 | 0.0014 | 0.9848 |
| still_object_round1 | tiny | 2 | 0.0012 | 0.9852 |
| still_object_round2 | sam31 | 0 | 0.0019 | 0.9800 |
| still_object_round2 | sam31 | 1 | 0.0017 | 0.9831 |
| still_object_round2 | sam31 | 2 | 0.0014 | 0.9854 |
| still_object_round2 | large | 0 | 0.0020 | 0.9865 |
| still_object_round2 | large | 1 | 0.0013 | 0.9890 |
| still_object_round2 | large | 2 | 0.0018 | 0.9893 |
| still_object_round2 | base_plus | 0 | 0.0023 | 0.9827 |
| still_object_round2 | base_plus | 1 | 0.0013 | 0.9872 |
| still_object_round2 | base_plus | 2 | 0.0011 | 0.9879 |
| still_object_round2 | small | 0 | 0.0016 | 0.9834 |
| still_object_round2 | small | 1 | 0.0019 | 0.9854 |
| still_object_round2 | small | 2 | 0.0021 | 0.9817 |
| still_object_round2 | tiny | 0 | 0.0022 | 0.9794 |
| still_object_round2 | tiny | 1 | 0.0032 | 0.9774 |
| still_object_round2 | tiny | 2 | 0.0026 | 0.9788 |
| still_object_round3 | sam31 | 0 | 0.0005 | 0.9926 |
| still_object_round3 | sam31 | 1 | 0.0030 | 0.9853 |
| still_object_round3 | sam31 | 2 | 0.0034 | 0.9829 |
| still_object_round3 | large | 0 | 0.0008 | 0.9934 |
| still_object_round3 | large | 1 | 0.0084 | 0.9870 |
| still_object_round3 | large | 2 | 0.0082 | 0.9844 |
| still_object_round3 | base_plus | 0 | 0.0007 | 0.9907 |
| still_object_round3 | base_plus | 1 | 0.0010 | 0.9896 |
| still_object_round3 | base_plus | 2 | 0.0018 | 0.9873 |
| still_object_round3 | small | 0 | 0.0020 | 0.9877 |
| still_object_round3 | small | 1 | 0.0014 | 0.9876 |
| still_object_round3 | small | 2 | 0.0035 | 0.9783 |
| still_object_round3 | tiny | 0 | 0.0004 | 0.9917 |
| still_object_round3 | tiny | 1 | 0.0015 | 0.9819 |
| still_object_round3 | tiny | 2 | 0.0066 | 0.9677 |
| still_object_round4 | sam31 | 0 | 0.0029 | 0.9868 |
| still_object_round4 | sam31 | 1 | 0.0027 | 0.9852 |
| still_object_round4 | sam31 | 2 | 0.0025 | 0.9879 |
| still_object_round4 | large | 0 | 0.0013 | 0.9888 |
| still_object_round4 | large | 1 | 0.0055 | 0.9870 |
| still_object_round4 | large | 2 | 0.0028 | 0.9873 |
| still_object_round4 | base_plus | 0 | 0.0014 | 0.9878 |
| still_object_round4 | base_plus | 1 | 0.0010 | 0.9914 |
| still_object_round4 | base_plus | 2 | 0.0024 | 0.9828 |
| still_object_round4 | small | 0 | 0.0047 | 0.9836 |
| still_object_round4 | small | 1 | 0.0015 | 0.9835 |
| still_object_round4 | small | 2 | 0.0012 | 0.9793 |
| still_object_round4 | tiny | 0 | 0.0009 | 0.9878 |
| still_object_round4 | tiny | 1 | 0.0020 | 0.9831 |
| still_object_round4 | tiny | 2 | 0.0007 | 0.9882 |
| still_rope_round1 | sam31 | 0 | 0.0038 | 0.9747 |
| still_rope_round1 | sam31 | 1 | 0.0046 | 0.9681 |
| still_rope_round1 | sam31 | 2 | 0.0106 | 0.9597 |
| still_rope_round1 | large | 0 | 0.0012 | 0.9857 |
| still_rope_round1 | large | 1 | 0.0152 | 0.9681 |
| still_rope_round1 | large | 2 | 0.0056 | 0.9678 |
| still_rope_round1 | base_plus | 0 | 0.0012 | 0.9886 |
| still_rope_round1 | base_plus | 1 | 0.0025 | 0.9808 |
| still_rope_round1 | base_plus | 2 | 0.0040 | 0.9745 |
| still_rope_round1 | small | 0 | 0.0031 | 0.9800 |
| still_rope_round1 | small | 1 | 0.0089 | 0.9709 |
| still_rope_round1 | small | 2 | 0.0084 | 0.9746 |
| still_rope_round1 | tiny | 0 | 0.0012 | 0.9768 |
| still_rope_round1 | tiny | 1 | 0.0086 | 0.9701 |
| still_rope_round1 | tiny | 2 | 0.0078 | 0.9762 |
| still_rope_round2 | sam31 | 0 | 0.0029 | 0.9820 |
| still_rope_round2 | sam31 | 1 | 0.0094 | 0.9734 |
| still_rope_round2 | sam31 | 2 | 0.0017 | 0.9864 |
| still_rope_round2 | large | 0 | 0.0018 | 0.9825 |
| still_rope_round2 | large | 1 | 0.0050 | 0.9744 |
| still_rope_round2 | large | 2 | 0.0061 | 0.9821 |
| still_rope_round2 | base_plus | 0 | 0.0022 | 0.9835 |
| still_rope_round2 | base_plus | 1 | 0.0014 | 0.9822 |
| still_rope_round2 | base_plus | 2 | 0.0017 | 0.9871 |
| still_rope_round2 | small | 0 | 0.0020 | 0.9778 |
| still_rope_round2 | small | 1 | 0.0046 | 0.9777 |
| still_rope_round2 | small | 2 | 0.0048 | 0.9850 |
| still_rope_round2 | tiny | 0 | 0.0027 | 0.9771 |
| still_rope_round2 | tiny | 1 | 0.0091 | 0.9702 |
| still_rope_round2 | tiny | 2 | 0.0030 | 0.9861 |

## Timing Contract

The reported SAM2.1 FPS is no-output propagation timing with per-step cudagraph markers after 5 warmup propagations per case/camera job in one long-lived checkpoint worker. Model load, JPEG preparation, init_state, prompt, warmup propagation, and separate mask collection are excluded.
