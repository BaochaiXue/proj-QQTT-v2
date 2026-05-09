# Batch=3 compile inspection

Timestamp: 2026-05-08T16:08:00-04:00

## Current services

```text
LISTEN 0      4096         0.0.0.0:5201       0.0.0.0:*    users:(("iperf3",pid=805787,fd=7))    
LISTEN 0      100          0.0.0.0:7001       0.0.0.0:*    users:(("python",pid=860860,fd=42))   
LISTEN 0      100          0.0.0.0:7002       0.0.0.0:*    users:(("python",pid=3883045,fd=9))
```

## GPU

```text
Fri May  8 16:08:00 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.211.01             Driver Version: 570.211.01     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        Off |   00000000:41:00.0 Off |                  Off |
|  0%   54C    P2             87W /  480W |     529MiB /  24564MiB |      1%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 4090        Off |   00000000:82:00.0  On |                  Off |
|  0%   48C    P5             56W /  480W |    6242MiB /  24564MiB |     33%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A          541828      G   /usr/lib/xorg/Xorg                        4MiB |
|    0   N/A  N/A         2304550      G   /usr/lib/xorg/Xorg                        4MiB |
|    0   N/A  N/A         4049886      C   /usr/share/rustdesk/rustdesk            495MiB |
|    1   N/A  N/A          541828      G   /usr/lib/xorg/Xorg                      564MiB |
|    1   N/A  N/A          555415      G   ...miniconda3/envs/ps/bin/python          6MiB |
|    1   N/A  N/A          860860      C   python                                 1808MiB |
|    1   N/A  N/A         1828630      G   .../8247/usr/lib/firefox/firefox        166MiB |
|    1   N/A  N/A         2304550      G   /usr/lib/xorg/Xorg                      457MiB |
|    1   N/A  N/A         2304871      G   /usr/bin/gnome-shell                     68MiB |
|    1   N/A  N/A         2334022      G   rustdesk                                 32MiB |
|    1   N/A  N/A         2366889      G   .../8247/usr/lib/firefox/firefox        103MiB |
|    1   N/A  N/A         3824133      G   /usr/share/code/code                    228MiB |
|    1   N/A  N/A         3883045      C   python                                 1808MiB |
|    1   N/A  N/A         4054632      G   /usr/share/rustdesk/rustdesk             23MiB |
+-----------------------------------------------------------------------------------------+
```

## Batch1 engine files

```text
/home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864/demo_out/cloud.ply
/home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864/demo_out/depth_meter.npy
/home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864/demo_out/disp_vis.png
/home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864/demo_out/left.png
/home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864/demo_out/right.png
/home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864/feature_engine_build.log
/home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864/feature_runner.engine
/home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864/feature_runner.onnx
/home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864/onnx.yaml
/home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864/post_engine_build.log
/home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864/post_runner.engine
/home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864/post_runner.onnx
```

## Inspection summary

- Existing `FastFoundationStereoTensorRTRunner.run_batch(...)` accepts a list of samples and already splits batched TensorRT outputs.
- `resolve_tensorrt_engine_static_batch_size(...)` can validate two-stage engine batch dimensions.
- `scripts/harness/verify_ffs_tensorrt_wsl.py` already exports two-stage ONNX with a configurable `--batch_size` and builds engines from ONNX.
- Batch=3 work can be isolated to new scripts and a new engine output directory.
- Required 100-kit replay folder was not found at `/home/xinjie/proj-QQTT-v2/result/demo_v0_3_ir_triplet_100kits_848x480` during initial inspection.
