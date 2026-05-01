# EdgeTAM vs SAM2.1 Compile Fairness Ablation

## Protocol

- case: `/home/zhangxinjie/proj-QQTT-v2/data/dynamics/ffs_dynamics_round1_20260414`
- cameras: `[0, 1, 2]`
- frames per camera: `71`
- runs: `25` total, first `5` warmup
- timed loop: `for ... in predictor.propagate_in_video(state): pass`
- excluded from FPS: model build, JPEG prep, init_state, prompt, threshold, CPU copy, mask save, PCD, render
- cudagraph step marker: `used`

## Local EdgeTAM Path

- repo: `/home/zhangxinjie/EdgeTAM`
- git_commit: `7711e012a30a2402c4eaab637bdb00a521302c91`
- checkpoint: `/home/zhangxinjie/EdgeTAM/checkpoints/edgetam.pt`
- checkpoint_sha256: `ed2d4850b8792c239689b043c47046ec239b6e808a3d9b6ae676c803fd8780df`
- config: `/home/zhangxinjie/EdgeTAM/sam2/configs/edgetam.yaml`
- backbone_repvit_m1_dist_in1k: `True`
- config_has_compile_image_encoder_false: `True`
- supports_compile_image_encoder: `True`
- build_sam2_video_predictor_has_vos_optimized: `False`
- supports_full_vos_optimized_compile: `False`

## Speed

| backend | status | ms/frame | FPS | warmup FPS | peak CUDA MB | ok cams |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SAM2.1 Small vos_optimized=True | ok | 18.93 | 52.82 | 17.80 | 1455 | 3/3 |
| SAM2.1 Tiny vos_optimized=True | ok | 17.94 | 55.73 | 19.23 | 1449 | 3/3 |
| EdgeTAM eager | ok | 22.16 | 45.13 | 45.95 | 1272 | 3/3 |
| EdgeTAM compile_image_encoder=true | failed | n/a | n/a | n/a | n/a | 0/3 |
| EdgeTAM compile_image_encoder=true + cache clone patch | failed | n/a | n/a | n/a | n/a | 0/3 |
| EdgeTAM compile_image_encoder=true + no position-cache patch | ok | 16.06 | 62.26 | 62.06 | 1039 | 3/3 |
| EdgeTAM manual torch.compile(image_encoder, reduce-overhead) | failed | n/a | n/a | n/a | n/a | 0/3 |

## Failures

- `edgetam_compile_image_encoder` cam0: RuntimeError('Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "/home/zhangxinjie/EdgeTAM/sam2/modeling/backbones/image_encoder.py", line 31, in forward\n    features, pos = self.neck(self.trunk(sample))\n  File "/home/zhangxinjie/EdgeTAM/sam2/modeling/backbones/image_encoder.py", line 132, in forward\n    pos[i] = self.position_encoding(x_out).to(x_out.dtype)\n  File "/home/zhangxinjie/miniconda3/envs/edgetam-max/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context\n    return func(*args, **kwargs)\n  File "/home/zhangxinjie/EdgeTAM/sam2/modeling/position_encoding.py", line 111, in forward\n    self.cache[cache_key] = pos[0]. To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.') log=`/home/zhangxinjie/proj-QQTT-v2/result/edgetam_vs_sam21_compile_fairness_ablation_20260501/logs/edgetam_compile_image_encoder_cam0.log`
- `edgetam_compile_image_encoder` cam1: RuntimeError('Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "/home/zhangxinjie/EdgeTAM/sam2/modeling/backbones/image_encoder.py", line 31, in forward\n    features, pos = self.neck(self.trunk(sample))\n  File "/home/zhangxinjie/EdgeTAM/sam2/modeling/backbones/image_encoder.py", line 132, in forward\n    pos[i] = self.position_encoding(x_out).to(x_out.dtype)\n  File "/home/zhangxinjie/miniconda3/envs/edgetam-max/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context\n    return func(*args, **kwargs)\n  File "/home/zhangxinjie/EdgeTAM/sam2/modeling/position_encoding.py", line 111, in forward\n    self.cache[cache_key] = pos[0]. To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.') log=`/home/zhangxinjie/proj-QQTT-v2/result/edgetam_vs_sam21_compile_fairness_ablation_20260501/logs/edgetam_compile_image_encoder_cam1.log`
- `edgetam_compile_image_encoder` cam2: RuntimeError('Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "/home/zhangxinjie/EdgeTAM/sam2/modeling/backbones/image_encoder.py", line 31, in forward\n    features, pos = self.neck(self.trunk(sample))\n  File "/home/zhangxinjie/EdgeTAM/sam2/modeling/backbones/image_encoder.py", line 132, in forward\n    pos[i] = self.position_encoding(x_out).to(x_out.dtype)\n  File "/home/zhangxinjie/miniconda3/envs/edgetam-max/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context\n    return func(*args, **kwargs)\n  File "/home/zhangxinjie/EdgeTAM/sam2/modeling/position_encoding.py", line 111, in forward\n    self.cache[cache_key] = pos[0]. To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.') log=`/home/zhangxinjie/proj-QQTT-v2/result/edgetam_vs_sam21_compile_fairness_ablation_20260501/logs/edgetam_compile_image_encoder_cam2.log`
- `edgetam_compile_image_encoder_cache_clone_patch` cam0: RuntimeError('Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "/home/zhangxinjie/EdgeTAM/sam2/modeling/backbones/image_encoder.py", line 31, in forward\n    features, pos = self.neck(self.trunk(sample))\n  File "/home/zhangxinjie/EdgeTAM/sam2/modeling/backbones/image_encoder.py", line 132, in forward\n    pos[i] = self.position_encoding(x_out).to(x_out.dtype)\n  File "/home/zhangxinjie/proj-QQTT-v2/scripts/harness/experiments/run_edgetam_vs_sam21_compile_ablation.py", line 285, in forward\n    self.cache[cache_key] = pos[0].detach().clone(). To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.') log=`/home/zhangxinjie/proj-QQTT-v2/result/edgetam_vs_sam21_compile_fairness_ablation_20260501/logs/edgetam_compile_image_encoder_cache_clone_patch_cam0.log`
- `edgetam_compile_image_encoder_cache_clone_patch` cam1: RuntimeError('Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "/home/zhangxinjie/EdgeTAM/sam2/modeling/backbones/image_encoder.py", line 31, in forward\n    features, pos = self.neck(self.trunk(sample))\n  File "/home/zhangxinjie/EdgeTAM/sam2/modeling/backbones/image_encoder.py", line 132, in forward\n    pos[i] = self.position_encoding(x_out).to(x_out.dtype)\n  File "/home/zhangxinjie/proj-QQTT-v2/scripts/harness/experiments/run_edgetam_vs_sam21_compile_ablation.py", line 285, in forward\n    self.cache[cache_key] = pos[0].detach().clone(). To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.') log=`/home/zhangxinjie/proj-QQTT-v2/result/edgetam_vs_sam21_compile_fairness_ablation_20260501/logs/edgetam_compile_image_encoder_cache_clone_patch_cam1.log`
- `edgetam_compile_image_encoder_cache_clone_patch` cam2: RuntimeError('Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "/home/zhangxinjie/EdgeTAM/sam2/modeling/backbones/image_encoder.py", line 31, in forward\n    features, pos = self.neck(self.trunk(sample))\n  File "/home/zhangxinjie/EdgeTAM/sam2/modeling/backbones/image_encoder.py", line 132, in forward\n    pos[i] = self.position_encoding(x_out).to(x_out.dtype)\n  File "/home/zhangxinjie/proj-QQTT-v2/scripts/harness/experiments/run_edgetam_vs_sam21_compile_ablation.py", line 285, in forward\n    self.cache[cache_key] = pos[0].detach().clone(). To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.') log=`/home/zhangxinjie/proj-QQTT-v2/result/edgetam_vs_sam21_compile_fairness_ablation_20260501/logs/edgetam_compile_image_encoder_cache_clone_patch_cam2.log`
- `edgetam_manual_image_encoder_reduce_overhead` cam0: RuntimeError('Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "/home/zhangxinjie/EdgeTAM/sam2/modeling/backbones/image_encoder.py", line 31, in forward\n    features, pos = self.neck(self.trunk(sample))\n  File "/home/zhangxinjie/EdgeTAM/sam2/modeling/backbones/image_encoder.py", line 132, in forward\n    pos[i] = self.position_encoding(x_out).to(x_out.dtype)\n  File "/home/zhangxinjie/miniconda3/envs/edgetam-max/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context\n    return func(*args, **kwargs)\n  File "/home/zhangxinjie/EdgeTAM/sam2/modeling/position_encoding.py", line 111, in forward\n    self.cache[cache_key] = pos[0]. To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.') log=`/home/zhangxinjie/proj-QQTT-v2/result/edgetam_vs_sam21_compile_fairness_ablation_20260501/logs/edgetam_manual_image_encoder_reduce_overhead_cam0.log`
- `edgetam_manual_image_encoder_reduce_overhead` cam1: RuntimeError('Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "/home/zhangxinjie/EdgeTAM/sam2/modeling/backbones/image_encoder.py", line 31, in forward\n    features, pos = self.neck(self.trunk(sample))\n  File "/home/zhangxinjie/EdgeTAM/sam2/modeling/backbones/image_encoder.py", line 132, in forward\n    pos[i] = self.position_encoding(x_out).to(x_out.dtype)\n  File "/home/zhangxinjie/miniconda3/envs/edgetam-max/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context\n    return func(*args, **kwargs)\n  File "/home/zhangxinjie/EdgeTAM/sam2/modeling/position_encoding.py", line 111, in forward\n    self.cache[cache_key] = pos[0]. To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.') log=`/home/zhangxinjie/proj-QQTT-v2/result/edgetam_vs_sam21_compile_fairness_ablation_20260501/logs/edgetam_manual_image_encoder_reduce_overhead_cam1.log`
- `edgetam_manual_image_encoder_reduce_overhead` cam2: RuntimeError('Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "/home/zhangxinjie/EdgeTAM/sam2/modeling/backbones/image_encoder.py", line 31, in forward\n    features, pos = self.neck(self.trunk(sample))\n  File "/home/zhangxinjie/EdgeTAM/sam2/modeling/backbones/image_encoder.py", line 132, in forward\n    pos[i] = self.position_encoding(x_out).to(x_out.dtype)\n  File "/home/zhangxinjie/miniconda3/envs/edgetam-max/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context\n    return func(*args, **kwargs)\n  File "/home/zhangxinjie/EdgeTAM/sam2/modeling/position_encoding.py", line 111, in forward\n    self.cache[cache_key] = pos[0]. To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.') log=`/home/zhangxinjie/proj-QQTT-v2/result/edgetam_vs_sam21_compile_fairness_ablation_20260501/logs/edgetam_manual_image_encoder_reduce_overhead_cam2.log`

## EdgeTAM Compile Note

`edgetam_compile_image_encoder_no_pos_cache_patch` is a process-local benchmark patch. It does not modify `/home/zhangxinjie/EdgeTAM`; it monkey-patches `PositionEmbeddingSine.forward` inside the worker process to avoid writing position encoding tensors into `self.cache` from inside the compiled image encoder. The unpatched EdgeTAM compile modes remain recorded as failures above.

## Decision

- best EdgeTAM mode: `edgetam_compile_image_encoder_no_pos_cache_patch`
- best EdgeTAM FPS: `62.25557303774901`
- best SAM2.1 Small/Tiny FPS: `55.7278071288491`
- EdgeTAM / best Small-or-Tiny: `1.1171366010115373`
- recommendation: Keep SAM2.1 Small as default, SAM2.1 Tiny as fast mode, EdgeTAM as experimental/edge backend.
