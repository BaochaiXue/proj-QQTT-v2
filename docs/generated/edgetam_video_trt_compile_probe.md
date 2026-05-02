# EdgeTAM Video TensorRT Compile Probe

## Scope

This probe attempted component-level export/build for the EdgeTAM pieces missing from the existing ONNX/TRT sanity path.
It does not implement a full `propagate_in_video(state)` scheduler.

## Environment

- EdgeTAM repo: `/home/zhangxinjie/EdgeTAM`
- EdgeTAM commit: `7711e012a30a2402c4eaab637bdb00a521302c91`
- checkpoint: `/home/zhangxinjie/EdgeTAM/checkpoints/edgetam.pt`
- checkpoint sha256: `ed2d4850b8792c239689b043c47046ec239b6e808a3d9b6ae676c803fd8780df`
- torch: `2.11.0+cu130`
- torch CUDA: `13.0`
- GPU: `NVIDIA GeForce RTX 5090 Laptop GPU`
- TensorRT: `10.16.1.11`
- output dir: `/home/zhangxinjie/proj-QQTT-v2/result/edgetam_video_trt_compile_probe_20260502`

## Results

| component | eager | ONNX export | ONNX check | TensorRT build |
| --- | --- | --- | --- | --- |
| video_sam_heads_no_prompt | ok | ok | ok | failed |
| video_sam_heads_mask_input | ok | failed | not_run | not_run |
| video_sam_heads_mask_input_no_antialias_patch | ok | ok | ok | failed |
| memory_encoder_spatial_perceiver | ok | ok | ok | ok |
| memory_attention_one_previous | ok | failed | not_run | not_run |
| memory_attention_one_previous_real_rope_patch | ok | ok | ok | ok |

## TensorRT Engines

| component | engine | size |
| --- | --- | ---: |
| memory_encoder_spatial_perceiver | `/home/zhangxinjie/proj-QQTT-v2/result/edgetam_video_trt_compile_probe_20260502/memory_encoder_spatial_perceiver/memory_encoder_spatial_perceiver.engine` | 10.36 MiB |
| memory_attention_one_previous_real_rope_patch | `/home/zhangxinjie/proj-QQTT-v2/result/edgetam_video_trt_compile_probe_20260502/memory_attention_one_previous_real_rope_patch/memory_attention_one_previous_real_rope_patch.engine` | 8.59 MiB |

## Failure Details

### video_sam_heads_no_prompt
- trt_build: `failed` [05/01/2026-23:03:56] [E] Error[4]: ITensor::getDimensions: Error Code 4: Shape Error (broadcast dimensions must be conformable In op at /_src/optimizer/common/shape/shapeContext.cpp:2700) / [05/01/2026-23:03:56] [E] [TRT] ModelImporter.cpp:138: While parsing node number 103 [Where -> "/sam_prompt_encoder/Where_6_output_0"]: / [05/01/2026-23:03:56] [E] [TRT] ModelImporter.cpp:140: --- Begin node --- / [05/01/2026-23:03:56] [E] [TRT] ModelImporter.cpp:141: --- End node --- / [05/01/2026-23:03:56] [E] [TRT] ModelImporter.cpp:149: ERROR: ModelImporter.cpp:506 In function parseNode: / [05/01/2026-23:03:56] [E] Failed to parse onnx file / [05/01/2026-23:03:56] [E] Parsing model failed / [05/01/2026-23:03:56] [E] Failed to create engine from model or file. / [05/01/2026-23:03:56] [E] Engine set up failed / &&&& FAILED TensorRT.trtexec [TensorRT v101601] [b11] # /opt/TensorRT-10.16.1.11/bin/trtexec --onnx=/home/zhangxinjie/proj-QQTT-v2/result/edgetam_video_trt_compile_probe_20260502/video_sam_heads_no_prompt/video_sam_heads_no_prompt.onnx --saveEngine=/home/zhangxinjie/proj-QQTT-v2/result/edgetam_video_trt_compile_probe_20260502/video_sam_heads_no_prompt/video_sam_heads_no_prompt.engine --builderOptimizationLevel=5 --skipInference --profilingVerbosity=detailed --fp16
- log: `/home/zhangxinjie/proj-QQTT-v2/result/edgetam_video_trt_compile_probe_20260502/video_sam_heads_no_prompt/video_sam_heads_no_prompt_trtexec.log`

### video_sam_heads_mask_input
- onnx_export: `UnsupportedOperatorError` Exporting the operator 'aten::_upsample_bilinear2d_aa' to ONNX opset version 17 is not supported

### video_sam_heads_mask_input_no_antialias_patch
- trt_build: `failed` [05/01/2026-23:03:57] [E] Error[4]: ITensor::getDimensions: Error Code 4: Shape Error (broadcast dimensions must be conformable In op at /_src/optimizer/common/shape/shapeContext.cpp:2700) / [05/01/2026-23:03:57] [E] [TRT] ModelImporter.cpp:138: While parsing node number 118 [Where -> "/sam_prompt_encoder/Where_6_output_0"]: / [05/01/2026-23:03:57] [E] [TRT] ModelImporter.cpp:140: --- Begin node --- / [05/01/2026-23:03:57] [E] [TRT] ModelImporter.cpp:141: --- End node --- / [05/01/2026-23:03:57] [E] [TRT] ModelImporter.cpp:149: ERROR: ModelImporter.cpp:506 In function parseNode: / [05/01/2026-23:03:57] [E] Failed to parse onnx file / [05/01/2026-23:03:57] [E] Parsing model failed / [05/01/2026-23:03:57] [E] Failed to create engine from model or file. / [05/01/2026-23:03:57] [E] Engine set up failed / &&&& FAILED TensorRT.trtexec [TensorRT v101601] [b11] # /opt/TensorRT-10.16.1.11/bin/trtexec --onnx=/home/zhangxinjie/proj-QQTT-v2/result/edgetam_video_trt_compile_probe_20260502/video_sam_heads_mask_input_no_antialias_patch/video_sam_heads_mask_input_no_antialias_patch.onnx --saveEngine=/home/zhangxinjie/proj-QQTT-v2/result/edgetam_video_trt_compile_probe_20260502/video_sam_heads_mask_input_no_antialias_patch/video_sam_heads_mask_input_no_antialias_patch.engine --builderOptimizationLevel=5 --skipInference --profilingVerbosity=detailed --fp16
- log: `/home/zhangxinjie/proj-QQTT-v2/result/edgetam_video_trt_compile_probe_20260502/video_sam_heads_mask_input_no_antialias_patch/video_sam_heads_mask_input_no_antialias_patch_trtexec.log`

### memory_attention_one_previous
- onnx_export: `RuntimeError` ScalarType ComplexFloat is an unexpected tensor scalar type

## Interpretation

- A TensorRT build success here means a component can become part of a future scheduler.
- A full EdgeTAM video TRT backend still needs host-side state management for `maskmem_features`, `maskmem_pos_enc`, `obj_ptr`, frame selection, and memory ring updates.
- The existing ONNX/TRT component result remains frame-level SAM-style mask sanity until these memory components and scheduler pass correctness checks.

## Practical Next Step

The memory side is now compile-feasible with local wrapper patches. The SAM prompt/video head still needs a TensorRT-friendly prompt encoder rewrite, because the current PyTorch-exported ONNX graph fails TensorRT parsing at `sam_prompt_encoder/Where_6` broadcast shape handling.
