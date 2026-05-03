# HF EdgeTAM Streaming Realcase Benchmark

- Timestamp UTC: `2026-05-03T19:15:11.681409+00:00`
- Status: `partial`
- Model: `yonigozlan/EdgeTAM-hf`
- Environment: `edgetam-hf-stream`
- Compile mode: `components-reduce-overhead`
- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 5090 Laptop GPU`
- Torch: `2.11.0+cu130`
- Transformers: `5.7.0`
- Jobs: `0/1` passed

## Contract

- Uses real aligned QQTT color frames, not synthetic frames.
- Reads one PNG frame at a time and calls the HF streaming model with a persistent inference session.
- Does not pass a full video path, MP4, or offline video-folder input to EdgeTAM.
- Does not modify `edgetam-max`; intended environment is `edgetam-hf-stream`.
- Measures frame-by-frame streaming with persistent inference sessions.
- This remains an experimental benchmark, not a production backend.

## Compile

- Mode: `components-reduce-overhead`
- Enabled: `True`
- Applied targets: `vision_encoder, memory_attention, memory_encoder, mask_decoder`
- `torch.compile` mode: `reduce-overhead`
- `fullgraph`: `False`
- `dynamic`: `False`

## Mode Summary

| mode | jobs | first median ms | subsequent median ms | e2e FPS median | model FPS median | IoU-to-frame0 min median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

## Jobs

| case | cam | mode | frames | first ms | subsequent median ms | p95 ms | e2e FPS | small IoU mean | tiny IoU mean | failures |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sloth_set_2_motion_ffs | 0 | mask | 0 | n/a | n/a | n/a | n/a | n/a | n/a | 1 |

## Failures

- `sloth_set_2_motion_ffs` cam`0` `mask`: `Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "/home/zhangxinjie/miniconda3/envs/edgetam-hf-stream/lib/python3.12/site-packages/transformers/utils/generic.py", line 976, in wrapper
    output = func(self, *args, **kwargs)
  File "/home/zhangxinjie/miniconda3/envs/edgetam-hf-stream/lib/python3.12/site-packages/transformers/utils/output_capturing.py", line 248, in wrapper
    outputs = func(self, *args, **kwargs)
  File "/home/zhangxinjie/miniconda3/envs/edgetam-hf-stream/lib/python3.12/site-packages/transformers/models/edgetam/modeling_edgetam.py", line 463, in forward
    fpn_hidden_states, fpn_position_encoding = self.neck(intermediate_hidden_states)
  File "/home/zhangxinjie/miniconda3/envs/edgetam-hf-stream/lib/python3.12/site-packages/transformers/models/edgetam/modeling_edgetam.py", line 413, in forward
    prev_features = lateral_features + top_down_features. To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.`
