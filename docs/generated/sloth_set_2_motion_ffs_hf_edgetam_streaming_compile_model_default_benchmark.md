# HF EdgeTAM Streaming Realcase Benchmark

- Timestamp UTC: `2026-05-03T19:14:44.801590+00:00`
- Status: `partial`
- Model: `yonigozlan/EdgeTAM-hf`
- Environment: `edgetam-hf-stream`
- Compile mode: `model-default`
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

- Mode: `model-default`
- Enabled: `True`
- Applied targets: `<model>`
- `torch.compile` mode: `default`
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

- `sloth_set_2_motion_ffs` cam`0` `mask`: `backend='inductor' raised:
RuntimeError: <weakref at 0x738e8f42a660; to 'torch.storage.UntypedStorage' at 0x738e8fb30a10>

While executing %image_batch : [num_users=1] = call_method[target=unsqueeze](args = (%to_1, 0), kwargs = {})
Original traceback:
  File "/home/zhangxinjie/miniconda3/envs/edgetam-hf-stream/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
  File "/home/zhangxinjie/miniconda3/envs/edgetam-hf-stream/lib/python3.12/site-packages/transformers/models/edgetam_video/modeling_edgetam_video.py", line 2192, in forward
    current_out = self._run_single_frame_inference(
  File "/home/zhangxinjie/miniconda3/envs/edgetam-hf-stream/lib/python3.12/site-packages/transformers/models/edgetam_video/modeling_edgetam_video.py", line 2950, in _run_single_frame_inference
    current_vision_feats, current_vision_pos_embeds = self._prepare_vision_features(
  File "/home/zhangxinjie/miniconda3/envs/edgetam-hf-stream/lib/python3.12/site-packages/transformers/models/edgetam_video/modeling_edgetam_video.py", line 2279, in _prepare_vision_features
    image_batch = inference_session.get_frame(frame_idx).unsqueeze(0)  # Add batch dimension

Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs)

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"
`
