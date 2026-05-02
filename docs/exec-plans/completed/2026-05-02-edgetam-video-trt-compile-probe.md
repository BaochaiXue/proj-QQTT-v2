# EdgeTAM Video TensorRT Compile Probe

## Goal

Try to compile the missing EdgeTAM video-tracking components for the local RTX
5090 Laptop WSL stack, without changing production runtime defaults.

## Scope

- Keep existing EdgeTAM component engines untouched:
  - vision encoder
  - prompt/mask decoder
- Add an experiment-only probe that attempts:
  - video SAM head export with `obj_ptr`
  - mask-input conditioning head export with `obj_ptr`
  - memory encoder plus spatial perceiver export
  - memory attention export
- For every attempt, record:
  - eager forward success/failure
  - ONNX export success/failure
  - ONNX checker success/failure
  - TensorRT build success/failure
  - error message and log path

## Outputs

- `scripts/harness/experiments/probe_edgetam_video_trt_compile.py`
- `result/edgetam_video_trt_compile_probe_20260502/`
- `docs/generated/edgetam_video_trt_compile_probe.md`
- `docs/generated/edgetam_video_trt_compile_probe.json`

## Result

- `memory_encoder_spatial_perceiver`: ONNX export/check passed; TensorRT engine built.
- `memory_attention_one_previous_real_rope_patch`: ONNX export/check passed; TensorRT engine built.
- Raw `memory_attention_one_previous`: ONNX export failed on `ComplexFloat` RoPE tensors.
- Raw `video_sam_heads_mask_input`: ONNX export failed on antialiased bilinear upsample.
- `video_sam_heads_no_prompt` and no-antialias mask-input SAM heads exported to ONNX, but TensorRT parsing failed at `sam_prompt_encoder/Where_6` broadcast shape handling.

## Verification

- `conda run --no-capture-output -n SAM21-max python -m py_compile scripts/harness/experiments/probe_edgetam_video_trt_compile.py`
- `python scripts/harness/check_harness_catalog.py`
- `conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py`

Note: `python scripts/harness/check_all.py` with the base Python failed because that interpreter does not have `cv2` installed; the validated `SAM21-max` run passed.
