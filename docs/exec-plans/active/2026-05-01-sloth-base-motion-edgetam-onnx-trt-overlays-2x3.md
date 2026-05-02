# 2026-05-01 sloth_base_motion EdgeTAM ONNX/TRT Overlays 2x3

## Goal

Render two 2x3 GIFs for `data/different_types/sloth_base_motion_ffs`:

- RGB mask XOR overlay vs SAM3.1
- fused PCD overlay vs SAM3.1

The comparison rows are `SAM2.1 Small` and `EdgeTAM ONNX/TRT per-frame
component sanity`; columns are `cam0/cam1/cam2`.

## Plan

1. Reuse existing SAM3.1 masks, SAM2.1 Small masks, and the ONNX/TRT component
   benchmark artifacts under `result/edgetam_onnx_trt_probe_20260501/`.
2. If a compatible EdgeTAM ONNX/TRT component sanity mask cache is missing,
   generate it from the TensorRT vision encoder + prompt/mask decoder engines
   using SAM3.1 per-frame boxes. Do not label this as video tracking output.
3. Record ONNX/TRT FPS from the existing encoder+decoder component benchmark,
   and clearly label it as component-level runtime rather than full video
   predictor propagation.
4. Render:
   - RGB overlay 2x3 GIF
   - fused PCD overlay 2x3 GIF using the existing enhanced PhysTwin-like PCD
     postprocess and explicit `depth_scale_override_m_per_unit=0.001`.
5. Write docs/generated markdown and JSON summaries.

## Validation

- Focused deterministic unittest for overlay/timing helpers.
- `conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py`
- Verify generated GIF frame counts and first-frame outputs.
