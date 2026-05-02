# EdgeTAM ONNX/TensorRT Component Probe

## Goal

Evaluate whether EdgeTAM can be accelerated through ONNX/TensorRT components on
the RTX 5090 Laptop without changing production defaults.

## Scope

- Keep production defaults unchanged:
  - default: SAM2.1 Small
  - fast mode: SAM2.1 Tiny
  - experimental: EdgeTAM compiled no-position-cache
- Do not attempt to compile the full EdgeTAM video predictor into one engine.
- Probe component assets from `onnx-community/EdgeTAM-ONNX`.
- Build TensorRT engines with `trtexec` when the ONNX graph is accepted.
- Benchmark component runtime on real frames from
  `data/different_types/sloth_base_motion_ffs`.
- Write all probe artifacts under `result/edgetam_onnx_trt_probe_20260501/`.

## Validation / Outputs

- `docs/generated/edgetam_onnx_trt_probe.md`
- `docs/generated/edgetam_onnx_trt_probe.json`
- `git diff --check`
- `conda run -n FFS-SAM-RS python scripts/harness/check_all.py`

## Outcome

- Downloaded and inspected the community fp16 ONNX component assets.
- Built TensorRT engines for the vision encoder and prompt/mask decoder.
- Benchmarked the component runtime on
  `data/different_types/sloth_base_motion_ffs`.
- Recorded artifacts under `result/edgetam_onnx_trt_probe_20260501/`.

Key runtime result:

```text
encoder + decoder aggregate: 4.490 ms, 222.74 FPS
```

This is a component-level result only. It is not a complete EdgeTAM video
predictor TensorRT runtime because memory/state propagation remains outside the
TRT path.
