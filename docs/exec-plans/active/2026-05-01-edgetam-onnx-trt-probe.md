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
