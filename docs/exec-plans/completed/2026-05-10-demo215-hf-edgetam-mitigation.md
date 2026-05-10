# 2026-05-10 Demo 2.1.5 HF EdgeTAM Mitigation

## Goal

Add low-risk instrumentation and experimental switches for Demo 2.1.5 HF
EdgeTAM underutilization mitigation. The KPI is EdgeTAM model p50/p90 and
end-to-end p90, not GPU utilization alone.

## Scope

- Keep production defaults unchanged.
- Add flags for detailed EdgeTAM stage profiling and NVTX ranges.
- Add an experimental CUDA-inline mask postprocess path.
- Expose existing dtype / compile / profiling flags through the Demo 2.1.5
  public wrapper.
- Generate a deterministic benchmark manifest and report template for the
  Phase 0-4 matrix.

## Plan

1. Extend runtime args and contract with `--profile-edgetam-stages`,
   `--profile-nsys-markers`, and `--mask-postprocess hf|cuda-inline`.
2. Time EdgeTAM preprocess, H2D, prompt add, model forward, mask resize,
   threshold, mask CPU transfer, and postprocess total.
3. Add NVTX ranges around those stages when requested.
4. Add summary metrics and Markdown sections to profile reports.
5. Add a benchmark-matrix helper script that records commands and existing
   profiles into:
   - `docs/generated/demo215_hf_edgetam_gpu_underutilization_mitigation.md`
   - `docs/generated/demo215_hf_edgetam_gpu_underutilization_mitigation.json`
6. Add deterministic smoke tests for flags, contract, and cuda-inline mask
   extraction.

## Validation

- `conda run --no-capture-output -n demo_2_max python -m py_compile demo_v2/realtime_masked_edgetam_pcd.py`
- `conda run --no-capture-output -n demo_2_max python -m unittest tests.test_demo_v2_1_5_realsense_depth_smoke tests.test_demo_v2_1_three_view_fused_pcd_smoke`
- `conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py`
