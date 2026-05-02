# 2026-05-02 HF EdgeTAM Streaming Realcase Benchmark

## Goal

Promote the HF EdgeTAM synthetic streaming proof into a real QQTT frame-by-frame benchmark using aligned cases, without modifying `edgetam-max`.

## Plan

1. Add an experiment harness CLI that runs in `edgetam-hf-stream` and loads real aligned RGB frames.
2. Reuse the established case set:
   - `still_object_round1` through `still_object_round4`
   - `still_rope_round1` and `still_rope_round2`
   - `ffs_dynamics_round1`
3. Support point, box, mask, and optional previous-mask prompt modes.
4. Record preprocessing, model forward, postprocessing, first-frame, subsequent-frame, and end-to-end streaming timing.
5. Compute mask quality metrics, including area stability, centroid/bbox drift, frame-t vs frame-0 IoU, IoU against SAM2.1 Small/Tiny masks when available, and FFS depth point-count deltas.
6. Write generated JSON/Markdown reports under `docs/generated/`.
7. Register the new experiment script in the harness catalog and docs.
8. Run a small smoke benchmark, then run the full requested real-case sweep if runtime is acceptable.
9. Run deterministic repo checks.

## Validation

- `conda run --no-capture-output -n edgetam-hf-stream python scripts/harness/experiments/run_hf_edgetam_streaming_realcase.py ...`
- `conda run --no-capture-output -n SAM21-max python scripts/harness/experiments/run_hf_edgetam_streaming_realcase.py --help`
- `conda run --no-capture-output -n SAM21-max python scripts/harness/check_harness_catalog.py`
- `conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py`

## Outcome

Fill in after validation.
