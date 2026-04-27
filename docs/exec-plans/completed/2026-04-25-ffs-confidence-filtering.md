# FFS Confidence Filtering

Add optional confidence-based filtering for Fast-FoundationStereo aligned depth.

## Scope

- Extend PyTorch logits confidence proxies to include `margin`, `max_softmax`,
  `entropy`, and `variance`.
- Keep confidence filtering disabled by default.
- Filter FFS color-aligned depth before quantization when explicitly requested.
- Keep canonical/formal depth output as uint16 depth with invalid pixels encoded
  as zero.
- Add optional uint8 confidence and valid-mask debug outputs behind explicit
  flags.
- Add a sweep harness that evaluates modes and thresholds without requiring
  Open3D.

## Plan

1. Inspect the existing FFS runner, confidence visualization helpers, and
   alignment path.
2. Extend `compute_confidence_proxies_from_logits()` and
   `run_pair_with_confidence()` return fields.
3. Add a shared confidence filtering helper for uint16 depth generation and
   stats.
4. Wire optional filtering into `data_process/record_data_align.py` using the
   existing color reprojection logic.
5. Add `scripts/harness/run_ffs_confidence_filter_sweep.py`.
6. Add smoke tests for confidence proxies, filtering, CLI integration, and debug
   outputs.
7. Document usage, output contract, and interpretation.

## Validation

- `conda run -n qqtt-ffs-compat python -m unittest -v tests.test_ffs_confidence_filtering_smoke tests.test_ffs_confidence_panels_smoke tests.test_depth_quantization_smoke`
- `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py`
