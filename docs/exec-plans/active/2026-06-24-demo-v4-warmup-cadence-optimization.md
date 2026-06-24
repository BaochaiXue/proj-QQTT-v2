# Demo v4 Warmup And Cadence Optimization

## Goal

Optimize Demo v4 so FuturePhysTwin-compatible output remains complete and
quality-consistent while:

- full end-to-end warmup, measured from Demo v4 start to the first complete
  `READY` chunk with shape prior, targets 60 seconds or less;
- fake-live/live camera to `final_data.pkl` maintains 5 FPS chunk cadence;
- warmup GPU routing and realtime GPU routing are independently configurable;
- dual-GPU warmup plus single-GPU realtime is supported as an explicit combo.

The primary acceptance metric is cold shape-prior generation without reusing
old surface/interior point caches. Cache reuse may be reported separately for
repeat experiments, but it does not count as the main 60-second proof.

## Non-Negotiable Product Contract

- Keep FuturePhysTwin case roots complete: `final_data.pkl`,
  `track_process_data.pkl`, masks, RGB, dense PCD, tracking/cotracker,
  calibration, metadata, split, manifest, and `READY`.
- Keep strict semantic/motion rules: first-frame object/controller identity,
  depth-valid mask gating, radius-outlier mask refinement, PhysTwin-style
  motion filtering, controller FPS 30, object 5 mm sampling, and 700/1000
  shape-prior target counts for product runs.
- Do not publish a chunk until validation, manifest, and `READY` are complete
  inside staging and the final case directory is atomically renamed into place.

## Implementation Steps

- [x] Split Demo v4 routing into `--realtime-gpu-mode` and
  `--warmup-gpu-mode`, while keeping `--gpu-mode` as a realtime compatibility
  alias.
- [x] Add manifest quality fields for object/controller counts, shape-prior
  presence/target counts, finiteness, and zero-depth first-frame counts.
- [x] Add READY-visible publish telemetry:
  `window_closed_wall_s`, `track_finalize_done_wall_s`,
  `final_data_written_wall_s`, `validation_done_wall_s`,
  `atomic_rename_done_wall_s`, `publish_latency_ms`, and backlog.
- [x] Change shape-prior interior sampling to deterministic voxel/raycast
  candidate generation first, with the old volume sampler retained as fallback.
- [ ] Run focused tests, smoke validation, and a clean-GPU cold SAM3D benchmark.
- [ ] If single-GPU cold first READY remains above 60 seconds, run the explicit
  `--warmup-gpu-mode dual --realtime-gpu-mode single` benchmark.
- [ ] Record exact commands, GPU assignment, profiles, chunk cadence, and
  FuturePhysTwin quality checks under `docs/generated/`.

## Validation Commands

```bash
conda run -n demo_2_max --no-capture-output \
  python -m pytest tests/test_demo_v4_futurephystwin_chunks.py \
  tests/test_phystwin_strict_product.py \
  tests/test_shape_prior_sampling_optimization.py \
  tests/test_demo32_shape_prior_warmup.py -q

conda run -n demo_2_max --no-capture-output \
  python scripts/harness/validation/run.py --profile smoke
```

## Experiment Gates

- `first_shape_prior_ready_chunk_wall_s <= 60.0` for the main target.
- steady-state `publish_wall_s` interval is no larger than the 5-second source
  window; any excursions must be reported, not hidden.
- `max_backlog_chunks` does not grow after startup.
- validation chunks have finite object/controller/surface/interior points,
  controller count 30, and product-run shape-prior target counts 700/1000.
