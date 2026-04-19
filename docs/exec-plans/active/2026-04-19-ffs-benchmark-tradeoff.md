# 2026-04-19 FFS Benchmark Tradeoff

## Goal

Add a reproducible Fast-FoundationStereo benchmark harness that runs on aligned saved stereo pairs, measures warmup-adjusted throughput, and summarizes practical speed-vs-reference-quality tradeoffs for online-setting decisions.

## Non-Goals

- no change to canonical aligned `depth/` outputs
- no TensorRT engine build automation inside this repo
- no mandatory live-camera benchmark dependency
- no object masking or floating-point cleanup changes in this pass

## Files To Touch

- new `data_process/depth_backends/benchmarking.py`
- new `scripts/harness/benchmark_ffs_configs.py`
- new benchmark-focused unit tests
- `scripts/harness/check_all.py`
- workflow / architecture / generated validation docs

## Implementation Plan

1. add pure benchmark helpers for:
   - config-grid expansion
   - latency/fps summary stats
   - reference-depth agreement metrics
   - target-fps tradeoff selection
2. add a thin CLI harness that:
   - loads one aligned case frame's `ir_left` / `ir_right` plus FFS geometry
   - runs a sweep over checkpoint / scale / valid_iters / max_disp
   - records latency, fps, peak memory, and reference-agreement metrics
   - writes `summary.json` plus a readable markdown report
3. add deterministic tests for config expansion and summary selection without requiring CUDA
4. wire the new CLI into deterministic `--help` coverage
5. run at least one real local benchmark and record exact commands and outcomes under `docs/generated/`

## Validation Plan

- `python scripts/harness/benchmark_ffs_configs.py --help`
- `python -m unittest -v tests.test_ffs_benchmarking_smoke`
- `python scripts/harness/check_all.py`
