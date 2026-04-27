# 2026-04-27 Harness Experiment Separation

## Goal

Separate recent experiment-only FFS visualization work from formal recording,
alignment, and stable comparison workflow code.

## Non-Goals

- no behavior change to camera preview, recording, calibration, or alignment
- no deletion of generated data under `data/` or `data_collect/`
- no removal of existing harness command paths; compatibility wrappers should
  remain where useful

## Planned Actions

1. Create explicit experiment namespaces:
   - `data_process/visualization/experiments/`
   - `scripts/harness/experiments/`
2. Move the recent static FFS experiment implementations and CLIs into those
   namespaces.
3. Leave thin compatibility wrappers at existing `scripts/harness/*.py`
   command paths and import-compatible workflow wrappers where needed.
4. Add a deterministic boundary guard so formal entrypoints and low-level
   packages cannot import experiment modules.
5. Update harness / architecture docs and validation lists to use the new
   experiment namespace as the canonical home.
6. Run targeted tests plus the repo quick validation.

## Validation

- `python scripts/harness/check_experiment_boundaries.py`
- `python scripts/harness/check_visual_architecture.py`
- `python -m unittest -v tests.test_experiment_boundary_smoke tests.test_check_all_smoke`
- `python scripts/harness/check_all.py`
