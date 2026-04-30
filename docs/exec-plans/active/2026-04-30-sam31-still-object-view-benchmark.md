# 2026-04-30 SAM 3.1 Still-Object View Benchmark

## Goal

Add a harness benchmark that times SAM 3.1 segmentation on a single still-object camera view for 30 frames, repeats it for camera views 0, 1, and 2, and reports per-view plus overall average timing.

## Scope

- Add an operator-facing harness CLI for still-object SAM 3.1 segmentation timing.
- Reuse the existing still-object raw case registry and SAM 3.1 helper code.
- Default to `still_object/round1`, 30 RGB frames, cameras `0 1 2`, prompt `stuffed animal`.
- Write a JSON timing summary under `data/experiments/`.
- Add deterministic smoke coverage that does not require the SAM runtime.
- Register the new harness CLI in the harness catalog and README.

## Non-Goals

- Do not change SAM 3.1 mask generation behavior for existing workflows.
- Do not move or rename existing harness CLIs.
- Do not add CI hardware or GPU requirements.

## Validation

- `python scripts/harness/benchmark_sam31_still_object_views.py --help`
- `python -m unittest -v tests.test_sam31_still_object_benchmark_smoke`
- `python scripts/harness/check_harness_catalog.py`
- `python scripts/harness/check_all.py`

## Progress

- Plan opened.
