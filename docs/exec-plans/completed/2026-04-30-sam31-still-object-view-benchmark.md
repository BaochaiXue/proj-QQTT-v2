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

- `python scripts/harness/benchmark_sam31_still_object_views.py --help`: passed.
- `python -m unittest -v tests.test_sam31_still_object_benchmark_smoke tests.test_check_all_smoke`: passed.
- `python scripts/harness/check_harness_catalog.py`: passed.
- `conda run -n FFS-SAM-RS python scripts/harness/check_all.py`: passed, including 72 quick unittest cases.

## Result

- Added `scripts/harness/benchmark_sam31_still_object_views.py`.
- Added `tests/test_sam31_still_object_benchmark_smoke.py`.
- Registered the benchmark in `scripts/harness/_catalog.py`, `scripts/harness/README.md`, and `scripts/harness/check_all.py`.
- Ran the real benchmark in `FFS-SAM-RS` on `both_30_still_object_round1_20260428`, cameras `0 1 2`, `30` frames per camera, prompt `stuffed animal`, RTX 5090 Laptop GPU.
- Timing summary written to `data/experiments/sam31_still_object_view_benchmark_both_30_still_object_round1_20260428/sam31_still_object_view_benchmark.json`.

## Timing

| Camera | Frames | Prompt s | Propagate s | Segment ms/frame | Total s | Total ms/frame |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 30 | 1.115 | 5.962 | 235.88 | 22.176 | 739.21 |
| 1 | 30 | 0.347 | 7.956 | 276.79 | 19.610 | 653.67 |
| 2 | 30 | 0.473 | 64.242 | 2157.17 | 73.606 | 2453.53 |

Average segment time: `889.95 ms/frame`, `26.698 s/camera`.

Average total time: `1282.14 ms/frame`, `38.464 s/camera`.
