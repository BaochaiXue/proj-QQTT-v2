# 2026-04-22 Static Replay Concurrent 3-View Rerun

## Goal

Replace the old static replay single-camera sequential benchmark semantics with
simultaneous 3-camera per-round benchmarking, then rerun the full 54-config
matrix and regenerate the PPTX.

## Scope

- update `scripts/harness/run_ffs_static_replay_matrix.py` to benchmark 3 views
  concurrently per round
- add `--artifact_root` so TRT artifacts can be reused independently of the new
  output root
- stop reusing old benchmark summaries/results
- regenerate a fresh output root, CSV, manifest, validation note, and PPT

## Non-Goals

- no change to the 54-config search space
- no change to batch size; keep `batch=1`
- no change to the PCD-only slide contract

## Validation

- `python -m unittest -v tests.test_ffs_static_replay_matrix_smoke`
- `python scripts/harness/check_all.py`
- full rerun with a fresh `output_root` and reused `artifact_root`
