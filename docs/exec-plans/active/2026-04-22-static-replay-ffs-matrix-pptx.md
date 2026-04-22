# 2026-04-22 Static Replay FFS Matrix + PPTX

## Goal

Add one offline experiment harness that:

- replays the three static aligned FFS rounds as a batch-1 realtime proxy
- searches the requested `model × scale × valid_iters × engine` matrix
- builds or reuses the needed TensorRT artifacts
- records per-round / per-camera FPS
- renders frame-10 masked `3x3` RGB and FFS-only masked PCD boards
- exports one globally ranked PPTX sorted by overall mean FPS

## Scope

- new orchestration CLI under `scripts/harness/`
- reuse the existing FFS runners, SAM 3.1 mask helper, PhysTwin-like mask refinement, and Open3D hidden-window renderer
- keep the experiment output under `data/experiments/`
- update docs, tests, and generated validation notes

## Non-Goals

- no change to `record_data_align.py`
- no change to the live camera viewer
- no change to the camera-only environment installer
- no attempt to add new external runtime repos or move external weights/checkpoints into this repo

## Files To Touch

- `scripts/harness/run_ffs_static_replay_matrix.py`
- `scripts/harness/check_all.py`
- `tests/test_ffs_static_replay_matrix_smoke.py`
- `README.md`
- `docs/WORKFLOWS.md`
- `docs/ARCHITECTURE.md`
- `docs/generated/README.md`
- generated validation note for the final run

## Implementation Plan

1. add deterministic matrix / artifact / ranking helpers and a full experiment CLI
2. reuse the current SAM 3.1 helper to cache frame-10 `stuffed animal` masks for the three static rounds
3. benchmark the 24 TRT-only configs with:
   - warmup frames `0..9`
   - measured frames `0..29`
   - one reused runner per experiment
4. render:
   - one shared masked RGB `3x3` board
   - one masked FFS-only PCD `3x3` board per successful experiment
5. export sorted CSV / JSON / PPTX outputs and record the exact run in `docs/generated/`

## Validation Plan

- `python -m unittest -v tests.test_ffs_static_replay_matrix_smoke`
- `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py`
- full 24-config experiment run in `qqtt-ffs-compat`
