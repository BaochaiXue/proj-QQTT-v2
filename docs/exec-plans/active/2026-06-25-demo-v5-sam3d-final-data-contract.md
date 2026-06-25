# Demo v5 SAM3D Final Data Contract Plan

Goal: make Demo v5 camera-to-final_data artifacts enforce the same semantic
quality contract as `data_process_sam3d` while preserving 5 FPS realtime chunk
continuity for `realtime_phystwin`.

## Scope

- Keep the active runtime path as `realtime_camera_final_data.py` ->
  `headless_chunk_bridge.py` -> `futurephystwin_chunk_writer.py` ->
  `online_chunk_output.py`.
- Do not introduce shadow quality helpers under `demo_v5`.
- Keep the realtime 5 FPS defaults and dual GPU warmup/runtime split unchanged.
- Strengthen product validation so a case can pass only when final_data sample
  ids, topology ids, semantic labels, and point columns agree.
- Preserve the bounded KNN revive already in the active streaming selectors for
  chunk continuity.

## Implementation Tasks

1. [x] Add failing tests for Demo v5 dead KNN helper removal and final_data
   sample-id semantic validation.
2. [x] Add a runtime contract marker and explicit SAM3D realtime semantic checks
   to the active FuturePhysTwin chunk writer/validator.
3. [x] Remove unused `demo_v5/knn_recovery.py` so tracking recovery has only one
   active implementation.
4. [x] Update README contract text and this plan with outcomes.
5. [ ] Run targeted unit tests, dry-run, guards, smoke validation, then commit
   and push to `origin single-camera`.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_realtime_phystwin tests.test_phystwin_strict_product realtime_phystwin.tests.test_online_topology_contract`
- `conda run -n demo_2_max --no-capture-output python demo_v5/realtime_futurephystwin_chunks.py --dry-run`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/guards/check_scope.py`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/guards/check_experiment_boundaries.py`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/guards/check_visual_architecture.py`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
