## Goal

Add a strict 3-camera batch mode to `cameras_viewer_FFS.py` that runs one shared FFS inference over all 3 active cameras at once, while keeping the existing non-batch PyTorch / two-stage TensorRT / single-engine TensorRT paths unchanged.

## Scope

- add a new viewer CLI mode for strict 3-camera batching
- extend the three current FFS runner implementations with batch inference support
- keep per-camera reprojection and viewer rendering unchanged
- validate TensorRT batch mode against separate batch-3 artifact directories
- update docs, tests, and generated validation notes

## Design

1. add `--ffs_batch_mode {off,strict3}` with default `off`
2. require `strict3` to run only with `--ffs_worker_mode shared` and exactly 3 active cameras
3. add `run_batch(...)` to the PyTorch, two-stage TRT, and single-engine TRT runners; keep `run_pair(...)` as a batch-size-1 wrapper
4. implement strict shared-worker batching with per-camera latest-payload caches so a batch is formed only when all 3 cameras have pending IR pairs
5. validate TensorRT batch mode by inspecting engine I/O tensor shapes and enforcing static batch dimension `3`
6. keep existing batch-1 TensorRT directories intact and generate separate batch-3 proof-of-life directories

## Validation

- deterministic tests cover CLI parsing, startup validation, shared batch scheduling, and batch helper contracts
- non-batch smoke tests still pass unchanged
- `python scripts/harness/check_all.py` passes
- generated batch-3 two-stage TRT and single-engine TRT directories load successfully in the repo-integrated batch runner path
