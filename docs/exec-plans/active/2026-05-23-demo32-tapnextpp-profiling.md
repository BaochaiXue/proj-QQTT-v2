# Demo 3.2 TAPNext++ Serial vs Batch Profiling

## Goal

Enable Demo 3.2 to use TAPNext++ as an explicit tracker backend, then profile
strict same-bundle rendered output for TAPNext++ serial and batch-view modes.

## Scope

- TAPNext++ has become the Demo 3.2 default after the serial-vs-batch profile.
- Keep `--cotracker-backend litetracker` available for Demo 3.2 A/B profiling.
- Keep FFS batch=3 opt=5, EdgeTAM, strict-source same-bundle render, and
  tracker-child IPC semantics unchanged.
- Update contract labels, docs, and tests so Demo 3.2 backend identity is not
  hard-coded to LiteTracker.

## Validation

- Run targeted Demo 3.2 contract tests.
- Run `scripts/harness/check_all.py`.
- Dry-run Demo 3.2 TAPNext++ serial and batch-view modes.
- Run 120s rendered profiles for TAPNext++ serial and batch-view modes when
  the local TAPNext++ runtime is available.

## Results

- Implemented Demo 3.2 TAPNext++ backend support; a later default-policy change
  made TAPNext++ serial the Demo 3.2 default while keeping LiteTracker explicit.
- Demo 3.2 contract labels now reflect the selected tracker:
  `tapnextpp_serial`, `tapnextpp_batch3`, and
  `ffs_edgetam_gpu0_tapnextpp_gpu1`.
- LiteTracker ONNX validation remains LiteTracker-only.
- Top-level/pre-teardown summaries now merge query, overlay, and all-tracks
  lift metrics from the tracker snapshot so generated profiles do not hide
  child-process values.
- Validation passed:
  - `python -m py_compile qqtt/demo/demo31_runtime.py qqtt/demo/demo32_runtime.py`
  - `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_demo31_dual_gpu_contract.py tests/test_demo3_cotracker_worker.py -q`
  - `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
- 120s live strict same-bundle profiles completed locally:
  - serial: `docs/generated/demo32_tapnextpp_serial_strict_120s_profile.json`
  - batch=3: `docs/generated/demo32_tapnextpp_batch3_strict_120s_profile.json`
