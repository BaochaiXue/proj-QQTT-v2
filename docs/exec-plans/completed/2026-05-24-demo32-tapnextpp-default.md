# Demo 3.2 TAPNext++ Default

## Goal

Make Demo 3.2 default to TAPNext++ while keeping LiteTracker available as an
explicit profiling backend.

## Scope

- Change Demo 3.2 preset defaults so no explicit tracker backend resolves to
  `tapnextpp`.
- Keep strict-source same-bundle, FFS batch=3 opt=5, EdgeTAM/SAM on GPU0, and
  tracker child on GPU1 unchanged.
- Keep LiteTracker-specific repo/weight defaults only when LiteTracker is
  explicitly requested.
- Update docs and deterministic contract tests.

## Outcome

- Added `demo3.2-ffs-tapnextpp` as the Demo 3.2 public default preset.
- Kept `demo3.2-ffs-litetracker` as a compatibility/A-B preset.
- Demo 3.2 dry-run now reports:
  - `preset = demo3.2-ffs-tapnextpp`
  - `tracker_backend = tapnextpp`
  - `tracking_backend_execution_mode = serial`
  - `pipeline_order = [..., tapnextpp_serial, ...]`
  - `output_root = result/demo32_ffs_tapnextpp`
- LiteTracker remains available through `--cotracker-backend litetracker` and
  the legacy preset.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m py_compile qqtt/demo/demo31_runtime.py qqtt/demo/demo32_runtime.py demo_v3_2/realtime_three_view_litetracker_ffs_dual4090.py`
- `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_demo31_dual_gpu_contract.py -q`
- `conda run -n demo_2_max --no-capture-output python demo_v3_2/realtime_three_view_litetracker_ffs_dual4090.py --dry-run --camera-ids 0,1,2 --mask-gpu 0 --cotracker-gpu 1 --require-two-cuda --calibrate-path calibrate.pkl`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
- `git diff --check`
