# Demo 3.1 / 3.2 Protected Bundle Leak And Metrics Fix

## Goal

Fix protected bundle leaks in Demo 3.1 / 3.2 tracker-result-driven rendering and make same-bundle profile metrics distinguish rendered-only success from tracker-result-level miss rate.

## Scope

- Return replaced latest-wins tracker inputs to the runtime and unprotect their frame bundles.
- Add protected-bundle TTL/window pruning as a safety net for stale or dropped tracker inputs.
- Make `BundleStore.take_for_tracker_result()` consume matched bundles.
- Unprotect both tracker result group and matched render/bundle group on nearest-debug paths.
- Fix empty `BundleProvenance` same-bundle proof and add clearer summary ratios.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_demo_services_latest_wins.py tests/test_frame_bundle_service.py tests/test_stage_mailbox.py tests/test_demo31_dual_gpu_contract.py tests/test_demo31_ipc_latest_wins.py -q`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
