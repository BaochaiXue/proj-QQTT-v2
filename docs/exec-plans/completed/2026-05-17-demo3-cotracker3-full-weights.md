# Demo 3 CoTracker3 Full Weights

## Goal

Pull the complete released CoTracker3 checkpoint set and make it ready for the
Demo 3.0 RealSense CoTracker overlay path.

## Scope

- Keep external CoTracker source and weights outside this repo.
- Do not change Demo 3 runtime code in this pass.
- Preserve the current experiment convention: object `stuffed animal`,
  controller `towel`.
- Validate local checkpoint load, the default Demo 3 backend load path, and the
  deterministic repo harness.

## Plan

1. Sync the repo with `origin/main`.
2. Inspect the official local CoTracker README/checkpoint list.
3. Download/reconcile all CoTracker3 `.pth` files into
   `/home/xinjie/co-tracker/checkpoints`.
4. Prewarm PyTorch Hub checkpoint/code caches for the existing Demo 3 backend
   path.
5. Record exact paths, sizes, hashes, commands, and validation outcomes under
   `docs/generated/` and external dependency docs.

## Results

- PASS: `git pull --ff-only origin main` reported already up to date.
- PASS: official CoTracker3 full set is present:
  `scaled_online.pth`, `scaled_offline.pth`, `baseline_online.pth`,
  `baseline_offline.pth`.
- PASS: all four checkpoints load in `demo3-max`.
- PASS: `CoTracker3OnlineBackend(device="cuda")` loads the torch.hub-backed
  online model to `cuda:0`.
- PASS: Demo 3 dry-run.
- PASS: `scripts/harness/check_harness_catalog.py`.
- PASS: `scripts/harness/check_all.py` quick profile, including 253 unittest
  tests.
- Reports written:
  - `docs/generated/demo3_cotracker3_full_weights_validation.md`
  - `docs/generated/demo3_cotracker3_full_weights_validation.json`
