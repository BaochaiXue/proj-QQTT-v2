# Single Demo V3.x Copy Conversion

## Objective

Turn the copied Demo 3.x folders into explicit single-camera demo entrypoints on
the `single-camera` branch, with one RealSense camera as the default and no
three-camera-only batch, fusion, calibration, or dual-GPU requirements in the
public single-demo contract.

## Scope

- Rename the copied `demo_v3* copy/` folders into `single_demo_v3*` folders.
- Add a shared single-camera Demo 3.x runtime wrapper for contract, validation,
  and live delegation to the existing single-camera RGB-D/FFS point-cloud demo.
- Replace copied three-view entrypoints with single-camera entrypoints.
- Update README/help coverage so the single-camera branch points at these
  single demo folders rather than the copied three-view names.
- Add deterministic smoke tests for single-camera defaults and dry-run contract
  output.

## Non-Goals

- Rewrite the historical three-camera Demo 3.x runtime internals.
- Preserve three-camera batch-view tracker execution in the single demo path.
- Make hardware validation automatic.

## Validation Plan

- Focused unit tests for the new single Demo 3.x runtime contracts.
- Help-script coverage through the harness list.
- Default `scripts/harness/check_all.py` after the branch surface is internally
  consistent.
- Full `scripts/harness/check_all.py --full` because the branch-level demo,
  docs, and harness surfaces changed together.

## Status

- 2026-06-12: Started on branch `single-camera` after confirming
  `git pull --ff-only origin main` reports up to date.
- 2026-06-12: Added `single_demo_v3*` entrypoints, shared single-camera Demo 3.x
  runtime contracts, README/workflow references, and deterministic tests. Quick
  and full harness checks passed.
