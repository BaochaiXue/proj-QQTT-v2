# 2026-04-09 Repo Hardening Pass

## Goal

Harden the current repo around the real maintenance and operator risks that still exist in the current preview / calibrate / record / align / compare stack, without expanding the repo boundary.

## Non-Goals

- Add a new downstream product.
- Reintroduce segmentation, tracking, shape priors, Gaussian, simulation, evaluation, or teleop.
- Rewrite the visualization stack from scratch.
- Change user-facing CLI behavior casually when a compatibility-preserving cleanup is possible.

## Current Repo Inventory

Current inventory is recorded in:

- `docs/generated/repo_hardening_inventory.md`

## Exact Risks Being Fixed

1. Visualization logic is still concentrated in a few large workflow modules.
2. AGENTS/docs/file-map can drift behind the actual compare stack.
3. `calibrate.pkl` semantics are narrow and useful, but the contract and frame semantics need to be more explicit and validated.
4. Record-time preflight policy exists, but the decision table and operator summary were implicit and partially duplicated in `record_data.py`.
5. ROI/object-first compare depends on multiple heuristics and needs stronger contract-level tests and more systematic debugability.
6. Comparison observability artifacts exist, but their structure and meaning need clearer contracts and docs.
7. Some existing tests are still mostly smoke-level around critical contracts.

## Target Architecture / Contract Changes

- Keep harness CLIs thin.
- Keep workflow modules orchestration-first.
- Keep case IO, artifact writing, crop math, orbit/view math, rendering, and layouts in shared modules.
- Make record preflight a reusable decision-table module instead of inline logic.
- Make calibration world semantics explicit in compare metadata and docs.
- Strengthen `calibration_io.py` validation:
  - duplicate serial checks
  - bottom-row / finite-value validation
  - mapping-mode summary
- Strengthen object-first compare contract:
  - explicit pass1/pass2 ROI semantics in docs/tests
  - object/context/source alignment checks in tests

## Migration Strategy

1. Inventory current repo state and document it.
2. Introduce explicit preflight and calibration-frame helper modules.
3. Rewire current record/compare code to use those helpers without changing main CLIs.
4. Add contract-level tests for new hardening paths.
5. Update docs only where stale.
6. Run deterministic validation with `scripts/harness/check_all.py`.

## Validation Plan

Run:

- targeted new tests for:
  - record preflight policy
  - calibration contract hardening
  - object-compare contract
  - turntable frame-contract metadata
- `python scripts/harness/check_visual_architecture.py`
- `python scripts/harness/check_all.py`

Record outcomes in:

- `docs/generated/repo_hardening_validation.md`

## Risks

1. Breaking existing compare tests that read old metadata paths or output files.
2. Overstating semantic-world support when the current compare still uses calibration-world geometry.
3. Making preflight stricter than current repo policy by accident.
4. Adding docs that drift again if not tied to actual module boundaries.

## Acceptance Criteria

1. Repo scope stays unchanged.
2. Visualization layering stays cleaner and more explicit.
3. Calibration semantics are explicit and validated.
4. Record-time preflight policy is explicit, operator-visible, and tested.
5. ROI/object-first contracts are easier to inspect and test.
6. Comparison observability artifacts are documented more clearly.
7. Docs are updated where stale.
8. Deterministic checks pass.

## Completion Checklist

- [ ] Write current repo inventory
- [ ] Add explicit record preflight contract module
- [ ] Harden calibration loader validation
- [ ] Expose calibration/frame semantics in compare metadata
- [ ] Add contract-level tests
- [ ] Update AGENTS/docs where stale
- [ ] Add repo hardening validation note
- [ ] Run `python scripts/harness/check_all.py`
