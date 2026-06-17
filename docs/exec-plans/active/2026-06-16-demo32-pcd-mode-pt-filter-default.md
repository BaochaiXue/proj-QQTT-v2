# Demo 3.2 PCD Mode PT Filter Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Demo 3.2/3.3 `pcd` visual mode use `pt-filter` for both object and controller by default.

**Architecture:** Keep the full fake-live/live pipeline unchanged: FFS, EdgeTAM, TAPNext++, sync PCD filter, same-seq pairing, and calibrated table-world PCD still run. Only the visual-mode filter policy changes: `pcd` requires object/controller `pt-filter`, while `tracking` keeps `enhanced-pt`.

**Tech Stack:** Python argparse runtime policy, existing Demo 3.x wrapper tests, harness smoke validation.

---

### Task 1: Change Demo 3.2/3.3 Visual Filter Policy

**Files:**
- Modify: `qqtt/demo/single_demo_v3_runtime.py`
- Test: `tests/test_single_demo_v3_runtime.py`

- [x] In `apply_preset_defaults`, default `--demo-visual-mode pcd` object/controller filters to `pt-filter`.
- [x] Keep `--demo-visual-mode tracking` object/controller filters defaulting to `enhanced-pt`.
- [x] In `validate_args`, require `pcd` visual mode to use `pt-filter` for both object and controller.
- [x] In `validate_args`, keep `tracking` visual mode requiring `enhanced-pt` for both object and controller.

### Task 2: Verify Contract And Delegate Args

**Files:**
- Test: `tests/test_single_demo_v3_runtime.py`

- [x] Update the `pcd` visual-mode contract test to expect `pt-filter`.
- [x] Add a rejection test for `pcd + --object-filter enhanced-pt`.
- [x] Keep the `tracking` visual-mode test expecting `enhanced-pt`.

### Task 3: Run Validation

**Files:**
- Test: `tests/test_single_demo_v3_runtime.py`

- [x] Run focused runtime tests.
- [x] Run Demo 3.2 dry-run for `pcd` mode and confirm object/controller filters are `pt-filter`.
- [x] Run smoke validation profile.
