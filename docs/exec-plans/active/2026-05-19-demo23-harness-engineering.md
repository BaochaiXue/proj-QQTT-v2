# Demo 2.3 Harness Engineering Hardening

## Goal

Make the current Demo 2.3 FPS / fused-PCD debugging loop agent-legible and mechanically checkable.

The repo already has a harness catalog and generated evidence policy. This plan adds a problem-specific
failure packet so future Codex runs can diagnose Demo 2.3 from repository-local artifacts instead of
reconstructing context from chat history.

## Scope

- Add a harness engineering source-of-truth doc that maps agent knowledge, harness commands, generated evidence, and escalation rules.
- Add a Demo 2.3 failure-packet summarizer for profile JSON, runtime summaries, calibration reports, and calibration preflight reports.
- Add a mechanical guard that keeps the harness engineering map wired into `AGENTS.md`, `scripts/harness/README.md`, `_catalog.py`, and `check_all.py`.
- Register the new harness scripts and cover them with deterministic smoke tests.

## Non-Goals

- No Demo 2.3 runtime behavior changes in this slice.
- No RealSense hardware run in this slice.
- No cleanup of unrelated dirty worktree changes.

## Validation

Completed on 2026-05-19:

```bash
conda run -n demo_2_max --no-capture-output python -m py_compile \
  scripts/harness/check_harness_engineering.py \
  scripts/harness/summarize_demo23_failure_packet.py

conda run -n demo_2_max --no-capture-output python -m unittest -v \
  tests.test_demo23_harness_engineering_smoke \
  tests.test_check_all_smoke

conda run -n demo_2_max --no-capture-output python scripts/harness/check_harness_engineering.py
conda run -n demo_2_max --no-capture-output python scripts/harness/check_harness_catalog.py
conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py
git diff --check
```

Results:

- `py_compile`: PASS
- targeted unittest smoke (`tests.test_demo23_harness_engineering_smoke`, `tests.test_check_all_smoke`): PASS
- `scripts/harness/check_harness_engineering.py`: PASS
- `scripts/harness/check_harness_catalog.py`: PASS
- `scripts/harness/check_all.py`: PASS, 260 unit tests in the quick profile
- `git diff --check`: PASS

Generated local evidence:

- `docs/generated/demo23_failure_packet.json`
- `docs/generated/demo23_failure_packet.md`

The current packet correctly starts from the latest local Demo 2.3 profile and
flags that the available profile does not prove the required FFS batch=3
builderOptimizationLevel=5 contract.
