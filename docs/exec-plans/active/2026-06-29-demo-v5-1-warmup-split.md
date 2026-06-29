# Demo v5.1 Warmup Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use inline execution for this
> refactor. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split Demo v5.1 warmup code into ordinary runtime warmup and
shape-prior warmup modules without changing CLI behavior, GPU routing, output
layout, or run timing semantics.

**Architecture:** Keep protocol/data packing, observation alignment, and
sampling in `demo_v5_1/shape_prior.py`. Move shape-prior lifecycle and managed
worker startup orchestration to `demo_v5_1/shape_prior_warmup.py`. Move
ordinary runtime startup and first-frame mask helpers to
`demo_v5_1/runtime_warmup.py`, leaving `demo_v5_1/realtime_dense_track.py` as
the per-frame runtime loop owner.

**Tech Stack:** Python, argparse subprocess orchestration, unittest.

---

## First-Principles Requirement

Problem:
Warmup concerns are mixed into protocol, worker, orchestration, and runtime loop
files, making Demo v5.1 hard to explain and maintain.

Required final behavior:
Ordinary warmup and shape-prior warmup live in separate internal modules while
all public CLI flags, subprocess commands, GPU namespaces, output paths,
profile JSON fields, and start/stop timing semantics stay unchanged.

Inputs:
Existing Demo v5.1 CLI arguments, first live/fake-live frame, saved mask paths,
SAM3.1 prompts, shape-prior worker settings, FFS/remote-depth settings, table
calibration, and headless capture settings.

Outputs:
The same camera subprocess command, viewer command, shape-prior profile JSON,
headless capture metadata, chunk outputs, and process exit behavior as before.

State changes:
Add two Python modules and update imports/call sites. Remove warmup lifecycle
classes from `shape_prior.py` and startup helpers from `shape_prior_worker.py`
and `main.py`.

Invalid cases:
Keep existing fail-fast behavior for invalid CLI combinations, missing saved
masks, SAM3.1 shape mismatches, worker startup failure, and non-positive
shape-prior timeouts.

Constraints:
Do not add a new standalone CLI. Do not recreate removed legacy modules. Keep
actual run on `demo_v5_1/visualize_track.py` by default. Keep realtime on
physical GPU 0 and warmup/viewer on physical GPU 1 by default. Do not touch
unrelated local `AGENTS.md` edits.

Unknowns:
None affecting correctness; the requested split is internal and behavior
preserving.

## Minimal Correct Design

Files or modules to change:
- Create: `demo_v5_1/shape_prior_warmup.py`
- Create: `demo_v5_1/runtime_warmup.py`
- Modify: `demo_v5_1/shape_prior.py`
- Modify: `demo_v5_1/shape_prior_worker.py`
- Modify: `demo_v5_1/main.py`
- Modify: `demo_v5_1/realtime_dense_track.py`
- Modify: `tests/test_demo_v5_1_shape_prior_simplification.py`
- Modify: `tests/test_demo_v5_1_default_config.py`
- Add or modify focused tests for module ownership and behavior preservation.

Core logic change:
Move code by ownership, then update callers to import the moved functions and
classes. Keep thin `main.py` wrappers only where existing tests import the
function names.

Error handling:
Preserve existing exceptions and return codes. The refactor must not introduce
fallback, degraded mode, or compatibility modules.

Data flow:
`main.py` resolves CLI and starts managed worker through
`shape_prior_warmup.py`; `shape_prior_worker.py` prepares startup through
`shape_prior_warmup.py`; `realtime_dense_track.py` prepares ordinary runtime
startup and first-frame masks through `runtime_warmup.py`; per-frame
segmentation, PCD, tracking, and rendering remain local to
`realtime_dense_track.py`.

Why this is sufficient:
The split matches the two warmup domains requested by the user and removes the
misplaced lifecycle code without adding new public surfaces or changing runtime
behavior.

---

### Task 1: Shape-Prior Warmup Module

**Files:**
- Create: `demo_v5_1/shape_prior_warmup.py`
- Modify: `demo_v5_1/shape_prior.py`
- Modify: `demo_v5_1/shape_prior_worker.py`
- Modify: `demo_v5_1/main.py`
- Test: `tests/test_demo_v5_1_shape_prior_simplification.py`

- [x] **Step 1: Move shape-prior client/manager/profile ownership**
- [x] **Step 2: Move worker startup preload helper**
- [x] **Step 3: Move managed worker command/env/start helpers from `main.py`**
- [x] **Step 4: Update call sites and focused tests**

### Task 2: Ordinary Runtime Warmup Module

**Files:**
- Create: `demo_v5_1/runtime_warmup.py`
- Modify: `demo_v5_1/realtime_dense_track.py`
- Test: `tests/test_demo_v5_1_shape_prior_simplification.py`

- [x] **Step 1: Move first-frame mask bundle and SAM3.1 helpers**
- [x] **Step 2: Move ordinary startup preparation into runtime helpers**
- [x] **Step 3: Add `prepare_segmentation_warmup(demo)` for `_seg_worker`**
- [x] **Step 4: Keep per-frame runtime loops in `realtime_dense_track.py`**

### Task 3: Behavior Preservation Tests

**Files:**
- Modify: `tests/test_demo_v5_1_shape_prior_simplification.py`
- Modify: `tests/test_demo_v5_1_default_config.py`

- [x] **Step 1: Assert module ownership for new warmup modules**
- [x] **Step 2: Assert worker startup preload behavior with a fake worker**
- [x] **Step 3: Assert warmup manager pending/ready/failed transitions**
- [x] **Step 4: Assert dry-run/default contracts remain unchanged**

### Task 4: Validation

- [x] **Step 1: Run focused unittests**

  ```bash
  conda run -n demo_2_max --no-capture-output python -m unittest \
    tests.test_demo_v5_1_shape_prior_simplification \
    tests.test_demo_v5_1_default_config \
    tests.test_demo_v5_legacy_key_cleanup
  ```

- [x] **Step 2: Run compileall**

  ```bash
  conda run -n demo_2_max --no-capture-output python -m compileall -q \
    demo_v5_1 tests
  ```

- [x] **Step 3: Run smoke validation**

  ```bash
  conda run -n demo_2_max --no-capture-output python \
    scripts/harness/validation/run.py --profile smoke
  ```

## Notes

- `git pull --ff-only origin main` was attempted before edits and failed
  because `single-camera` cannot fast-forward to `origin/main`.
- `git pull --ff-only origin single-camera` succeeded before edits.

## Validation Results

- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_1_shape_prior_simplification tests.test_demo_v5_1_default_config tests.test_demo_v5_legacy_key_cleanup` passed.
- `conda run -n demo_2_max --no-capture-output python -m compileall -q demo_v5_1 tests` passed.
- `git diff --check -- demo_v5_1 tests docs/exec-plans/active/2026-06-29-demo-v5-1-warmup-split.md` passed.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke` passed.
