# Demo v5.1 Teachme Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use inline execution for this
> documentation task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a static teachme page that explains Demo v5.1 warmup and actual
run pipelines, including separate shape-prior warmup, other main warmups,
two-GPU routing, and the current visualizer actual-run branch.

**Architecture:** Add a standalone static page under `teachme/` so it can be
opened directly in a browser without a dev server. Keep runtime code untouched;
the page derives its explanation from `demo_v5_1/main.py`,
`demo_v5_1/main_data_processing.py`, `demo_v5_1/shape_prior.py`,
`demo_v5_1/shape_prior_worker.py`, and `demo_v5_1/visualize_track.py`.

**Tech Stack:** HTML, CSS, vanilla JavaScript.

---

### Task 1: Static Teachme Page

**Files:**
- Create: `teachme/demo-v5-1-pipeline/index.html`
- Create: `teachme/demo-v5-1-pipeline/styles.css`
- Create: `teachme/demo-v5-1-pipeline/app.js`

- [x] **Step 1: Create the content shell**

  Add a Chinese HTML page with sections for:

  - Current defaults and vocabulary.
  - Warmup overview.
  - Two-GPU swimlane flow for shape-prior worker warmup.
  - Two-GPU swimlane flow for main camera/tracker/data-process warmup.
  - Actual run flow where the current downstream branch is
    `demo_v5_1/visualize_track.py`, not `realtime_phystwin`.
  - Optional note for continuous realtime_phystwin optimization.

- [x] **Step 2: Add responsive visual styling**

  Add CSS for a dense technical explainer: sticky navigation, segmented
  controls, GPU swimlanes, connected flow nodes, compact evidence cards, and
  readable mobile layout. Avoid changing any runtime CSS because this is a
  standalone teachme artifact.

- [x] **Step 3: Add interactive mode toggles**

  Add JavaScript that lets the reader switch between:

  - `overview`
  - `shape`
  - `runtime`
  - `actual`

  The script filters and highlights the relevant nodes and updates the detail
  panel without requiring any network or build step.

- [x] **Step 4: Validate static assets**

  Run:

  ```bash
  git diff --check -- teachme/demo-v5-1-pipeline docs/exec-plans/active/2026-06-28-demo-v5-1-teachme-pipeline.md
  node --check teachme/demo-v5-1-pipeline/app.js
  ```

  Expected: `git diff --check` passes and Node reports no JavaScript syntax
  errors.

- [x] **Step 5: Scope verification**

  Run:

  ```bash
  git status --short
  ```

  Expected: the new teachme files and this plan are from this task; existing
  unrelated user edits remain untouched.

## Validation Results

- `git diff --check -- teachme/demo-v5-1-pipeline docs/exec-plans/active/2026-06-28-demo-v5-1-teachme-pipeline.md` passed.
- `node --check teachme/demo-v5-1-pipeline/app.js` passed after replacing
  nullish coalescing with older-Node-compatible fallback syntax.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke` passed.
- `git status --short` showed this task's new teachme files and plan, plus
  pre-existing unrelated modifications/deletions that were not touched.

## Notes

- `git pull --ff-only origin main` was attempted before edits but failed because
  `single-camera` and `origin/main` are not in a fast-forward relationship.
- Existing local modifications in `data_process_origin/align.py` and
  `data_process_origin/data_process_sample.py` predate this task and are not in
  scope.
- During this task, the old tracked files under `docs/exec-plans/active/` were
  already absent from the working tree. This plan file was added without
  restoring those unrelated deletions.
