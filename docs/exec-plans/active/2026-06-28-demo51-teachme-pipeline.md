# Demo 5.1 Teachme Pipeline Explainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a static `teachme` HTML/CSS/JS explainer for the current Demo 5.1 pipeline, mapping it to the repository's Demo 3.3 shape-prior warmup and Demo 3.2/3.3 dual-GPU actual run code paths.

**Architecture:** Create one self-contained static page with separated files for markup, styling, and interaction data. The page shows GPU0 and GPU1 as explicit lanes, splits shape-prior warmup from ordinary warmup, and documents the actual-run visualizer branch.

**Tech Stack:** Plain HTML, CSS, and JavaScript; no build step or external runtime dependency.

---

## Files

- Create: `teachme/demo51_pipeline.html`
- Create: `teachme/demo51_pipeline.css`
- Create: `teachme/demo51_pipeline.js`

## Source Facts To Preserve

- Demo 5.1 naming in this explainer maps to the current repo's Demo 3.3 shape-prior warmup variant of Demo 3.2.
- GPU0 owns FFS TensorRT depth, SAM3.1/HF EdgeTAM masks, main-process fusion, render preparation, and Open3D visualization.
- GPU1 owns the isolated tracker child process: TAPNext++ or LiteTracker, with latest-wins IPC input/result exchange.
- Shape-prior warmup is separate from ordinary warmup. It snapshots the first valid strict-source RGB-D/mask/calibration bundle and writes a FuturePhysTwin-style case.
- The shape-prior heavy route is `image_upscale.py -> segment_util_image.py -> data_process_sam3d/shape_prior.py -> data_process/align.py -> data_process_sam3d/data_process_sample.py --shape_prior`.
- Shape prior is render-only: it attaches a gray canonical reference layer and does not alter tracker input, live fused PCD, masks, or tracking markers.
- Actual run uses strict-source group IDs: RGB, FFS depth, masks, tracker input/result, lift inputs, and render packet must match by group in the default path.
- The actual-run alternate pass should be described as a visualizer-based diagnostic/teaching route, not as launching the full `realtime_phystwin` training or physics runtime.

### Task 1: Build The Static Page Skeleton

**Files:**
- Create: `teachme/demo51_pipeline.html`

- [x] **Step 1: Add semantic sections**

Create a static HTML document with sections for:

```html
<section id="warmup"></section>
<section id="shape-prior"></section>
<section id="actual-run"></section>
<section id="visualizer-pass"></section>
```

- [x] **Step 2: Link local assets**

Reference local CSS and JS:

```html
<link rel="stylesheet" href="./demo51_pipeline.css">
<script src="./demo51_pipeline.js" defer></script>
```

### Task 2: Style Dual-GPU Flow Lanes

**Files:**
- Create: `teachme/demo51_pipeline.css`

- [x] **Step 1: Define readable layout primitives**

Add CSS classes for `gpu-lane`, `step-node`, `phase-chip`, `detail-panel`, and responsive grid behavior.

- [x] **Step 2: Keep mobile layout stable**

Use responsive grid tracks and explicit minimum sizes so flow nodes do not overlap at narrow widths.

### Task 3: Add Interactive Flow Data

**Files:**
- Create: `teachme/demo51_pipeline.js`

- [x] **Step 1: Model warmup and actual-run nodes**

Create JavaScript arrays for ordinary warmup, shape-prior warmup, actual run, and visualizer pass nodes. Each node includes:

```js
{
  id: "capture-group",
  gpu: "CPU/IO",
  phase: "actual",
  title: "三相机 capture group",
  detail: "RealSense 产生 group_id 绑定的 RGB/IR/depth 输入。"
}
```

- [x] **Step 2: Render cards and tab filtering**

Implement a small renderer that draws nodes into CPU/IO, GPU0, and GPU1 lanes and filters by phase buttons.

### Task 4: Validate The Static Files

**Files:**
- Test: `teachme/demo51_pipeline.html`
- Test: `teachme/demo51_pipeline.css`
- Test: `teachme/demo51_pipeline.js`

- [x] **Step 1: Inspect file references**

Run:

```bash
python - <<'PY'
from pathlib import Path
root = Path("teachme")
html = (root / "demo51_pipeline.html").read_text(encoding="utf-8")
assert "./demo51_pipeline.css" in html
assert "./demo51_pipeline.js" in html
for name in ["demo51_pipeline.css", "demo51_pipeline.js"]:
    assert (root / name).is_file(), name
print("teachme static references OK")
PY
```

Expected output:

```text
teachme static references OK
```

- [x] **Step 2: Check JavaScript syntax**

Run:

```bash
node --check teachme/demo51_pipeline.js
```

Expected result: exits with status `0`.

- [ ] **Step 3: Run deterministic scope checks**

Run:

```bash
conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py
```

Expected result: harness exits with status `0`, unless the unrelated missing `docs/` tree prevents the harness from running.

## Self-Review

- Spec coverage: The page covers ordinary warmup, shape-prior warmup, dual-GPU separation, actual run, visualizer alternate path, and static teachme output.
- Placeholder scan: No `TBD`, `TODO`, or unspecified implementation steps remain.
- Type consistency: HTML, CSS, and JavaScript filenames are consistent across tasks.
