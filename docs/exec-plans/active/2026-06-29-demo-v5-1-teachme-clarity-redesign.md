# Demo v5.1 Teachme Clarity Redesign Plan

> **For agentic workers:** REQUIRED SUB-SKILL:
> `first-principles-readable-code`. Keep this as a static teachme-only change;
> do not modify Demo v5.1 runtime behavior.

## Requirement

Problem: the current teachme flowchart reads like three long node lists and
still does not provide the requested mind-map view. It does not clearly show
the conceptual branches, stage order, cross-GPU ownership, handoff points,
inputs, outputs, or wait conditions.

Required final behavior: rebuild the teachme page so the reader can separately
understand:

- a top-level mind map for Demo v5.1 pipeline ownership and data flow;
- shape-prior warmup on GPU 1 and its CPU/GPU 0 handoff;
- ordinary runtime warmup on GPU 0 with CPU/IO service setup;
- actual run where the other side is `visualize_track.py`, not
  `realtime_phystwin` by default.

Inputs: existing Demo v5.1 code and the static files under
`teachme/demo-v5-1-pipeline/`.

Outputs: clearer static HTML, CSS, and JavaScript. No runtime artifact, CLI,
GPU routing, or Python behavior changes.

State changes: only update teachme documentation assets and this execution
plan.

Invalid cases: if the page cannot represent a stage unambiguously from the code,
do not invent a stage; keep the diagram tied to code-backed facts.

Constraints:

- Keep this browser-openable with plain HTML/CSS/JavaScript.
- Split shape-prior warmup, ordinary warmup, and actual run.
- Show GPU 0 and GPU 1 separately.
- Update the code source map after the warmup split.
- Do not add compatibility paths or new runtime logic.

Unknowns: none that block implementation.

## Minimal Design

Files or modules to change:

- `teachme/demo-v5-1-pipeline/index.html`
- `teachme/demo-v5-1-pipeline/styles.css`
- `teachme/demo-v5-1-pipeline/app.js`
- this plan file

Core logic change: replace the vague generated lane list with explicit
mind-map overview plus stage-numbered swimlane boards. Each stage node should
name owner, GPU/process, input, output, and wait/blocking condition where
relevant.

Error handling: no runtime error handling changes. JavaScript should fail closed
by doing nothing if optional DOM elements are absent.

Data flow:

1. Static HTML defines the mind map, three diagrams, and source map.
2. CSS makes rows, lanes, arrows, and nodes visually crisp.
3. JavaScript only switches focus modes and expands/collapses stage details.

Why this is sufficient: the user complaint is about clarity of the teachme
diagram, not runtime behavior. A static, explicit diagram directly fixes the
reading problem without touching Demo v5.1 code.

## Tasks

- [x] Add a top-level mind map before the detailed flow diagrams.
- [x] Replace the generated three-column flow with explicit stage swimlanes.
- [x] Add per-stage input/output/wait facts for shape prior, ordinary warmup,
      and actual run.
- [x] Update side-panel summaries and source map to the new split modules.
- [x] Simplify JavaScript to focus static diagrams instead of generating them.
- [x] Validate syntax, static structure, and repo smoke profile.

## Validation

Run:

```bash
git diff --check -- teachme/demo-v5-1-pipeline docs/exec-plans/active/2026-06-29-demo-v5-1-teachme-clarity-redesign.md
node --check teachme/demo-v5-1-pipeline/app.js
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke
```

Also render the static page in a browser and inspect a screenshot for overlap,
blank sections, and readability across at least one desktop viewport.

## Validation Results

- `xmllint --html --noout teachme/demo-v5-1-pipeline/index.html` passed.
- `node --check teachme/demo-v5-1-pipeline/app.js` passed.
- `git diff --check -- teachme/demo-v5-1-pipeline docs/exec-plans/active/2026-06-29-demo-v5-1-teachme-clarity-redesign.md` passed.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke` passed.
- Browser screenshot succeeded:
  `/tmp/demo-v5-1-teachme-mindmap.png`.
