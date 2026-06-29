# Demo v5.1 Warmup Simplification Plan

**Goal:** Replace the old defensive shape-prior warmup orchestration with the
current single-camera Demo v5.1 flow: `main_warmup` and
`main_data_processing` on GPU namespace 0, `shape_prior_warmup` and
`visualizer` on GPU namespace 1, and one local shape-prior pipeline that writes
the agreed artifacts.

## Requirement

Problem:
The Demo v5.1 warmup path carries too many mode/config branches for a demo that
always uses GPU 0 for main warmup/realtime data processing and GPU 1 for
shape-prior warmup/visualization.

Required final behavior:
Defaults come from `demo_v5_1/config/default.yaml`; CLI only keeps the small
set of useful overrides. Shape-prior warmup accepts the first prepared RGB/mask
request, writes a single-camera shape-prior case, runs
`shape_prior_generate.py -> shape_prior_align.py -> shape_prior_sample.py`, and
returns `ShapePriorResult` for the realtime capture metadata/chunk writer.

Inputs:
Demo v5.1 CLI args, default config, first valid frame-0 RGB/object mask,
controller mask, depth, camera intrinsics, and camera-to-world transform.

Outputs:
Realtime capture metadata with `shape_prior_status=ready`, `shape_prior/points.npz`
containing display points plus surface/interior points, and simplified dry-run
GPU routing summary.

State changes:
Rewrite `demo_v5_1/shape_prior_warmup.py`, simplify Demo v5.1 GPU/shape-prior
orchestration in `main.py`, remove the Demo v5.1 realtime PhysTwin launch path,
and update focused tests/docs for the new flow.

Invalid cases:
Invalid frame-0 arrays, empty object mask, missing generated/aligned/sample
artifacts, and failed subprocess stages raise explicit errors and mark the
manager failed.

Constraints:
Stay on `single-camera`; do not resurrect deleted `shape_prior.py` or
`shape_prior_worker.py`; keep external SAM3D dependencies outside this repo.

Unknowns:
None blocking implementation. `git pull --ff-only origin main` was attempted
before edits and failed because this branch cannot fast-forward to
`origin/main`.

## Design

Files or modules to change:
- `demo_v5_1/shape_prior_warmup.py`
- `demo_v5_1/main.py`
- `demo_v5_1/config/default.yaml`
- `demo_v5_1/pipeline.md`
- Focused Demo v5.1 tests

Core logic change:
Move shape-prior warmup from managed remote-worker orchestration to a local
one-shot pipeline manager. Replace GPU modes with direct default namespaces:
`main_data_processing=0`, `shape_prior_warmup=1`, `visualizer=1`, with
explicit override flags.

Error handling:
Fail fast for invalid frame/case data and failed stage commands. The manager
records `failed` status and the error string in the profile.

Data flow:
`main_data_processing.py` builds `ShapePriorFrame0Request`; the manager writes
the shape-prior case under the capture dir, runs generate/align/sample, reads
`final_data.pkl`, and returns `ShapePriorResult`.

Why this is sufficient:
It matches the current single-camera demo flow and removes the unused
worker-mode/mode-mapping configuration without adding fallback paths.

## Tasks

- [x] Rewrite `shape_prior_warmup.py` around the new single-camera case flow.
- [x] Simplify `main.py` GPU and shape-prior worker configuration.
- [x] Remove Demo v5.1 realtime PhysTwin/optimization launch plumbing.
- [x] Rename GPU keys/CLI/summary fields to main data processing,
      shape-prior warmup, and visualizer.
- [x] Update default config and pipeline docs.
- [x] Update focused tests for the new flow.
- [x] Run focused validation and smoke checks where practical.
