# Demo v5.1 Align Render Optimization

## Requirement

Problem:
Demo v5.1 shape-prior alignment spends about 16.7 seconds in the align stage.
Profiling shows the largest safe target is PyTorch3D rendering: the current
helper rasterizes once for depth and then rasterizes again through
`MeshRenderer.forward` for color, and `render_multi_images` loads the same GLB
twice for its two render batches.

Required final behavior:
Keep pose sampling, render resolution, matching settings, PnP, scale fitting,
ARAP, and output contracts unchanged while rendering each batch with one
rasterization and loading the GLB once per multi-view render.

Inputs:
Existing Demo v5.1 shape-prior case files and
`demo_v5_1/utils/align_util.py` render helpers.

Outputs:
Equivalent rendered color/depth, matching metrics, final mesh, and faster align
runtime. No CLI or data-contract changes.

State changes:
Only the internal render implementation changes. Existing dirty worktree
changes are preserved.

Invalid cases:
Existing PyTorch3D load/render errors should still fail at the same boundary;
do not add fallback or degraded behavior.

Constraints:
Stay on `single-camera`. `git pull --ff-only origin main` cannot fast-forward
because `single-camera` diverges from `origin/main`; local HEAD is aligned with
`origin/single-camera`.

Unknowns:
No correctness-blocking unknowns.

## Plan

- [x] Save and time a baseline align run from the existing full fake-live
  warmup case before editing `align_util.py`.
- [x] Refactor `render_image`/`render_multi_images` to reuse one rasterization
  per batch and one loaded GLB across split batches.
- [x] Compare baseline and optimized render/matching/mesh outputs on the same
  copied case.
- [x] Run focused unit tests.
- [x] Run a full fake-live Demo v5.1 validation pass and record warmup timing.

## Validation

Branch/setup:

- Confirmed current branch is `single-camera`.
- `git pull --ff-only origin main` failed because `single-camera` cannot
  fast-forward to `origin/main`; continued because local `HEAD` is aligned with
  `origin/single-camera`.

Baseline isolated align before editing:

- Source case:
  `outputs/demo_v51_full_fake_warmup_20260702_201143/shape_prior_case/shape_prior_frame0`
- Copied to `/tmp/demo_v51_align_baseline/shape_prior_frame0`, removed
  `shape/matching/`, then ran `demo_v5_1/shape_prior_align.py`.
- Elapsed: `19.50s`
- Match count: `79`
- Reprojection error: `8.858978271484375`
- Rescale: `0.45529852812964866`

Optimized isolated align on a fresh copy:

- Copied to `/tmp/demo_v51_align_optimized/shape_prior_frame0`, removed
  `shape/matching/`, then reran `demo_v5_1/shape_prior_align.py`.
- Elapsed: `15.25s`
- Match count: `79`
- Reprojection error: `8.858978271484375`
- Rescale: `0.45529852812964866`
- Best pose index stayed `(70, 0.0)`.
- `final_mesh.glb`: `4110` vertices and `6238` faces in both runs; faces exact,
  vertices max absolute diff `0.0`, vertices allclose within `1e-6`.
- Render arrays from `render_multi_images`: color exact, depth exact,
  intrinsics exact, sampled poses exact.

Focused tests:

- `PYTHONDONTWRITEBYTECODE=1 conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo_v5_1_shape_prior_simplification`
- Result: passed, `25` tests.

Full fake-live Demo v5.1 validation:

- Run directory:
  `outputs/demo_v51_align_render_opt_20260702_202809`
- Command used `demo_v5_1/main.py --input-source fake-live --visualizer-mode disabled`.
- `main_data_processing_return_code`: `0`
- `main_data_processing_stop_reason`: `main_data_processing_completed`
- `shape_prior_status`: `ready`
- `shape_prior_align_ms`: `12788.986266998108`
- Previous full fake-live baseline align time:
  `16731.474863976473ms`
- Full fake-live align delta: about `3942.49ms` faster.
- `chunk_count`: `23`
- Manifest status: `finished`, latest committed chunk `22`, latest committed
  frame `805`.
- `online_publish_skipped_chunk_count`: `0`
- `track_process_status_counts`: `{'degraded': 22, 'invalid': 0, 'normal': 1}`
