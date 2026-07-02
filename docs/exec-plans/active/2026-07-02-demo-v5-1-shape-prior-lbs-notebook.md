# Demo v5.1 Shape-Prior LBS Notebook

## Requirement

Use `demo_v5_1/others/obj_shape_asap.ipynb` on the current Demo v5.1
`outputs/` shape-prior artifacts, then drive a shape-prior mesh/point display
from chunk tracking `object_points`.

## Current Problem

The notebook still points at the old PhysTwin path
`/home/shen/PhysTwin/data/different_types/...` and includes cells that rewrite
the source `final_data.pkl`. The current Demo v5.1 outputs are instead:

- `outputs/shape_prior_case/shape_prior_frame0/final_data.pkl`
- `outputs/shape_prior_case/shape_prior_frame0/shape/matching/final_mesh.glb`
- `outputs/data/final_data.pkl`
- `outputs/online_data/chunks/chunk_*.pkl`

## Required Behavior

- Resolve paths relative to this checkout and fail fast if required outputs are
  missing.
- Load shape-prior mesh/surface/interior data from the warmup case.
- Load tracking object points from `outputs/online_data/chunks/chunk_*.pkl`.
- Build mesh deformation and embedded surface/interior/object trajectories from
  the chunk tracking object points.
- Write derived LBS/ASAP diagnostics and visualization files under
  `demo_v5_1/others/obj_shape_asap_outputs/`, without mutating the published
  Demo v5.1 `final_data.pkl` or adding artifacts to `outputs/`.

## Inputs

- Demo v5.1 `outputs/` directory.
- Shape-prior warmup case name `shape_prior_frame0`.
- Online chunk files under `outputs/online_data/chunks`.

## Outputs

- Updated notebook cells pointing at the correct Demo v5.1 files.
- A derived diagnostic pickle containing mesh/object/surface/interior
  trajectories.
- A notebook-visible MP4 preview and contact sheet.
- A refreshed shape-prior inspection report.

## Validation

- [x] Run the shape-prior inspection script with `--no-view`.
- [x] Execute the notebook headlessly far enough to build and save the
  diagnostic trajectories.
- [x] Inspect the saved diagnostic shapes.
- [x] Run the smoke validation profile:
  `conda run -n demo_2_max --no-capture-output python`
  `scripts/harness/validation/run.py --profile smoke`.

## Outcome

- Shape-prior inspection report:
  `demo_v5_1/others/obj_shape_asap_outputs/shape_prior_outputs_inspection.md`
- LBS diagnostic:
  `demo_v5_1/others/obj_shape_asap_outputs/shape_prior_lbs_from_chunks.pkl`
- LBS report:
  `demo_v5_1/others/obj_shape_asap_outputs/shape_prior_lbs_from_chunks.md`
- LBS preview:
  `demo_v5_1/others/obj_shape_asap_outputs/shape_prior_lbs_preview.mp4`
- LBS contact sheet:
  `demo_v5_1/others/obj_shape_asap_outputs/shape_prior_lbs_preview_sheet.png`
- LBS summary:
  805 frames, 2,001 object tracking controls, 4,319 mesh vertices,
  540 surface points, and 1,124 interior points.
