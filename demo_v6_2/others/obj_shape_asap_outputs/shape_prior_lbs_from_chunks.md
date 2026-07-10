# Demo v6.2 Shape-Prior Orbit Preview

## Inputs

- outputs root: `outputs_v6_1`
- case: `outputs_v6_1/shape_prior_case/shape_prior_frame0`
- shape mesh:
  `outputs_v6_1/shape_prior_case/shape_prior_frame0/shape/matching/final_mesh.glb`

## Current Shape Prior

- raw masked object PCD points: 19,957
- mesh vertices: 5,869
- mesh triangles: 8,866
- surface supplement points: 484
- interior supplement points: 1,371

## Visualization

- orbit video:
  `demo_v6_2/others/obj_shape_asap_outputs/shape_prior_lbs_preview.mp4`
- contact sheet:
  `demo_v6_2/others/obj_shape_asap_outputs/shape_prior_lbs_preview_sheet.png`
- video: 90 frames at 5 FPS, 960 x 720
- contact sheet: 12 evenly spaced views
- gray: raw masked object PCD
- cyan: shape-prior surface points
- pink: shape-prior interior points

The preview is a static one-circle orbit of the current warmup shape prior. It
does not read the online chunks and does not overwrite
`outputs_v6_1/data/final_data.pkl`. The legacy report and preview filenames are
retained so existing links keep working; this is not a dynamic LBS trajectory.
