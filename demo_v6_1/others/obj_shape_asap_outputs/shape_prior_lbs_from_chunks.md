# Demo v6.1 Shape-Prior LBS From Chunks

## Inputs

- outputs root: `outputs/demo_v6_1_align_render_opt_20260702_202809`
- case: `outputs/demo_v6_1_align_render_opt_20260702_202809/shape_prior_case/shape_prior_frame0`
- shape mesh: `outputs/demo_v6_1_align_render_opt_20260702_202809/shape_prior_case/shape_prior_frame0/shape/matching/final_mesh.glb`
- chunk count: 23
- query schema hash: `4b4ac66e2efeee2e6c180fab2177417d7ef0addfdc819855a8bc9737e33ee44a`

## LBS Diagnostic

- frames: 805
- chunk object tracking points: 1,975
- mesh vertices: 4,471
- mesh triangles: 6,735
- surface points: 513
- interior points: 1,180
- control K: 8
- frame stride: 1

## Tracking Quality

- object visibility ratio: 90.28%
- object motion-valid ratio: 84.58%
- chunk track status counts: `{'normal': 1, 'degraded': 22}`

The published chunk object points are used as the LBS controls. The
derived mesh/surface/interior trajectories are diagnostics only and do
not overwrite `outputs_v6_1/data/final_data.pkl`.
