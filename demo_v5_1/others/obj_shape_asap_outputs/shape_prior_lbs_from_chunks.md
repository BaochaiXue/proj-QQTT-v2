# Demo v5.1 Shape-Prior LBS From Chunks

## Inputs

- outputs root: `outputs`
- case: `outputs/shape_prior_case/shape_prior_frame0`
- shape mesh: `outputs/shape_prior_case/shape_prior_frame0/shape/matching/final_mesh.glb`
- chunk count: 23
- query schema hash: `3bef17666c124174ac56e127b88293cc1305e5e376752c4cb7d7a2cae4ea0182`

## LBS Diagnostic

- frames: 805
- chunk object tracking points: 2,001
- mesh vertices: 4,319
- mesh triangles: 6,614
- surface points: 540
- interior points: 1,124
- control K: 8
- frame stride: 1

## Tracking Quality

- object visibility ratio: 90.26%
- object motion-valid ratio: 84.54%
- chunk track status counts: `{'normal': 1, 'degraded': 22}`

The published chunk object points are used as the LBS controls. The
derived mesh/surface/interior trajectories are diagnostics only and do
not overwrite `outputs/data/final_data.pkl`.
