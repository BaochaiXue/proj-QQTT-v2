# Demo v6.2 Shape Prior Outputs Inspection

## Paths

- case: `outputs_v6_1/shape_prior_case/shape_prior_frame0`
- warmup final data: `outputs_v6_1/shape_prior_case/shape_prior_frame0/final_data.pkl`
- published final data: `outputs_v6_1/data/final_data.pkl`

## Warmup Case

- input source: `fake-live`
- depth backend: `native-realsense`
- depth source internal: `realsense`
- object label: `stuffed animal`
- controller label: `hand`

## Masked PCD Stats

processed object mask:
  count: 19,957
  bbox min: [-0.0236, -0.1068, -0.0786]
  bbox max: [0.3491, 0.2507, -0.0025]
  mean: [0.1726, 0.0799, -0.0416]
  std: [0.0718, 0.0945, 0.0169]

processed controller mask:
  count: 5,231
  bbox min: [-0.0904, 0.1637, -0.0655]
  bbox max: [0.4021, 0.2989, 0.0106]
  mean: [0.1552, 0.2274, -0.0353]
  std: [0.2051, 0.0337, 0.0173]

object/controller overlap pixels: 0

The full depth-valid frame includes background and unrelated pixels:

depth-valid frame:
  count: 357,327
  bbox min: [-1.1335, -0.2717, -0.3078]
  bbox max: [1.5183, 1.1098, 0.8530]
  mean: [0.1420, 0.1570, 0.0444]
  std: [0.4577, 0.3033, 0.1646]

## Shape Prior Supplement

- warmup object points: 3,601
- warmup controller points: 5,231
- surface supplement points: 484
- interior supplement points: 1,371
- total supplement points: 1,855

The supplement is stored separately as `surface_points` and
`interior_points`; it is not appended to `object_points`.

## Published Final Data

- object points shape: `(560, 1976, 3)`
- controller points shape: `(560, 30, 3)`
- surface points: 484
- interior points: 1,371
- semantic label counts: `{0: 491, 1: 3610, 2: 899}`
- object visibility ratio: 95.83%
- object motion-valid ratio: 92.97%
- controller proxied ratio: 13.08%
- track process status: `degraded`
