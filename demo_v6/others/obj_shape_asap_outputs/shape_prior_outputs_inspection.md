# Demo v6 Shape Prior Outputs Inspection

## Paths

- case: `/home/xinjie/single_proj_qqtt/outputs/shape_prior_case/shape_prior_frame0`
- warmup final data: `/home/xinjie/single_proj_qqtt/outputs/shape_prior_case/shape_prior_frame0/final_data.pkl`
- published final data: `/home/xinjie/single_proj_qqtt/outputs/data/final_data.pkl`

## Warmup Case

- input source: `fake-live`
- depth backend: `native-realsense`
- depth source internal: `realsense`
- object label: `stuffed animal`
- controller label: `hand`

## Masked PCD Stats

processed object mask:
  count: 19,049
  bbox min: [0.0346, -0.0042, -0.0798]
  bbox max: [0.3921, 0.4501, -0.0003]
  mean: [0.1805, 0.2204, -0.0430]
  std: [0.0878, 0.0860, 0.0188]

processed controller mask:
  count: 4,722
  bbox min: [0.0960, -0.0542, -0.1091]
  bbox max: [0.2001, 0.4941, 0.0010]
  mean: [0.1482, 0.2132, -0.0622]
  std: [0.0221, 0.2311, 0.0257]

object/controller overlap pixels: 1

The full depth-valid frame includes background and unrelated pixels:

depth-valid frame:
  count: 352,341
  bbox min: [-0.2364, -15.1548, -0.3253]
  bbox max: [11.2190, 1.5079, 13.3737]
  mean: [0.1805, 0.2239, 0.0300]
  std: [0.3869, 0.5743, 0.3599]

## Shape Prior Supplement

- warmup object points: 3,623
- warmup controller points: 4,722
- surface supplement points: 540
- interior supplement points: 1,124
- total supplement points: 1,664

The supplement is stored separately as `surface_points` and
`interior_points`; it is not appended to `object_points`.

## Published Final Data

- object points shape: `(805, 2001, 3)`
- controller points shape: `(805, 30, 3)`
- surface points: 540
- interior points: 1,124
- semantic label counts: `{0: 515, 1: 3591, 2: 894}`
- object visibility ratio: 90.26%
- object motion-valid ratio: 84.54%
- controller proxied ratio: 11.61%
- track process status: `degraded`
