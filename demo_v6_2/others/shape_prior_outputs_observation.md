# Demo v6.2 Shape Prior Outputs 观察记录

## 本轮输入

本轮结果来自 `outputs_v6_1/shape_prior_case/shape_prior_frame0`：

- input source: `fake-live`
- depth backend: `native-realsense`
- object: `stuffed animal`
- controller: `hand`
- camera count: 1
- RGB-D grid: `1 x 480 x 848`

`pcd/0.npz` 里的 dense grid 是整张图。判断 object/controller PCD 质量时，
应看 processed mask 内的点，而不是用全图的坐标范围。

## Warmup PCD

processed object mask：

- count: 19,957
- bbox min: `[-0.0236, -0.1068, -0.0786]`
- bbox max: `[0.3491, 0.2507, -0.0025]`
- mean: `[0.1726, 0.0799, -0.0416]`

processed controller mask：

- count: 5,231
- bbox min: `[-0.0904, 0.1637, -0.0655]`
- bbox max: `[0.4021, 0.2989, 0.0106]`
- mean: `[0.1552, 0.2274, -0.0353]`

object/controller overlap 是 0 pixel。整张图共有 357,327 个 depth-valid
pixel，其中包含背景和无关像素。

## Shape-Prior 补点

- warmup object points: 3,601
- warmup controller points: 5,231
- surface points: 484
- interior points: 1,371
- total supplement points: 1,855

补点保存在：

- `outputs_v6_1/shape_prior/points.npz`
- `outputs_v6_1/capture/shape_prior/points.npz`

warmup `final_data.pkl` 仍把 `surface_points` 和 `interior_points` 作为独立
字段保留；它们不会直接 append 到 `object_points` 尾部。

## 最终发布结果

`outputs_v6_1/data/final_data.pkl` 当前状态：

- object points: `(560, 1976, 3)`
- controller points: `(560, 30, 3)`
- surface points: `(484, 3)`
- interior points: `(1371, 3)`
- query schema version: `data_process_sam3d_realtime_query_schema_v1`
- query schema hash:
  `fe58efb61da1da1c8dddb5567860324f6df79634452fffc9030568d27c52c0c4`
- track process status: `degraded`

Semantic query label counts：

- `0` none: 491
- `1` object: 3,610
- `2` controller: 899

运行质量摘要：

- object visibility: 95.83%
- object motion-valid: 92.97%
- controller proxied: 13.08%
- chunks: 16（1 normal，15 degraded）

## 更新后的可视化

- object/controller tracking：560 帧，5 FPS
- online RealSense-style depth：560 帧，5 FPS
- static shape-prior orbit：90 帧，5 FPS；contact sheet 为 12 个视角

结果写在 `demo_v6_2/others/obj_shape_asap_outputs/`。tracking 会只绘制同时
满足 visible 和 motion-valid 的 object points。当前结果可以用于检查本轮
shape-prior-augmented 单相机输出，但 `degraded` 状态仍说明部分 frame 使用了
controller proxy 或包含 object motion-invalid 样本。

tracking 的最后一帧只显示 controller：该帧 1,976 个 object points 的
`motion-valid` 全部为 false，所以 renderer 按上述规则没有绘制 object。这是
上游 tracking mask 的真实结果，不是 MP4 编码损坏。
