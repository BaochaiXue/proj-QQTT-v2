# Demo v6.1 Shape Prior Outputs 观察记录

## Open3D 窗口

这次打开的是：

- case: `outputs_v6_1/shape_prior_case/shape_prior_frame0`
- warmup final data: `final_data.pkl`

Open3D 场景颜色：

- 白色: frame-0 object observation points, 经过 shape-prior sampling 后保留
- 红色: frame-0 controller points
- 青色: shape-prior surface 补点
- 蓝色: shape-prior interior 补点

窗口进程 PID 是 `2504389`；关闭 Open3D 窗口后 viewer 会退出。日志路径是
`/tmp/demo_v6_1_shape_prior_pcd_only_open3d.log`。

## Warmup 产物

这次 warmup case 是单相机：

- metadata: `input_source=fake-live`
- depth backend: `native-realsense`
- object: `stuffed animal`
- controller: `hand`
- camera count: 1
- RGB-D grid: `1 x 480 x 848`

`pcd/0.npz` 里的 dense grid 是整张图，不能直接用全图 min/max 判断质量。
真正应该看的，是 processed mask 里的 object/controller 点：

- depth-valid pixels: 352,341
- object mask pixels: 19,049
- controller mask pixels: 4,722
- object/controller overlap: 1 pixel

Masked object PCD 的世界坐标 bbox：

- min: `[0.0346, -0.0042, -0.0798]`
- max: `[0.3921, 0.4501, -0.0003]`
- mean: `[0.1805, 0.2204, -0.0430]`

Masked controller PCD 的世界坐标 bbox：

- min: `[0.0960, -0.0542, -0.1091]`
- max: `[0.2001, 0.4941, 0.0010]`
- mean: `[0.1482, 0.2132, -0.0622]`

全图 depth-valid bbox 很大，因为里面还有背景和无关像素。判断这次 PCD
是否靠谱，应该看上面的 masked object/controller 统计。

## 我们补了哪些点

明确的 shape-prior 补点是：

- surface points: 540
- interior points: 1,124
- total supplement points: 1,664

这些点写在两个地方：

- `outputs_v6_1/shape_prior/points.npz`
- `outputs_v6_1/capture/shape_prior/points.npz`

warmup 的 `final_data.pkl` 里把补点作为独立 PCD 字段保留：

- `surface_points`: `(540, 3)`
- `interior_points`: `(1124, 3)`

采样逻辑是 origin-style 5 mm voxel policy：

1. frame-0 观测到的 object points 先占 voxel；
2. shape-prior surface samples 填还没有被占掉的 voxel；
3. shape-prior interior samples 再填剩下还空的 voxel。

所以补点不是简单 append 到 `object_points` 尾部。判断“补了哪些点”时，
最可靠的来源是独立的 `surface_points` 和 `interior_points` 字段。

## 最终发布的 PCD 状态

发布出去的 `outputs_v6_1/data/final_data.pkl` 当前是：

- object points: `(805, 2001, 3)`
- controller points: `(805, 30, 3)`
- surface shape-prior points: `(540, 3)`
- interior shape-prior points: `(1124, 3)`
- query schema version: `data_process_sam3d_realtime_query_schema_v1`
- track process status: `degraded`

Semantic query label counts：

- `0` none: 515
- `1` object: 3,591
- `2` controller: 894

运行质量摘要：

- object visibility: 1,453,839 / 1,610,805, 约 90.26%
- object motion-valid: 1,361,852 / 1,610,805, 约 84.54%
- controller proxied: 2,804 / 24,150, 约 11.61%

结论：当前 object PCD 可以作为 shape-prior-augmented 的单相机产物使用。
frame-0 masked object cloud 是紧的，surface/interior 补点也落在同一段
object-scale bbox 内；最终 static case 也保留了这两类补点。但 tracking
不是完全 normal：`track_process_status=degraded`，805 帧里有一部分
controller proxy 和 object motion-invalid 样本。
