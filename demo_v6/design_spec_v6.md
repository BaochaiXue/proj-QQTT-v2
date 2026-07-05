# Demo v6 Design Spec — 实时 ASAP 补全（object / surface / interior）

Demo v6 = Demo v5.1 的 tracking/chunk 管线（design_spec.md 不变）+ 实时
ASAP 增强：每个 chunk 物化时，用对齐后的 shape-prior mesh
（`final_mesh.glb`）的逐帧 ARAP 形变，把无效的 object 观测点和 shape-prior
surface/interior 点估算到合理位置，并一起发布进 `object_points`。

来源：下游同事提供的离线全量后处理
`july2_chunk_vis.py::write_asap_online_chunks`（vis.ipynb cell 12），本模块
（`demo_v6/asap.py`）保持其数学与默认参数
（`transfer_method='local_rigid'`、`embed_k=8`、`arap_iter=20`、
`max_constraint_dist=0.03`、`max_constraints=1500`、`min_constraints=30`），
把执行方式从"录完后全量重算"改为"chunk 物化时增量处理"。

## 每帧算法

1. **约束构建**：参考帧 = 本次会话 chunk 0 的第 0 帧 object 列（tracking
   已冻结列身份，因此参考列→mesh 最近顶点的映射在初始化时一次算好）。
   参考门与原版一致：只有第 0 帧上
   `visibilities & motions_valid & finite & nonzero` 成立的列才可能成为
   约束把手（一次判定后冻结）。当前帧上同样四条件全部成立的列成为该帧
   约束（参考位置 ≤ 0.03 m 才挂到顶点；同顶点多约束取平均；超过 1500 个
   按 linspace 采样截断）。
2. **ARAP 形变**：以约束把基准 mesh `deform_as_rigid_as_possible`
   （20 次迭代）到当前帧。约束少于 30 个或结果非有限时，**沿用上一帧的
   mesh 顶点**——这是下游实验代码提供的回退行为，我们不喜欢它会在长时间
   遮挡时静默冻结几何，但为了与离线后处理契约一致先保留（代码内有注释，
   未来与下游一起重新讨论）。
3. **点位搬运（local_rigid，向量化）**：object 参考点、surface 点、
   interior 点各自在初始化时嵌入参考 mesh（k=8 最近顶点 + 反距离权重，
   一次预计算）；每帧对每个点做加权刚体拟合（批量 3x3 SVD），把点随
   mesh 运动搬运。数学与原版逐点循环一致，只是批量化以满足实时。
4. **回填**：`object_visibilities & object_motions_valid & finite & nonzero`
   成立的条目永远用真实测量覆盖估算值（估算只填无法实现该条件的条目）。

## 发布契约

- `object_points = [filled original object points, deformed surface points,
  deformed interior points]`（点轴拼接）。原始未恢复的 tracking **不另存**。
- `object_visibilities` / `object_motions_valid`：估算条目**保持原始值**
  （即 False），surface/interior 列恒为 False。依据下游消费方式：
  `realtime_phystwin` 的 chamfer loss 按 `object_visibilities` 门控、track
  loss 按 `object_motions_valid` 门控——保持 False 使估算值永远不会被当作
  监督信号，而直接消费 `object_points` 的一侧（粒子初始化/可视化/几何）
  得到完整且时间连贯的点集。"估算与否"可由掩码推导
  （object 列：`~(vis & motions)` 即估算；prior 列恒为估算），无需额外键。
- `object_colors` 与 `object_points` 同形状：object 列保留追踪色，
  surface 列用默认青色 (0,1,1)，interior 列用默认橙色 (1,0.65,0)
  ——沿用仓库 shape-prior 可视化配色约定。
- **合成 query id**：surface/interior 不是 tracker query。候选方案比较：
  负数 id（与 trace 数组的 -1 填充哨兵冲突，弃）；高位标志位（日志不可
  读，弃）；大偏移基址（可读、可排序、范围判定即成员判定，选用）。
  surface id ∈ [1e9, 2e9)，interior id ∈ [2e9, 3e9)，与会话 tracker id
  （arange，数量级 ≤1e4）不可能重合。`object_sample_query_ids` 等身份数组
  同步扩展；发布层 `query_ids` / `query_semantic_labels` 也追加这些
  synthetic id，对应 semantic label 为 object (`1`)；tracking runtime 的
  原始 tracker schema 不回写这些发布层 id。
  `object_volume_sample_indices`/`object_sample_indices` 以 -1 填充；
  `object_track_status` 扩展为 `"prior"`。
- 静态字段 `surface_points` / `interior_points` 原样保留。
- manifest 增加 `asap_*` 遥测（mesh 路径、约束数 min/max、回退帧数、
  估算条目数、每 chunk 耗时 `asap_ms`）。manifest 的
  `object_track_*` 字段在增强**之前**计算（反映真实 tracking）；payload
  内的 `object_point_count` 等质量计数在增强之后计算，因此包含 prior 列
  ——真实列数以 `asap_object_column_count` 为准；
  `first_frame_zero_object_points` 在增强后恒为 0（零占位已被估算覆盖）。
- 流式路径下，若首个窗口物化时 warmup 尚未写完（显式 surface/interior
  覆盖会跳过 shape-point 等待），会以 `shape_prior_wait_timeout_s` 等待
  `shape_prior_case_dir` 出现；终态失败或超时仍然 fail fast。

## 失败语义

- ASAP 需要 `final_mesh.glb`（默认
  `<shape_prior_case_dir>/shape/matching/final_mesh.glb`，可用
  `--asap-mesh-path` 覆盖）。mesh 缺失/为空 → **fail fast**
  （`AsapMeshError`），不静默跳过。
- 关闭开关：`--no-asap-augment`（默认开启）。

## 实时性

- 每 chunk 的增量成本 = T 次 ARAP（约束→形变）+ 3 次批量刚体搬运。
  嵌入、KDTree、参考列→顶点映射全部一次性预计算。ARAP 是主要成本
  （C++ 实现，T=25/35 帧在 7 s 的 chunk 预算内），`asap_ms` 遥测用于
  实测监控。
- 借帧（design_spec.md 的 lookahead）先于 ASAP 发生：ASAP 只处理已切片
  的发布帧，借帧数据不会进入 ASAP 状态。
- 跨 chunk 状态仅有：基准 mesh、参考嵌入、上一帧顶点（回退用）。
