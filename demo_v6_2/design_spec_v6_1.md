# Demo v6.1 Design Spec — 实时 ASAP 补全（object / surface / interior）

Demo v6.1 = Demo v5.1 的 tracking/chunk 管线（design_spec.md 不变）+ 实时
ASAP 增强：每个 chunk 物化时，用对齐后的 shape-prior mesh
（`final_mesh.glb`）的逐帧 ARAP 形变，把无效的 object 观测点就地估算到
合理位置（`object_points` 宽度不变），并把形变后的 shape-prior
surface/interior 轨迹作为独立逐帧键 `asap_surface_points` /
`asap_interior_points` 发布。

来源：下游同事提供的离线全量后处理
`july2_chunk_vis.py::write_asap_online_chunks`（vis.ipynb cell 12），本模块
（`demo_v6_1/asap.py`）保持其数学与默认参数
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

- `object_points` **宽度保持 tracking 原样**（M 列）：无效条目就地回填
  估算值。原始未恢复的 tracking **不另存**。
- 形变后的 shape-prior 轨迹发布为**独立逐帧键**：
  `asap_surface_points (T, S, 3)`、`asap_interior_points (T, I, 3)`
  （`data_keys.py` 的 OPTIONAL_TIME_KEYS 早已声明这两个键，沿用下游离线
  后处理的键名）。它们**不再**拼进 `object_points`，也不占位
  `object_colors` / `object_visibilities` / `object_motions_valid`。
- `object_visibilities` / `object_motions_valid`：估算条目**保持原始值**
  （即 False）。依据下游消费方式：`realtime_phystwin` 的 chamfer loss 按
  `object_visibilities` 门控、track loss 按 `object_motions_valid` 门控——
  保持 False 使估算值永远不会被当作监督信号，而直接消费 `object_points`
  的一侧（粒子初始化/可视化/几何）得到完整且时间连贯的点集。
  "估算与否"可由掩码推导（`~(vis & motions)` 即估算），无需额外键。
- object 的身份数组（`object_sample_query_ids` 等）、colors、status 全部
  不变——prior 点不再进入 object 列，不需要合成 query id、默认色和
  `"prior"` 状态（payload 层的 `_extend_query_schema_for_sample_ids`
  机制保留，对本契约为 no-op）。
- 静态字段 `surface_points` / `interior_points` 原样保留（第 0 帧参考
  位形）；`asap_surface_points[0] == surface_points`、
  `asap_interior_points[0] == interior_points`（首窗口首帧为参考帧）。
- manifest 增加 `asap_*` 遥测（mesh 路径、约束数 min/max、回退帧数、
  估算条目数、每 chunk 耗时 `asap_ms`）。manifest 的
  `object_track_*` 字段在增强**之前**计算（反映真实 tracking）；
  `first_frame_zero_object_points` 在增强后恒为 0（object 列零占位已被
  估算覆盖）。
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

## online_data 逐帧 RGB-D 归档（2026-07-05）

`online_data/` 在 `chunks/` + `manifest.json`（保持不变）之外，为每个已发布的
online frame k 增加原始传感器产物，布局与离线 recording case 一致：

```
online_data/
    color/0/{k}.png        # 该帧的原始 RGB（非分割着色）
    depth/0/{k}.npy        # (H, W) uint16 毫米，invalid = 0
    calibrate.pkl          # [4x4 camera-to-world]（单相机 list）
    metadata.json          # intrinsics(1,3,3) / WH / frame_num / serial_numbers
    enhance_metadata.json  # online frame -> source frame 映射表
    chunks/ manifest.json  # 不变
```

- **一一对应**：只归档真正进入 chunk 并发布的帧（连续 online index
  0..N-1），与 `capture/input_rgb`（记录所有输入帧）不同。chunk 0 的第
  0 帧（shape-prior/warmup 锚帧）同样输出为 `color/0/0.png` +
  `depth/0/0.npy`。
- **深度格式统一**：canonical 格式为 uint16 毫米（invalid=0），由
  `phystwin_strict_product.depth_m_to_mm_u16` 从流水线的 float 米深度
  转换。RealSense 原始 uint16 units（标准 0.001 m/unit 刻度）经
  units→米→毫米往返 bit-exact，等价于直接 copy；FFS 生成的 float 米走
  同一转换，下游格式完全一致。`PreparedPhysTwinFrame` 新增
  `depth_mm_u16` 字段并随 NPZ 持久化。
- **下游契约**：对齐 `data_process_origin/data_process_pcd.py` 的读取
  方式 —— depth `np.load(...)/1000.0`、color `cv2.imread`（BGR 落盘）、
  `calibrate.pkl` 为可索引的 per-camera 4x4 c2w 序列、`metadata.json`
  提供 `intrinsics`（(num_cam,3,3)）/`WH`/`frame_num`/`serial_numbers`、
  文件名为连续整数 `0..frame_num-1`。
- **写入顺序**：帧文件先落盘（fsync）→ commit 对应 chunk → 原子重写
  `metadata.json` / `enhance_metadata.json` —— `frame_num` 永不指向
  不存在的文件，也永不统计未 commit chunk 的帧；已 commit 的 chunk
  必有归档帧。深度 > 65.535 m（FFS 远场垃圾）归为 invalid=0 而非饱和。
  缺表标定时 c2w 回退 identity、接受 fx/fy/cx/cy intrinsics 形式，与
  chunk stream 的容错一致。
- **fail fast**：`frames.jsonl` 行缺少 canonical
  `prepared_phystwin_frame_path` 或对应文件不存在时，在读取该行时立即以
  `OnlineFrameArchiveError` 中止流；旧 prepared NPZ 无 `depth_mm_u16`或
  online index 不连续也同样立即失败。不再从 RGB/depth/mask/trajectory
  sidecar 重建 chunk。
- **清理**：新 run 开始由 `prepare_realtime_output_for_new_run` 移除整个
  `online_data/`；`OnlineFrameArchive` 构造时额外清理 `color/`、
  `depth/`、`metadata.json`、`calibrate.pkl`、`enhance_metadata.json`
  （不触碰 `chunks/`），防御性覆盖手工复用场景。

## downstream.mode：demo 可视化与 Phystwin_shen 二选一（2026-07-05）

原先 config 只有 visualizer 一种下游设计。现改为显式枚举
`downstream.mode: disabled | demo_visualizer | phystwin_shen`（YAML 默认值
为 `phystwin_shen`；YAML 值绕过 argparse choices，因此
`resolve_downstream_mode` 在运行时再次校验，未知值 fail fast）。
每个 session 只运行一种下游。

- **demo_visualizer**：原 `visualizer_mode: "window"` 行为不变
  （side-by-side 随 capture 启动、output-only 等第一个 chunk）。
- **phystwin_shen**：自动化原手动流程 —— 在 shape prior ready 后直接在
  第二块 GPU 上启动 Phystwin_shen 的 `train_online_warp.py` + HTML
  viewer（`scripts/html_realtime_viewer.py`，phystwin_shen 模式总是带
  viewer）。
  - **触发点**：`shape_prior/points.npz` 出现（warmup 完成的落盘产物，
    此时 SAM3D stage 子进程已退出、GPU 1 已清空）；由
    `stream_chunk_data_from_headless_capture` 的 `before_poll` 每轮轮询，
    只触发一次。warmup 关闭时首轮立即启动。`train_online_warp.py` 自身
    继续等待第一个 committed chunk（轮询 `online_data/manifest.json`）。
  - **GPU**：子进程 `CUDA_VISIBLE_DEVICES` 取
    `gpu.phystwin_shen_cuda_visible_devices`（默认 "1"），同时传
    `--device cuda:0`，即进程内 cuda:0 = 物理 GPU 1。
  - **repo/env 进 YAML**：`phystwin_shen.repo_path`（默认
    `/home/xinjie/Phystwin_shen`）与 `phystwin_shen.conda_env`（默认
    `demo_2_max`）；viewer/train 参数同样在 `phystwin_shen:` 段，YAML
    叶子名直接使用 Shen argparse 名称（如 `host`、`port`、`device`、
    `batch_size`、`segment_len`）。
  - **当前 Shen CLI 契约**：trainer 只接收
    `--online_dir <base_path>/online_data`，不再接收旧
    `--base_path/--case_name/--static_data_path`；viewer 接收
    `--base_path <base_path> --case_name online_data`，并显式接收必需的
    `--rgb_dir <base_path>/online_data/color`。两者共享显式传入的
    realtime snapshot 目录。
  - **端口**：viewer 绑定 `host:port`（默认
    127.0.0.1:8765）。启动前若端口被占用，直接 kill 占用进程
    （SIGTERM→SIGKILL）；无法识别占用者或 kill 后端口仍被占 →
    `PhystwinShenLaunchError` fail fast。
  - **case dir 预置**：trainer 从 `online_dir`、viewer 从
    `base_path/case_name` 读取同一份
    `online_data/{calibrate.pkl,metadata.json}`，早于第一个 chunk commit。
    因此 `OnlineFrameArchive.initialize_case` 在 capture metadata 可用时
    立刻预置 calibrate.pkl + metadata.json
    （frame_num=0）+ 空 enhance_metadata.json，帧计数不变式保持不变。
  - **生命周期**：与 viewer 窗口同策略 —— demo run 结束后 train/viewer
    继续运行，run_summary 记录 `phystwin_shen_*`（命令、日志路径、
    viewer URL、端口接管、return code / left_running）。退出码策略：
    phystwin_shen 模式下未启动 → 1；任一子进程非零退出 → 透传。
    日志写入 `base_path/phystwin_shen/*.log`。
