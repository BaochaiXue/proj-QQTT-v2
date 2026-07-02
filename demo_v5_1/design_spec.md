# Demo v5.1 Cross-Chunk Tracking Design

本文解释 Demo v5.1 是如何把单相机实时 RGB-D、mask、PCD、TAPNext++
tracking、shape-prior 结果组织成可增量消费的 online chunks，并重点说明
跨 chunk tracking 身份是如何保持稳定的。

`demo_v5_1/pipeline.md` 主要解释 warmup。本文补上运行时设计，尤其是
`online_data/chunks/chunk_*.pkl` 和 `data/final_data.pkl` 背后的身份契约。

## 设计问题

原始离线 `data_process_sam3d` 可以等整个 case 完成后一次性写出
`track_process_data.pkl` 和 `final_data.pkl`。Demo v5.1 不能这样做：

- 相机和 tracker 在实时运行，chunk 必须边录边发布。
- FuturePhysTwin/visualizer 侧希望尽快消费已完成窗口。
- 每个 chunk 又不能像独立 case 一样重新选择 object/controller 点，否则
  chunk 0 的第 12 列和 chunk 1 的第 12 列可能代表不同 tracker query。

所以 v5.1 的核心设计不是“把每个 chunk 伪装成一个新的离线 case”，而是：

1. 把整次 live/fake-live run 视为一个在线 case。
2. 用固定大小窗口切时间轴。
3. 在第一个有效 chunk 里冻结 query schema、object sample、controller handles。
4. 后续 chunk 只更新这些 frozen columns 的每帧位置、可见性和恢复状态。
5. 通过 `query_schema_hash` 证明每个 chunk 的 topology 是同一个。

## 路径总览

运行入口是 `demo_v5_1/main.py`。

1. `main.py` 读取 `config/default.yaml`，把
   `chunk_frame_count = round(replay_fps * chunk_seconds)` 作为窗口大小。
   默认 `replay_fps=5.0`、`chunk_seconds=7.0`，所以默认每个 chunk 是 35 帧。
2. `main.py` 启动 `main_data_processing.py`。这个进程负责实时 RGB-D、
   mask、PCD、TAPNext++ tracking，并在 headless capture 目录下写：
   `metadata.json`、`frames.jsonl`、`prepared_phystwin/*.npz`。
3. `main.py` 同时调用
   `stream_chunk_data_from_headless_capture()` tail `frames.jsonl`。
   每收满 `chunk_frame_count` 行就关闭一个窗口。
4. 窗口关闭后，chunk materializer 等待 shape-prior surface/interior points
   变为可用。等待只发生在 materialization 阶段，不改变相机/tracker 的源时间轴。
5. `_chunk_data_window_from_prepared_frames()` 读取窗口内 prepared frames，
   运行与 origin data process 对齐的 mask/PCD/track/filter/sample 语义。
6. `build_chunk_data_payload()` 生成本窗口的 `final_data`、`track_process`
   diagnostics 和 manifest fields。
7. `ChunkDataWriter` 写：
   - `online_data/chunks/chunk_000000.pkl`
   - `online_data/manifest.json`
   - `data/final_data.pkl`
   - `data/metadata.json`

其中 `online_data/chunks/*.pkl` 是低延迟增量视图，`data/final_data.pkl`
是已发布 prefix 的聚合视图。二者表达同一个在线 case，而不是两套数据。

## 关键名词

`source_frame_index`
: 相机或 fake-live 输入时间轴上的原始帧号。因为 warmup、backlog 或
  source replay，它可以跳号。

`start_frame` / `end_frame`
: online final-data 时间轴上的半开区间 `[start_frame, end_frame)`。
  已发布 chunks 必须从 0 开始连续。

`source_frame_indices`
: 当前 chunk 每个 online frame 对应的源帧号。长度必须等于 chunk frame
  count，但数值可以不连续。

`query_points_yx`
: tracker 初始化时采样出来的 2D query 点。它的顺序定义了后续
  `tracks_yx`、visibility、query ids 的原始拓扑。

`query_ids`
: v5.1 写入 final-data contract 的稳定 query 身份。它不等价于 chunk 内
  column index；column index 是最终 sample/handle 的位置，query id 是该列
  来自哪个 tracker query。

`query_semantic_labels`
: 每个 query 的语义类别。v5.1 使用 object/controller 语义，two-hands 模式
  仍会在 processed mask 中折回 controller union 供最终控制点使用。

`object_sample_query_ids`
: 最终 `object_points[:, j, :]` 第 `j` 列对应的 frozen object query id。

`controller_sample_query_ids`
: 最终 `controller_points[:, j, :]` 第 `j` 个 handle 对应的 frozen controller
  query id。

`query_schema_hash`
: 对 `query_ids`、`query_semantic_labels`、`object_sample_query_ids`、
  `controller_sample_query_ids` 做 sha256 得到的 topology 证书。它证明多个
  chunks 可以按时间拼接而不改变列身份。

## Chunk 边界如何定义

chunk 边界只由 `frames.jsonl` 的完整 rows 数量决定。

在 live path 中，`stream_chunk_data_from_headless_capture()` 循环读取
`frames.jsonl` 新增完整行。每一行代表一个已经由 camera process 写好的
prepared frame。row buffer 收满 `chunk_frame_count` 后，当前窗口关闭。

shape prior 不参与决定窗口边界。它只在窗口关闭后决定这个窗口能不能写出
完整 `final_data`：

- shape prior enabled 时，materializer 等待 `surface_points` 和
  `interior_points`。
- shape prior disabled 时，chunk 可以不带 shape-prior points。
- shape prior 失败、不可用或超时会在 materialization 边界报错。

warmup frame 也属于正式 online 时间轴。若 frame 0 因 shape-prior warmup
延迟写入，v5.1 会保留它，让它成为 `chunk_000000.pkl` 的第 0 帧，而不是
写成一个 sidecar chunk。

因此有两个不变量：

- `start_frame/end_frame` 是 online frame 的连续编号。
- `source_frame_indices` 是源时间线映射，可以跳号，但不能改变 online
  chunks 的连续性。

## 跨 Chunk Tracking 的核心设计

### 1. 先冻结 session query schema

`chunk_data_stream.py` 在整个 stream 生命周期里维护一个
`session_query_schema` 字典。它不是每个 chunk 的局部变量，而是跨 chunk
复用的运行状态。

第一个 chunk 进入 `_track_input_with_session_query_schema()` 时，
`session_query_schema` 还没有 `query_ids`。函数会调用 strict product 的
`build_track_process_input()`，用第一帧 mask 对 tracker query 做语义分类，
然后把得到的 `query_ids` 和 `query_semantic_labels` 存到 session state。

后续 chunk 再进入这个函数时，会把同一组 `query_ids` 和
`query_semantic_labels` 传回 strict builder，并且用 `np.array_equal()` 校验
结果没有改变。任何 query id 或 semantic label 变化都会直接 `ValueError`。

这一步的含义是：chunk 边界不能重启 tracker topology。只要同一次 run
还在继续，query 数量、顺序、语义都必须与第一组可发布 chunk 一致。

### 2. Object columns 由第一 chunk 冻结

object 侧由 `StreamingObjectTrackSelector` 维护跨 chunk 状态。

第一 chunk：

1. 找出第一帧可见、有限、非零的 object candidate queries。
2. 用 5 mm volume grid 做采样，shape-prior surface/interior points 参与
   bounds 计算。
3. 把被选中的 query ids 记录为 `_initial_query_indices`。
4. 初始化 `_active_query_indices`、`_last_points`、`_last_colors`。
5. 输出：
   - `object_track_query_indices`
   - `object_track_active_query_indices`
   - `object_sample_query_ids`
   - `object_selected_query_ids`
   - `object_track_status`

后续 chunk：

1. 对每个 frozen object column，优先用 `_initial_query_indices[j]` 找同一个
   query。
2. 如果该 query 不可用，再尝试 `_active_query_indices[j]`。
3. 如果 direct candidate 存在且有效，就直接把该 candidate 写回第 `j` 列。
4. 如果 direct candidate 丢失，则尝试用邻域运动恢复 lost anchor。
5. 如果仍无法恢复，则保留上一段 `_last_points/_last_colors`，把状态写成
   `missing`，并把 active query 置为 `-1`。

因此 `object_points[:, j, :]` 的第 `j` 列不会因为新 chunk 开始而被重新
volume-sample 成另一个 query。列数也不应在已发布 chunks 间变化。

### 3. Controller handles 由第一 chunk 冻结

controller 侧由 `StreamingControllerTrackSelector(count=30)` 维护跨 chunk
状态。

第一 chunk：

1. 从 controller candidates 中筛出可选择点。
2. 用 farthest point sampling 选择 30 个 controller handles。
3. 保存 30 个 `_initial_query_indices`。
4. 保存 `_active_query_indices`、`_last_points`、`_last_velocity`。
5. 为每个 anchor 建 backup query bundle，用于后续恢复。
6. 输出 30 列 `controller_points`，每列状态为 direct。

后续 chunk：

1. 每个 controller handle 先按 frozen query id 找 primary candidate。
2. 如果 primary 通过 raw visible、processed mask、depth、motion checks，
   该帧写 `direct_valid`。
3. 如果 primary 因 processed mask 或 motion consistency 被拒绝，但 raw
   point 可用，则尝试接受 raw measurement 或用 backup bundle 恢复。
4. 如果 primary depth invalid 或 TAPNext++ lost，则优先用 bundle recovery。
5. 如果 bundle recovery 也不可靠，则用上一点位和上一速度做 prediction，
   并写低 confidence / unrecoverable mode。
6. 每个 handle 的 source query、confidence、failure reason、neighbor support
   counts 都写入 diagnostics。

核心约束是：controller 的 30 个 handle 是第一 chunk 选出来的 30 个控制锚点，
后续 chunk 只能 direct/recover/predict 这些锚点，不能重新 FPS 选择另外 30 个。

### 4. Query hash 是跨 chunk 的 topology 证书

`build_query_schema_payload()` 把以下字段写入 final_data 和 track_process：

- `query_schema_version`
- `query_schema_hash`
- `query_ids`
- `query_semantic_labels`
- `object_sample_query_ids`
- `controller_sample_query_ids`

`query_schema_hash` 覆盖 query id、semantic label、object sample ids、
controller sample ids。它不是内容 hash，不会随每帧 3D 位置改变；它只描述
“这些 chunks 的列身份是不是同一套 topology”。

`build_chunk_data_payload()` 把同一组 query schema fields 放进 final_data 和
track_process，manifest 也记录 `query_schema_version/query_schema_hash`。这样
读者可以不用扫描大数组，也能知道当前 chunk 的 topology 是否仍然一致。

### 5. Writer 只发布稳定结果，不重新解释 identity

`ChunkDataWriter` 的职责是发布，不是重建 tracking 身份。

它做三件事：

1. 把本窗口的 frame-axis arrays 写入
   `online_data/chunks/chunk_XXXXXX.pkl`。
2. 把所有已提交 chunks 的 frame-axis arrays 拼成 prefix aggregate，
   重写 `data/final_data.pkl`。
3. 把 static fields 和 query schema fields 复制到 chunk 和 aggregate 中。

writer 注释中已经说明 static arrays 会被最新值覆盖，但上游要求它们在已提交
chunks 间保持稳定。也就是说，identity 的正确性来自：

- `session_query_schema` 的 fail-fast 校验。
- `StreamingObjectTrackSelector` 的 frozen object query ids。
- `StreamingControllerTrackSelector` 的 frozen controller handles。
- `query_schema_hash` 的稳定性。

writer 不应该在发布阶段偷偷修复 topology，也不应该在发现不一致时创造新的
兼容路径。

## 发布与失败语义

`track_process_status` 是质量 warning，不是发布门槛。

- `normal`: tracking 质量正常。
- `degraded`: tracking 质量下降，保留在 diagnostics / manifest 中。
- `invalid`: tracking 质量严重异常，保留在 diagnostics / manifest 中。

这三个状态都不能改变发布、回调、stream 继续条件、online frame 编号或
runner 返回码。每个 materialized chunk 都必须写入 `online_data/chunks`，
并 append 到 `data/final_data.pkl` 的 prefix aggregate。这样 online frame
axis 和已 materialized 源窗口保持一一对应，不会因为质量 warning 在中间制造
隐藏时间洞。

downstream 如果需要拒绝低质量数据，应显式读取 `track_process_status`、
confidence、mode、failure reason 等 diagnostics 自己做决策；publisher 不在
发布阶段把 warning 升级成控制流。

## Shape Prior 与 Tracking Identity 的关系

shape prior 只提供几何结构点，不定义 tracker identity。

它影响这些事情：

- 第一个 object volume sampling 的空间 bounds。
- `final_data.pkl` 中的 `surface_points` 和 `interior_points`。
- visualizer / downstream structure points。

它不影响这些事情：

- `query_ids` 的生成。
- object/controller semantic labels。
- `object_track_query_indices`。
- `controller_track_query_indices`。
- 已冻结 columns 的数量和顺序。

所以 shape prior 可以晚到，但不能让 chunk 边界重开，也不能让 selector 在后续
chunk 里重采样 object/controller topology。

## 和 `data_process/record_data_align.py` 的不同

仓库内 formal `data_process/record_data_align.py` 是 aligned-case 生成器，不是
tracking pipeline。

它的输入单位是 raw recording case：

```text
data_collect/<case_name>/
```

它做的事情是：

1. 读取 raw metadata、calibration、camera streams。
2. 以 camera 0 的 step 为主轴。
3. 在其他 camera 的 `step_idx +/- 3` 内找 timestamp 最近的帧。
4. 把匹配成功的 frames 重编号为 `0..N-1`。
5. 写 aligned output：
   - `color/<camera>/<frame>.png`
   - `depth/<camera>/<frame>.npy`
   - 可选 `ir_left/`、`ir_right/`
   - 可选 FFS outputs，如 `depth_ffs/`
   - `calibrate.pkl`
   - split metadata

它携带的是相机身份和 depth contract：

- `serial_numbers`
- camera intrinsics/extrinsics
- depth scale
- depth backend used
- native RealSense depth 或 FFS depth

它不携带：

- object/controller masks
- tracker query ids
- `object_points`
- `controller_points`
- `track_process_status`
- `final_data.pkl`
- online chunks
- shape-prior surface/interior points

因此 formal aligned-case 的 frame continuity 是“相机帧对齐后重编号连续”，
Demo v5.1 的 chunk continuity 是“对象/控制器 tracking topology 跨窗口连续”。
这是两个层级的连续性，不能混用。

## 和 `data_process_origin` / `data_process_sam3d` 的不同

`data_process_origin/` 和 vendored `data_process_sam3d/` 是 Demo v5.1 对齐的
tracking/final-data 语义来源。v5.1 复用了它们的产品语义，但改变了执行方式。

origin tracking path 的形状是离线 case：

1. `data_process_pcd.py` 为每帧生成 dense PCD。
2. `data_process_mask.py` 生成 processed object/controller masks。
3. `dense_track.py` 或等价 tracker 生成全 case tracks。
4. `data_process_track.py` 用第一帧 mask 给 tracks 分类为 object/controller。
5. `data_process_track.py` 对 object/controller motion 做 neighbor consistency
   filtering。
6. `data_process_track.py` 对 controller 做 farthest point sampling，选出最终
   control handles。
7. `data_process_sample.py` 对 object points 做去重/volume sampling，并可把
   shape-prior surface/interior points 写入 `final_data.pkl`。

origin 的关键前提是：整段 case 已经完成，所以“选择最终 object samples”和
“选择最终 30 个 controller handles”天然只发生一次。

Demo v5.1 的难点是：每个 chunk 到来时，只看得到一个窗口。如果直接在每个
chunk 里运行 origin sampling，就会出现：

- chunk 0 选出的 object columns 和 chunk 1 不同。
- chunk 0 的 controller FPS handles 和 chunk 1 不同。
- downstream 拼接时同一列代表不同物理点。
- motion/recovery 状态无法跨窗口延续。

所以 v5.1 的改变是把 origin 的“一次性 case-level selection”显式变成
“stateful streaming selection”：

- `session_query_schema` 替代每个 chunk 局部 query topology。
- `StreamingObjectTrackSelector` 替代每个 chunk 局部 object volume sampling。
- `StreamingControllerTrackSelector` 替代每个 chunk 局部 controller FPS。
- `ChunkDataWriter` 把每个窗口切片发布，同时维护 prefix aggregate
  `data/final_data.pkl`。

换句话说，v5.1 不是削弱 origin 语义，而是把 origin 的 case-level identity
显式外提成跨 chunk 状态。

## Output Contract

每个 online chunk 至少包含：

- `case_name`
- `chunk_id`
- `start_frame`
- `end_frame`
- `source_frame_indices`
- `object_points`
- `object_colors`
- `object_visibilities`
- `object_motions_valid`
- `controller_points`
- query schema static fields

可选 diagnostics 包括：

- `controller_source_query_ids`
- `controller_track_mode`
- `controller_track_confidence`
- `controller_filter_reason`
- `controller_neighbor_support_count`
- `controller_neighbor_*_count`
- `controller_neighbor_fit_residual`
- `object_track_status`
- `controller_track_status`
- `track_process_status`

`data/final_data.pkl` 是同一 schema 的 prefix aggregate：

- frame-axis arrays 通过 time axis concat。
- static arrays 保持同一套 query topology。
- `surface_points/interior_points` 是结构点，不沿 frame axis 变化。

## Downstream Topology Naming Note

当前 v5.1 生产侧使用：

- `query_schema_version`
- `query_schema_hash`

`realtime_phystwin` 的 online reader 期望：

- `topology_version`
- `topology_hash`

二者表达的是同一种概念：跨 chunk 的 query/column topology 证书。但字段名和
版本值当前不是同一个 contract。直接集成前必须明确统一命名或更新 reader
contract，不能假设 downstream 会自动把 `query_schema_hash` 当作
`topology_hash`。

## 必须保持的不变量

1. `start_frame/end_frame` 必须从 0 开始连续，表示 online frame 半开区间。
2. `source_frame_indices` 长度必须等于 chunk frame count，可以跳号，只作为源
   时间线映射。
3. 同一次 run 的 `query_points_yx` 顺序和数量必须 session-stable。
4. `query_ids` 和 `query_semantic_labels` 一经第一 chunk 固定，后续 chunk
   必须完全一致。
5. `query_schema_hash` 必须覆盖 query ids、semantic labels、object sample ids、
   controller sample ids。
6. `object_points.shape[1]` 和 `controller_points.shape[1]` 在已发布 chunks
   之间不得变化。
7. 同一 object column 必须对应同一个 frozen object query id，丢失时只能
   revive 或 hold last state，不能重采样替换。
8. 同一 controller column 必须对应同一个 frozen controller handle，丢失时只能
   direct/recover/predict，不能重新 FPS 替换。
9. shape-prior points 可以决定结构点和第一段 object sampling bounds，但不能
   定义或改变 tracker identity。
10. `track_process_status` 只能作为 warning metadata，不能跳过发布、停止
    stream 或改变 runner 返回码。

## 代码证据地图

- `demo_v5_1/main.py`: orchestration、chunk frame count、subprocess launch、
  live tailing。
- `demo_v5_1/main_data_processing.py`: realtime camera/mask/PCD/tracker、
  prepared frame 写入。
- `demo_v5_1/chunk_data_stream.py`: live/offline capture to chunk bridge、
  shape-prior gate、session query schema、stateful selectors。
- `demo_v5_1/chunk_data_payload.py`: final-data payload、query schema hash、
  runtime contract。
- `demo_v5_1/chunk_data_output.py`: online chunk writer、prefix aggregate
  `data/final_data.pkl`。
- `qqtt/demo/phystwin_strict_product.py`: strict mask/PCD/track semantics、
  streaming object/controller selectors。
- `data_process/record_data_align.py`: formal raw recording to aligned case path。
- `data_process_origin/data_process_track.py`: origin track classification、
  motion filtering、controller FPS。
- `data_process_origin/data_process_sample.py`: origin final-data object sampling
  and shape-prior point sampling。
- `realtime_phystwin/qqtt/data/online_stream.py`: downstream online buffer and
  topology validation expectation。
