我在第一frame（warmup frame） have already label query points into object points and conttrller points，这个是固定的，之后一直不变。第一帧必须 visible 且落在第一帧 object processed mask 内，才被定义为 object query。第一帧必须 visible 且落在第一帧 controller processed mask 内，才被定义为 controller query。

正式 Demo v6.1 的 EdgeTAM 是一个 streaming session 里追三个 obj_id：`hand_a`、`object`、`hand_b`。两个 hand id 分开由 EdgeTAM propagate；`controller` 在后续 PCD、TAPNext++、tracking 里仍然定义为 `hand_a | hand_b` 的 union。这个不是参数控制的模式，frame 0 必须能把 controller prompt 分成两只 hand，否则 fail fast。

这里的 `processed mask` 只有一个正式来源：5 FPS formal frame 先经过
`0.2 < depth < 1.5 m` depth gate，再对 object/controller 分别执行固定
PT radius rule（`radius=0.01 m`, `nb_points=40`），由三维离群点反向清理二维
mask。tracker、runtime PCD、shape prior 和 prepared PhysTwin 必须消费同一份
结果；这个前置收敛不改变下文的 `temporary_invalid`、controller anchor 冻结或
恢复规则。

## warmup / 正式 chunk 两阶段时间线

- warmup / shape-prior 阶段：只服务初始化和 shape prior 生成；左侧预览（input_frames.jsonl）持续显示；这个阶段处理过的帧**不进入**正式 final_data chunk 时间线（frames.jsonl 只写 warmup frame 0 这一行，其余帧照常喂 EdgeTAM/TAPNext++ 但不落 product 行）。
- 正式 chunk 阶段：shape prior ready 之后，从当前第一帧起记为 output frame 1，直接拼接在 warmup frame 0 后面，此后每 35 帧生成一个 chunk。发布时间线按 `frame0=0.0s, frame1=0.2s, frame2=0.4s, ...`（frame_index × 1/fps）计；行里的 `source_timestamp_s`/`source_frame_index` 保留真实采集来源作为 provenance。
- 依据操作员 hold-still 约定：warmup 期间手/controller/object 保持不动，因此 frame 0 与 frame 1 之间的接缝运动视为 0；等待期积压的历史帧不补处理。
- shape prior 进入终态失败（failed/unavailable）时闸门解除，让 chunk bridge 的 shape-prior 错误路径正常报错，而不是让行流无声停滞。
- 闸门自带 deadline：`--shape-prior-timeout-ms` 限定行流最多被扣留多久；超时后行流永久恢复，由 bridge 的 shape-prior 等待/失败路径响亮报错（防止 prior 子进程挂死导致无限静默停滞）。
- warmup frame 0 的锚位只能由 chunk-ready 的行占据（controller ≥ 30 点、object > 0 点，与 bridge 的 `_row_ready_for_realtime_chunk_start` 一致）。canonical PT 后 object/controller 任一为空会在 camera 进程立即 fail fast；类别非空但 controller 尚不足 30 点的 startup row 仍可由 bridge 修剪，且不触发闸门。
- run 在闸门期内结束（prior 未就绪、正式时间线从未开始）时，收尾必须响亮报错并在 metadata 标记 `formal_timeline_incomplete`，绝不能以"成功零 chunk"收场。`--duration-s` 只计正式采集时段，不含闸门等待。

## SAM3D generate 输出契约

- Demo v6.2 调用 SAM3D pipeline 时固定使用
  `with_layout_postprocess=False`。本流水线只消费生成的 GLB mesh 和 Gaussian；
  SAM3D 返回的 layout/pose 字段没有下游消费者。
- 生成后的 mesh 会由独立 `shape_prior_align` 阶段对齐到 frame-0 观测，因此在
  generate 内再次执行 layout pose 后处理既不决定最终配准，又会增加 warm-up
  关键路径。
- `with_mesh_postprocess=True` 和 `with_texture_baking=True` 保持启用；前者服务
  后续 mesh 对齐与采样，后者保留 GLB 的纹理产物。禁用范围仅限 layout pose
  后处理。

## warmup 时间分析契约

- `capture/shape_prior_profile.json` 是 warmup 性能分析的权威快照；
  `pipeline_status.jsonl` 只服务实时 UI，不用于跨进程相减或性能结论。
- 所有 duration 使用 `time.perf_counter()` wall time，单位统一为毫秒。
  跨 shape-prior 子进程的 READY/GO 判断使用同机 `time.time()` epoch，并只
  用于计算预热是否及时；source timestamp 不参与性能计算。
- `shape_prior_timing.schema_version = 1`。`pre_submit` 记录 runtime start、
  frame-0 receive/mask/PCD ready 到 submit；`critical_path` 固定按
  `case_write, upscale, segment_image, generate, align, sample,
  result_finalize` 排列。
- `critical_path` 的父进程 wall duration 是瓶颈排名的依据。子进程
  `timing_ms.total_ms` 是主动工作合计、不含等待 GO 的 idle；
  `process_lifetime_ms` 单独包含进程生命周期，`go_wait_ms` 单独报告 idle。
- `accounted_ms` 是七段 duration 之和，`unattributed_ms = total_ms -
  accounted_ms`；负数、非有限值、阶段错序、错误 schema、缺失 completed
  stage profile 都立即失败，不能输出误导性的零耗时。
- `warmup_runtime_start_to_shape_prior_ready_ms` 表示 prior 已算完；
  `warmup_shape_prior_ready_to_gate_open_ms` 还包括 capture 产物写入和正式路径
  实际观察 READY 的延迟；`warmup_total_ms` 才是操作员从 camera runtime
  start 等到可以开始正式运动的总时间。
- 删除旧的 `shape_prior_submit_ms`、`first_mask_depth_pair_ms` 和
  `first_strict_pair_ms`：本地 v6.2 从未给后两项赋真实测量，且前者曾错误表示
  request 总耗时。对应语义现在分别由
  `runtime_start_to_shape_prior_submit_ms`、`shape_prior_request_total_ms` 和明确
  的 pre-submit milestone 取代，不能继续写误导性的 `0.0`。

在我们设计demo 6.1开始会建立一个总表，我们会用dictionary like的数据结构存储每一个controller point的最近的100 controler point。（如果不足则选满同一只手），你必须确保我们的每一个controller point的最近的100 controler point选择的时候都是在同一只手上
我们会考虑二个大类3种情况。这里先把 invalid state 定义清楚：

- `temporary_invalid`：这一帧的直接观测不能用，但不改变 query/anchor 身份。它包括两种来源：第一，tracker 这一帧仍然认为该 query visible，但是这个 visible track pixel 没有通过当前语义 mask、depth/PCD、或者 motion consistency gate；第二，TAPNext++ 这一帧认为该 query invisible、out of track、或者 loss track。TAPNext++ 本身不提供“这个 query 从此永久丢失”的 lifetime state，所以 tracker invisible / loss track 也只归为这一帧的 `temporary_invalid`。

#大类一 tracker 认为它 visible：
##visible
对 object point：
它当前像素仍然落在 object processed mask 里，再看这个像素对应的 PCD/depth 是否有效，我们就认为这一帧的 object observation 有效，然后从该像素的 PCD 里取 3D 点。
如果这一帧 tracker visible，但是当前像素不在 object processed mask 里，或者 PCD/depth 无效，或者 motion consistency 失败，那么这一帧标为 `temporary_invalid`。如果 tracker invisible、out of track、或者 loss track，也标为 `temporary_invalid`。但这个 object query 本身不会因此被永久删除。
对 controller point：
后续每一帧，同样要求 tracker visible，并且当前像素落在 controller processed mask 里，再看这个像素对应的 PCD/depth 是否有效，我们就认为这一帧的 controller observation 有效，然后从该像素的 PCD 里取 3D 点。如果 tracker visible，但当前像素没有通过 controller processed mask、PCD/depth、或者 motion consistency gate，这一帧标为 `temporary_invalid`。如果 tracker invisible、out of track、或者 loss track，也标为 `temporary_invalid`。在原始的Phystwin中，controller 后面会更严格：它必须整段序列都保持有效可见。只要中间某一帧丢失、出界、深度无效、或者离开 controller mask，它通常就不能作为最终 controller handle。我们第一个chunk也是这样，但是我们必须保证chunk之间的controller anchor的不变和一致。后续 chunk 已经冻结的 controller anchor 不会因为某一帧 `temporary_invalid` 而被改成另一个 anchor；这些状态只决定这一帧是否需要恢复/代理。

##做运动一致性过滤
object point 会做局部邻域运动一致性检查：如果一个 object point 的运动和周围 1cm 内邻居的运动不一致，或者有效邻居太少，那么该点在这一帧的 motion valid 会被置为 false。我们会标记这一个frame出现了temporary_invalid。
controller point 也做同样的邻域运动一致性检查，但更严格：controller 是控制 handle，要求整段稳定。如果某个 controller point 缺帧或运动不一致，它会从最终 controller 候选里被剔除。这个对于我们chunk第一个是这样的，但是后续chunk已经选定了controller anchor，所以我们必须保持controller anchor一致；这一帧的 motion valid 会被置为 false，表明出现了`temporary_invalid`，并进入恢复/代理逻辑。但是和原始phytwun不同，这个只是就这一个frame而言，不改变已经冻结的 controller anchor 身份。

#大类二 tracker 认为它 invisible：

这个状态仍然是 `temporary_invalid`：tracker 已经认为该 query invisible、out of track、或者 loss track，但是会在后续frame恢复，不要再搞什么删除query点之类的活动。对于 object points，这一帧没有可用直接观测；对于第一个chunk里的controller candidates，最后的controller anchor不会选择它；但是第一个chunk之后，controller anchor 已经冻结，我们不能改变 controller anchor identity，这个时候就需要处理这一帧的 `temporary_invalid`。


For 后续chunks and controller anchor：
这里必须区分 controller anchor 的身份和每一帧的数值来源。第一个 chunk 选定 controller anchors 之后，chunk topology 永远不变：`controller_sample_query_ids[j]` 始终表示第 `j` 个 controller anchor 在初始化时选中的原始 query id。后续任何 frame/chunk 都不能改写 `query_ids`、`query_semantic_labels`、`controller_sample_query_ids`，也不能因为某个 anchor tracking failure 而把第 `j` 列改成另一个 query。也就是说，`query_schema_hash` 必须保持不变。

当第 `j` 个 controller anchor 在后续 frame/chunk 变成 `temporary_invalid` 时，我们只允许替换这一列的数值来源，不替换这一列的身份。`controller_points[t, j]` 可以由附近 controller candidates 拟合、恢复、或者代理得到，但是第 `j` 列的身份仍然是原始的 `controller_sample_query_ids[j]`。

如果后续chunks某一个frame controller anchor出现了`temporary_invalid`，我们通过他附近的dictionary like的数据结构存储每一个controller point的最近的100 controler point，我们从中挑选出最近的15个当前帧没有`temporary_invalid`的临近controller points，通过局部刚体配准（用附近仍然有效的 controller points 估计一个从第一帧到当前帧的局部刚体变换，然后把这个变换作用到丢失的 anchor 上）临时在这个frame拟合出这个frame controller anchor的位置。如果 `temporary_invalid` 的来源是 TAPNext++ invisible / out of track / loss track，也走同一套恢复/代理逻辑，不改变第 `j` 列的 controller anchor 身份，不改变 `controller_sample_query_ids[j]`，也不改变 `query_schema_hash`。


1.特殊情况：比如 100 个邻居里少于 15 个有效，我们可以只选10个，100 个邻居里少于 10 个有效，我们只选5个，100 个邻居里少于 5 个有效，则从剩余的其他有效(not tempority invlaid)的controller anchor中挑选5-15个最近的controller anchor来估算（越多越好但是不能跨手），如果这个都没有再fail back-》 throw 异常
2.最近 100 是按 first-frame 3D 位置计算，算完之后就不再更新来节省算力
3.motion consistency或者一些细节问题，在不损害我们实时性的前提下，你必须完全参考data_process_origin做法，确保一致性
