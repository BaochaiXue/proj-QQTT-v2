# Demo v5.1 Tracking Realtime Chunk v2 — 对标 data_process_origin 的重设计方案

状态：设计提案 rev2（未实施）。rev1 经三路对抗评审（origin 对标忠实度 /
realtime 状态机可行性 / 契约迁移安全性）修订，评审确认的 17 个问题已全部
合入，见 §12 评审记录。
前置阅读：`demo_v5_1/design_spec.md`（现状契约）、`demo_v5_1/pipeline.md`。
对标基准：`data_process_origin/`（`dense_track.py` → `data_process_track.py`
→ `data_process_sample.py`）。

## 0. 目标与非目标

目标：realtime chunk 管线产出的 tracking 产物（`final_data.pkl` /
`online_data/chunks/*.pkl`）在语义和质量上对标 origin 离线管线，且对标
**可测量**（§8 harness），不能只靠代码 review 断言。

非目标：

- 不改变 5 FPS 的吞吐决策（故意的，见 AGENTS.md 实验约定）。
- 不改变 warmup 手保持不动的操作约定。
- 不在本方案内更换 tracker 模型本体（升级路径在 §10 作为独立决策）。

## 1. 核心诊断：origin 的保证是什么，realtime 挡住了什么

origin controller 链的四个判据（`data_process_track.py`）：

| # | 判据 | 因果性 |
|---|------|--------|
| G1 | 逐帧测量有效（frame-0 mask 分类 + 逐帧 mask 门 + 深度 0.2–1.5 m） | 逐帧 |
| G2 | 邻域运动一致（r=0.01 m 内 ≥5 计数含自身、≥50% 邻居运动差 < r/2=0.005 m/帧） | **与 G3 相互递归 → acausal** |
| G3 | 全程可见（`np.prod(visibilities, axis=0)`）+ 任一帧运动不一致即永久除名并回写掩码 | **acausal** |
| G4 | 从 G3 幸存者的 frame-0 位置 FPS 采样 30 个 handle | acausal（依赖 G3） |

G2 的因果性必须说清楚（rev1 此处错误）：origin 在第 i 帧做邻域共识时，
邻居池先被 G3 的全程掩码稀释（`data_process_track.py:275-286`），任何帧的
失败又回写掩码稀释后续所有帧的池子（`:289,296`）。所以 controller 的 G2
不是"+1 帧前瞻"，而是与 G3 耦合的 acausal 过程；窗口化后邻居池是 origin
的**超集**，共识测试系统性更宽松——§4.2 的 latch 携带就是为了压回这个
偏差。object 侧的 G2 没有这个耦合（无全程掩码、无回写），确实只需一帧
前瞻。

其余关键事实：

1. **origin 的 tracker 本身是流式的**：`cotracker3_online`，16 帧滑窗、
   8 帧步长，内部工作分辨率 384×512（非全分辨率），无全局回溯。
2. G1 逐帧可算；object-G2 只需一帧前瞻。
3. chunk 在关窗后才物化（本来就等 shape prior），窗口内回溯计算零额外
   延迟。
4. origin 的最终保证："**发布的每一列 controller 都是同一物理点的全程真实
   测量**"——注意这比"valid 的都是真实测量"强：它包含**轨迹身份**的全程
   可信（一帧失踪即整列删除，含追溯）。realtime 不可完整复刻，§4.2 给出
   显式弱化后的翻译，不假装等价。

当前实现的病根（审计 F1）：为了让冻结的 30 列每帧有值，
`StreamingControllerTrackSelector` 用 bundle 恢复（conf≤0.85）、航位推算
（conf 0.1）、冻结旧值填充，`conf≥0.25` 写成 visibility；且 payload 阶段
把 selector 的 handle 轴 visibilities/motions_valid **直接丢弃**
（`chunk_data_payload.py:31-38` 只发布 points），发布的
`controller_motions_valid` 实为候选轴数组（`:327-334`）。下游因此对 handle
的真实性一无所知。

## 2. 设计原则

- P1 **身份与数值分离**：冻结拓扑承载跨 chunk 身份；只有真实测量标 valid；
  合成值永不标 valid。
- P2 **时间轴完整，真相用掩码**：每个物化窗口都发布；质量以逐帧掩码表达，
  不用控制流表达（对齐 warning-only exec plan）。
- P3 **acausal 判据显式窗口化**：G2/G3/G4 的窗口化翻译写进契约（§4），
  弱化之处明说，不做隐式近似。
- P4 **常数物理单位化**：随 fps 缩放的阈值以 m/s 表达，按**逐行实测
  时间差**换算（§6），不用名义 fps。
- P5 **对标可测量**：任何"语义等价"的声称必须能被 §8 harness 验证。

## 3. 架构：两级处理

```
Stage A（逐帧，流式，现有为主）
  camera → mask → dense PCD → TAPNext++（可插拔）
        → prepared_phystwin/{seq}.npz + frames.jsonl

Stage B（逐窗口，关窗时，"windowed mini-origin"）
  B1 逐帧有效性       ← build_track_process_input（现有）+ 深度窗恢复
  B2 运动一致过滤     ← apply_phystwin_motion_filters（速度单位化 + 借帧 + latch 携带）
  B3 冻结拓扑更新     ← Streaming selectors（语义改造，§5）
  B4 payload 组装     ← 贴地钳制 + 掩码进 final_data 契约（§5.4/§7）
  B5 发布             ← 全窗口发布，status 仅 warning
```

**前置不变量（时间轴）**：Stage B 的运动过滤假设窗口内逐行连续。demo 标准
配置走 lossless 管线（`_lossless_tracker_worker` + `SameSeqPairer` 强制
连续 seq，`main_data_processing.py:4575-4588,1777-1795`）,该假设成立；
`mask_slot.get_latest_after` 的丢帧 worker 仅是非 lossless 回退路径。本方案
把 **lossless 管线声明为 Stage B 的前置条件**，非 lossless 配置下 B2 的
运动过滤必须禁用（掩码全 False 而非错误计算）。

### 3.1 一帧前瞻物化（消灭窗口边界死帧，修 F6）

origin 的 `motions_valid[i] = vis[i] & vis[i+1]` 需要 i+1 帧。现状按窗口
独立计算,每窗末帧被 `_frame_motion_valid` 的 `out[-1] = out[-2]` 复制
（`phystwin_strict_product.py:1129-1138`）——这是对死帧问题的一个错误
workaround（伪造而非计算）。

改为：窗口 `[start, end)` 收满后**再等一行**（借帧）才物化；运动过滤在
`[start, end]` 上运行。必须同时满足以下状态机与切片规则（rev1 缺失，
评审确认为设计漏洞）：

1. **终态 flush**：`capture_finished()` 且缓冲满而借帧永不到来时，直接
   物化，末帧 `motions_valid = False`——这恰是 origin 对整段视频最后一帧
   的处理。离线转换路径（行数恰为 chunk_size 整倍数时最后一窗无借帧）同
   规则。禁止沿用现有"循环退出丢弃缓冲"的行为把**完整**窗口丢掉。
2. **切片不变量**：所有时间轴数组（build_track_process_input +
   motion filters 产出的全部 (T,N) 键）在进入 selector **之前**统一切回
   `[start, end)`；借帧数据永不进入 selector 状态（`_last_points/_last_velocity/
   _unreliable_chunk_counts`）、不进入 manifest（`source_frame_indices`、
   `chunk_ready_source_seq` 仍取自发布末行而非借帧行）。
3. **删除 `out[-1] = out[-2]`**：末帧 motions_valid 由借帧真实计算（终态
   flush 时为 False）。
4. **计量口径**：`window_closed_wall_s` 保持"缓冲收满"时刻不变（延迟 SLO
   序列不改语义），新增 `borrow_row_wall_s`；`_complete_chunk_backlog` 在
   借帧模式下按 `(rows - 1) // chunk_size` 计算。
5. **借帧同时是下一窗首帧**（carry），不重复计费：质量统计
   （`_quality_status`、`_unreliable_chunk_counts`）只在其所属窗口计一次。

代价：物化延迟 +1 帧（5 FPS 下 200 ms）。

### 3.2 深度有效窗恢复（修 F3）

origin：`0.2 < z_cam < 1.5`（`data_process_pcd.py:111`）。realtime 路径只查
`depth > 0`。恢复 origin 值为默认，进 config（§6）。

### 3.3 warmup 拼接帧豁免

chunk 0 的 frame 0→1 是 warmup 拼接（源帧跨秒级间隔，见 pipeline.md 的
hold-still 约定）。该转移的运动步长跨越数十秒却会被按单帧阈值测试。规则：
`_trim_warmup_delayed_rows` 检出 `warmup_row` 时，chunk 0 的第一个运动步
（frame 0→1）**豁免运动门控**（motions_valid 按 vis 计算、不参与共识测试
也不作为邻居证据），并且不计入 §4.1 的选择判据。手不动约定下该步位移
本应≈0，豁免只防 tracker 漂移误杀。

## 4. acausal 判据的窗口化翻译（G2/G3/G4）

### 4.1 G4：handle 选择 = "chunk-0 窗口内的 origin 选择 + 幸存者下限"

**现状核对（rev1 此处误述）**：现实现的 chunk-0 选择**已经**近似全窗口
门——`controller_visibilities` 进 selector 前已是 `measurement_valid`
（`phystwin_strict_product.py:502,518`），`once_false_mask=True` 的运动过滤
已含窗口内 `np.prod` 与永久级联（`:545,553-583`），`_selectable_candidate_mask`
要求全窗非零（`:1140-1143`）。因此本节的真正增量不是"加严门"，而是：

1. **G1 补深度窗**（§3.2）与拼接帧豁免（§3.3）后的门。
2. **幸存者下限与 defer-freeze（修当前就存在的崩溃）**：幸存者 <30 时现
   实现直接 `RuntimeError`（`_farthest_point_sample_indices`,
   `phystwin_strict_product.py:626-627`），无捕获，整个 demo 会话被杀死
   （`chunk_data_stream.py` 行循环无 try/except）。而全窗口有效门在
   TAPNext++ 可见性抖动下可能只剩个位数甚至 0 个幸存者（评审用仓库自身
   过滤器实测：54% 逐帧可见性下 2000 候选仅 2 个全窗幸存；90% 可见性下
   min_neighbors 级联可归零）。注意 `_row_ready_for_realtime_chunk_start`
   的 `controller_point_count >= 30` 门数的是**稠密 PCD 像素数**，防不住
   这个。规则：
   - 设 `controller_min_handles`（默认 10）。幸存者 ≥ 下限 → 冻结
     `min(30, 幸存者数)` 列。
   - 幸存者 < 下限 → **defer-freeze**：本窗口不冻结、不发布，选择顺延到
     下一窗口重试；重试次数进 manifest；连续 `controller_freeze_max_retries`
     （默认 3）个窗口失败 → 显式报错退出（可操作的错误信息：检查遮挡/
     mask/tracker）,而不是发布一个 0 列拓扑把整个会话废掉。
   - 列数一经冻结不变（保持 design_spec 不变量 #6），`controller_handle_count`
     进 query schema 与 manifest。

- warmup 手不动使 chunk 0 是最干净的窗口；运动共识在静态场景下 trivially
  通过（比较的是邻居间运动差），可见性与邻域密度才是 binding 门。

### 4.2 G3：全程可见 → "窗口复验 + latch 携带 + 身份重捕获门"

**诚实声明**：origin 的 G3 表达的是**整条轨迹的身份不信任**（含追溯删除）。
逐帧掩码表达不了追溯语义——已发布的 chunk 不能收回。v2 的保证弱于
origin，弱化点包括：(a) origin 会因第 t 帧失败而否定 t 之前已发布的帧，
v2 不能；(b) 可见性中断后 tracker 重锁到**错误物理点**上的"真实测量"，
origin 用全程掩码整列排除，v2 只能用下述重捕获门近似。

三层机制：

1. **窗口内**：`once_false_mask=True` 的运动级联照旧（窗口版 G3）。
2. **跨窗口 latch 携带（修 rev1 未定义项 + G2/G3 耦合）**：每个 handle
   （及其 bundle 邻居 query）维护跨窗口的 `distrust_latch`。窗口内任何
   运动共识失败置位；置位的 query 在后续窗口**不得作为邻域共识的证据**
   （压回 §1 所述"窗口池是 origin 超集"的宽松偏差），其自身测量走第 3 条
   的重捕获门后方可复位。latch 状态随 selector 的 session 状态持久化。
3. **身份重捕获门**：handle 的 `measurement_valid` 在可见性中断（或
   latch 置位）后不得立即恢复 True；需连续 `reacquire_frames_k`（默认 3）
   帧满足 G1 **且**落在 bundle 预测位置的残差带内（复用现有
   0.015 m 带；bundle 不可用时用 last-valid 位置 + 每帧
   `reacquire_drift_mps × dt` 的扩张带），才恢复 valid 并复位 latch。
   重捕获期间的帧 `measurement_valid = False`（值照 §4.4 填充）。

### 4.3 掩码分层（对齐 origin 的层次，rev1 定义不一致已修正）

origin 的分层：mask/depth 门只写 `visibility`，运动共识只写
`motions_valid`/全程掩码，互不混淆。v2 同构：

| 键 | 定义 | 因果性 |
|----|------|--------|
| `controller_measurement_valid[t,j]` | **仅 G1**：raw visible ∧ processed mask ∧ depth 窗 ∧ 重捕获门已通过（§4.2.3） | 逐帧（重捕获门含短历史） |
| `controller_motions_valid[t,j]` | `mv[t] ∧ mv[t+1] ∧ 窗口运动共识`（origin 公式，借帧补末帧） | +1 帧（物化时已有） |

**消费策略（显式声明）**：trainer 的边界条件门 = `measurement_valid[t,j]
∧ motions_valid[t,j]`（§7.3）。origin 发布的 controller 值恰好通过全部
两层，此策略与 origin 的消费面等价；单独用 measurement_valid 会让漂移
测量驱动弹簧，单独用 motions_valid 语义混层。

### 4.4 数值列填充语义

掩码 False 的帧，`controller_points[t,j]` 填 **bundle 恢复值，否则
last-valid**，同步发布 `controller_track_mode/confidence` 诊断。航位推算
（`prev + velocity` 外推）**退出填充路径**（实测产生 19.5 m 瞬移；速度
状态仅保留用于 bundle 权重）。消费端以掩码为准，填充值只影响可视化连续性。

## 5. 冻结拓扑 selectors 的语义改造

### 5.1 controller（`StreamingControllerTrackSelector`）

保留：handle 冻结、backup bundle 机制、恢复阶梯**计算**、诊断字段。

| 项 | 现状（rev2 核对后） | v2 |
|----|--------------------|-----|
| 初选候选 | 已近似全窗有效 + 运动级联（§4.1），<30 时 RuntimeError 崩溃 | + 深度窗/拼接豁免；`min(30, 幸存者)`、下限 defer-freeze（§4.1.2） |
| selector 输出 vis/motions | 计算出 `conf≥0.25` 但 **payload 阶段被丢弃**，发布的 motions_valid 是候选轴数组 | handle 轴 `measurement_valid`（§4.3）与 `motions_valid`（origin 公式）进入发布契约（§7.1） |
| 失效帧数值 | 恢复/推算/冻结值，无标记 | bundle 或 last-valid 填充 + 掩码 False（§4.4） |
| 航位推算 | 参与填充，conf 0.1 | 退出填充路径 |
| 重捕获 | 直接恢复 direct_valid | 重捕获门 + latch（§4.2） |

### 5.2 object（`StreamingObjectTrackSelector`）

origin 的 object 侧不要求全程可见——列保留、逐帧 vis/motions 表达真相。
改造点（修 F8）：

| 项 | 现状 | v2 |
|----|------|----|
| direct 匹配前提 | 锚点在 chunk 首帧有效，否则整窗丢弃 | 逐帧独立（origin 语义，`data_process_track.py:74-94`） |
| revive 值 | vis/motions 强制 True | 值照写（连续性），vis/motions 如实 False；`object_track_status` 保留 `revived` |
| hold-last | 同上 | 同上，掩码 False |

### 5.3 mask 门语义（决策点，rev1 表述已修正）

origin 的精确语义：object/controller mask **互不相减**（重叠像素两类都
建轨迹），但两类共享同一 `visibility` 数组——重叠轨迹的逐帧存活要求
**同时**在两类 mask 内（AND，`data_process_track.py:79-94` 顺序回写）。
现状 realtime 是 `obj & ~ctrl`（相减）+ OR 门。三个候选：相减（现状）/
不减+OR / 不减+AND（真 origin）。抓握区 object 轨迹存活率的提升幅度在
AND 语义下小于 rev1 的预估。**由 §8 harness 的 A/B 指标定夺，默认候选为
不减+AND（对标优先）。** PCD 渲染路径的相减是独立显示问题，不随本决策变。

### 5.4 贴地钳制（rev1 遗漏，修流式路径缺失）

origin 对 final_data 的 object 点做全帧贴地钳制
（`data_process_sample.py:63`：`z>0 → 0`）。warmup 路径已复刻
（`shape_prior_sample.py:98`），**流式 chunk 路径完全缺失**。Stage B4 的
payload 组装加入 object 点逐帧钳制（仅 object，controller 不钳制，与
origin 一致）。

## 6. 阈值物理单位化（兼容故意的 5 FPS）

**换算规则**：逐步长换算 `threshold_step = velocity_mps × dt`，`dt` 取
相邻行 `source_timestamp_s` 实测差（钳制在 `[0.5, 2] × 1/fps` 内防抖），
不用名义 fps（live 采集按墙钟决帧，行距有抖动）。

| 常数 | origin 值 | 物理化 | 备注 |
|------|-----------|--------|------|
| motion_similarity | `neighbor_dist/2` = 0.005 m/帧 @30FPS | **0.15 m/s** | origin 中它是 r/2 的**导出量**，不是独立常数；×dt 缩放会破坏"带宽 ≤ 半径"的耦合（5 FPS 下 0.03 > 0.01 半径），故 band 与 radius **成对**进 §8 扫描，不单独定值 |
| neighbor_dist | 0.01 m | 空间量，不随 fps | 与 band 成对扫描 |
| min_neighbors | 5（**计数含查询点自身**，实效 4 邻居；且自身运动差=0 恒投"同意"票） | 密度阈值 | 密度基线要按类别、按 attrition 后的实际池子算（origin 的 controller 池随时间萎缩），不能用静态 15000 估；候选值 {3,4,5} 进 §8 扫描 |
| 恢复残差带 0.015/0.03 m | — | 空间量 | 不变；复用于 §4.2 重捕获门 |
| bundle 半径 0.03 m | — | 空间量 | 不变 |

config 新增 `tracking` 节（`default.yaml`；注意
`tests/test_demo_v5_1_default_config.py` 的 `EXPECTED_CONFIG_SECTIONS` 断言
**精确且顺序敏感**，需同位插入）：

```yaml
tracking:
  motion_similarity_mps: 0.15
  motion_neighbor_radius_m: 0.01
  motion_min_neighbors: 5
  depth_valid_range_m: [0.2, 1.5]
  controller_handle_count: 30
  controller_min_handles: 10
  controller_freeze_max_retries: 3
  reacquire_frames_k: 3
```

**同步接线（rev1 遗漏）**：`DATA_PROCESS_SAM3D_METRICS`
（`chunk_data_payload.py:87-113`）把这些常数硬编码进每个 chunk manifest，
配置化后必须参数化该 dict，否则 manifest 撒谎；`controller_handle_count`
的硬编码调用点：`chunk_data_stream.py:937,1071`（`select_final_controller_points(count=30)`）
与 `:1351,1506`（selector 构造）。

## 7. 契约变更

### 7.1 生产端（final_data / chunk / aggregate）

**命名冲突（rev1 未察觉，必须先解决）**：`controller_measurement_valid`
已存在——候选轴 (T×N候选) 诊断键（`phystwin_strict_product.py:523`，经
`CONTROLLER_CANDIDATE_TIME_KEYS`（`chunk_data_payload.py:69`）入
track_process）。而 `commit_chunk_data` 把 final_data 与诊断合并为一个
dict（`chunk_data_output.py:344-347`，同名时诊断覆盖 final_data）。规则：
**候选轴诊断键整组改名 `controller_candidate_*` 前缀**（codebase 已有此
前缀先例），handle 轴新键使用原名：

| 键 | 形状 | 必需性 | 语义 |
|----|------|--------|------|
| `controller_measurement_valid` | (T, M_handle) bool | 必需（schema v2 起） | §4.3 |
| `controller_motions_valid` | (T, M_handle) bool | 必需（schema v2 起） | §4.3；候选轴同名键改前缀 |
| `controller_track_confidence` | (T, M_handle) f32 | 可选诊断 | 恢复阶梯置信度 |

**强制执行（rev1 缺失）**：`data_keys.REQUIRED_TIME_KEYS` 目前无任何
执行点（writer 对全部时间键 skip-if-None），且可选键"部分 chunk 有"会
造成 aggregate **静默错位拼接**。规则：`commit_chunk_data_record` 校验
required 键存在；optional 键做会话级 all-or-nothing（首 chunk 出现即全程
必须出现，仿 reader 的 asap 检查）。

**接线清单**（rev1 清单 + 评审补全）：
`chunk_data_payload.py`（`_final_data_payload:416`、候选键改名 `:65-71,320-334`、
`DATA_PROCESS_SAM3D_METRICS:87-113`、schema 版本 `:15`）、
`demo_v5_1/data_keys.py`、`chunk_data_output.py`（分区 + 强制执行 + aggregate）、
`phystwin_strict_product.py`（`:523,863` 候选键改名；`:31-38` 发布集）、
`visualize_track.py:1250,1502`（改读发布掩码，**键缺失时回退 isfinite**，
保证 pre-v2 chunk 可视化不破）、
`realtime_phystwin/scripts/fake_online_tracker.py:26-33,127-140`（其自有
TIME_KEYS 闭集需加新键，否则 §8 harness 的重放侧静默丢掉掩码）。

`query_schema_version` 升 `data_process_sam3d_realtime_query_schema_v2`。

### 7.2 reader 侧收敛（修现存断裂，且不破 demo_v4 重放）

现状：`online_stream.py` 要求 `topology_version == "demo_v4_session_topology_v1"`
/`topology_hash`；v5.1 生产端发 `query_schema_*`（legacy-key 测试禁止 v5.1
用 topology 命名）——v5.1 chunk 今天 reader 读不了。但**单向改名会破坏
demo_v4 假重放**：`fake_online_tracker.py:37-44,93-94` 合成 topology_* 且
`realtime_phystwin/tests/test_online_topology_contract.py` 钉死该契约。

规则：reader **双接受**——按 chunk 中存在的版本键分流：
`query_schema_version`（v2 起）走新校验（必需键含两个掩码，按版本门控，
pre-v2 录制回放不要求新键）；`topology_version` 走 legacy 校验（demo_v4
路径不变）。`TIME_KEYS` 闭集加两个掩码键（bool，`sync_to_device` 转 bool
tensor）。topology 契约测试补双接受用例。

### 7.3 trainer 侧掩码消费（真正闭环 F1）

controller 是**边界条件**不是 loss 目标。门 = `measurement_valid ∧
motions_valid`（§4.3）。接入点**不止一处**（rev1 只列了一处）：

- `spring_mass_warp.set_controller_target`（`:806-820`）与
  **`spring_mass_warp_batched.set_controller_target`（`:1103`）**——batched
  trainer 路径用的是后者，掩码门要实现两遍。
- `cma_optimize_warp.py:1242,1450`（`optimize_online_cma` 消费同一批
  chunk）与推理路径 `rollout_zero_order_params.py:116`、
  `run_inference_by_checkpoint.py:73`，掩码随 controller_points 一起
  batch/切段（`trainer_warp.py:1141-1154,1305-1355`；
  `cma_optimize_warp.py:103,291-292`）。
- 语义：invalid 帧 handle 的 target 保持上一有效 target（持久 held-target
  缓冲 + masked-copy kernel；`set_controller_target` 在 CUDA graph 捕获区
  **之外**（`spring_mass_warp.py:769-802` 只捕获 step/loss），无 kernel
  签名障碍）。`_slice_time_with_padding` 的末帧重复填充与 hold-last 语义
  兼容。
- 与 object 侧对照：object 用 `motions_valid` 门 loss，controller 用双层
  掩码门边界条件。

### 7.4 design_spec 不变量修订

- 保持：#1 #2 #3 #4 #6 #7 #9 #10。
- 修订 #5：`query_schema_version` 升 v2；hash 覆盖字段增加
  `controller_handle_count`。
- 修订 #8：同一 controller 列对应同一冻结 handle；丢失帧只能以 bundle
  恢复值或 last-valid 填充**并以掩码 False 标记**；重捕获须过 §4.2 门；
  禁止把恢复/推算值标为有效测量；禁止重新 FPS 选择。

## 8. 对标验收 harness

新增 `scripts/harness/diagnostics/track/compare_track_origin_parity.py`：
同一段录制，A = origin 离线链（以相同 5 FPS 子采样重跑），B = demo v5.1
fake-live → aggregate `final_data.pkl`。A 侧与 B 侧同样做贴地钳制
（§5.4）后再比几何。

| 维度 | 指标 | 初始验收线 |
|------|------|-----------|
| controller 真实性 | 相邻 valid 帧最大单步位移 | B ≤ max(A 的 p99, 0.05 m)；无米级瞬移 |
| controller 覆盖 | valid 帧占比（对齐窗口内） | B ≥ 0.9 × A |
| **controller 身份**（rev2 新增） | B 在"origin 删除列"上标 valid 的帧数（惩罚项） | ≤ 总 valid 帧的 2% |
| **逐帧判定一致率**（rev2 新增） | A/B 对同一 (t,j) 的 motion-valid 判定一致率 | ≥ 90% |
| controller 轨迹 | frame-0 最近邻配对逐帧 RMSE | 中位数 < 0.01 m |
| object 覆盖 | vis / motions_valid 占比 | B ≥ 0.85 × A |
| object 几何 | 逐帧 valid 点集 chamfer(A,B) | 中位数 < 0.01 m |
| 时间轴 | 发布帧数、间隙数 | 无隐藏间隙；帧数 = 物化窗口总帧数（含终态 flush 窗口） |
| **计算预算**（rev2 新增） | Stage B 每窗墙钟 | ≤ 2 s @ 10k queries（实测基线：0.5 s @ 5k、1.0 s @ 10k 的运动过滤） |

阈值扫描：`(motion_similarity_mps, neighbor_radius)` **成对**、
`min_neighbors ∈ {3,4,5}`、query count ∈ {5000, 10000}、§5.3 三种 mask
语义、`reacquire_frames_k ∈ {2,3,5}`。产出 parity 报告存
`docs/generated/`。

## 9. 实施阶段

- **P0 前置**：warning-only 收尾。**注意当前工作区处于半迁移且不可运行
  状态**（`chunk_data_stream.py:1346,1501` 仍向已删参数的 `ChunkDataWriter`
  传 `allow_degraded`，`test_demo_v5_1_chunk_data` 现 2/7 失败）。残余清
  单：`_track_process_invalid:714-715`、
  `_track_process_online_publish_skip_reason:718-728`、skip 分支
  `:1234-1255`、`allow_degraded_online` 参数与透传
  `:1118,1236,1303,1346,1398,1467,1501,1572`、`on_chunk_written` 门
  `:1405-1409,1579-1583`、invalid `break:1410-1411,1584-1585,1592-1593`。
- **P1 发布真相**：候选键改名 + §7.1 新键接线 + 强制执行 + §5.1/5.2 语义
  + §5.4 钳制 + §4.1 幸存者下限/defer-freeze（这个崩溃今天就存在，提前到
  P1）。先补 selector 单测网（目前**零单测**）再动实现。
- **P2 消费闭环**：§7.2 reader 双接受 + §7.3 两处 set_controller_target
  与 CMA/推理路径门控。
- **P3 windowed origin 门**：§3.1 借帧物化（含终态 flush、切片不变量、
  删 `out[-1]=out[-2]`）、§3.2 深度窗、§3.3 拼接豁免、§4.2 latch 携带与
  重捕获门、§6 速度单位化（含 metrics dict 参数化）。
- **P4 对标定值**：§8 harness，真实录制，扫描定值，parity 报告。
- **P5 可选**：handle 衰减重播种（schema 再升级）；tracker 后端升级（§10）。

## 10. tracker 后端升级路径（独立决策，修 F9 的通道）

origin 用 `cotracker3_online`：流式、16 帧窗/8 帧步长，**内部工作分辨率
384×512**（`predictor.py` 将输入重采样至 `model_resolution`）。demo 用
TAPNext++ 256×256 fp16（实测可见性 54%、0.27 m 重捕获跳变）。有效分辨率
差距约 3×（384×512 vs 256²），不是 rev1 所称"全分辨率 vs 256²"。因为
origin 的 tracker 是流式的，CoTracker3-online 做 realtime 架构可行，瓶颈
是算力预算：

- adapter 可插拔（`PointTrackerAdapterConfig`）；新 backend 跑 GPU 1
  （shape-prior warmup 后基本空闲）。
- 吞吐需实测（384×512 × 5000 query @ 5 FPS）；不达标则降 query 或保留
  TAPNext++。
- 决策依据：P4 harness 加 tracker 后端对比列。Stage B 与后端正交。

## 11. 已否决的替代方案

- **每 chunk 独立跑 origin 选择**：列身份跨 chunk 漂移（design_spec 已论证）。
- **合成值标 valid（现状）**：origin 保证的直接否定，实测 19.5 m 瞬移入训练。
- **degraded 整块丢弃（旧状）**：隐藏时间洞；warning-only 方向正确，但必
  须与 P1/P2 掩码闭环同期落地（顺序依赖）。
- **生产端改用 topology_* 命名**：legacy-key 测试禁止；reader 双接受。
- **推算值参与填充**：实测米级瞬移；填充仅 bundle 恢复 / last-valid。
- **单向改名 reader**：破坏 demo_v4 假重放与其契约测试；双接受替代。

## 12. 评审记录（rev1 → rev2）

三路对抗评审共确认 17 个问题，全部合入：

- origin 对标忠实度（8 项）：G2/G3 耦合误分类（§1）、§4.2 夸大等价（改
  为显式弱化 + 重捕获门 + latch 携带 + 身份惩罚指标）、掩码分层不一致
  （§4.3）、贴地钳制遗漏（§5.4）、r/2 导出关系与成对扫描（§6）、
  min_neighbors 含自身（§6）、重叠 mask 实为 AND（§5.3）、CoTracker3
  内部分辨率 384×512（§10）。
- realtime 可行性（6 项）：借帧终态 flush 缺失（§3.1.1）、chunk-0 零幸存
  者状态与现存崩溃（§4.1.2）、借帧切片/选择器状态污染（§3.1.2）、
  `out[-1]=out[-2]` 冲突（§3.1.3）、warmup 拼接帧与时间抖动（§3.3/§6）、
  计算预算入 harness（§8）。
- 契约迁移（含 3 项阻塞）：P0 半迁移现状与残余清单（§9）、
  `controller_measurement_valid` 命名冲突（§7.1）、reader 单向改名破坏
  demo_v4（§7.2）、双 `set_controller_target` 与 CMA/推理路径（§7.3）、
  REQUIRED 键无执行点与 aggregate 静默错位（§7.1）、metrics dict 硬编码
  （§6）、fake tracker TIME_KEYS（§7.1）、visualize 回退（§7.1）、config
  节顺序敏感（§6）。
