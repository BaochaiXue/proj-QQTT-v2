# QQTT / FFS / Real-Time PCD 工作整理

这份文档把近期对话整理成一个可执行的信息集合。核心目标是：把 3-camera RealSense 输入稳定地变成可解释、可比较、可用于下游 inverse physics 的 RGB-D / object PCD 数据流。

## 总目标

- 实时链路：3 个 RealSense 相机输入，输出 RGB + depth stream，并重建每帧的 3-view merged 3D object PCD。
- 离线评估：用 static round 1-3 frame 0 做 benchmark，对比 native depth、raw FFS、confidence-filtered FFS 和后处理结果。
- 质量目标：减少 floating points，避免引入无效浮点深度；在没有 floating artifacts 的基础上尽量减少空洞。
- 工程目标：FFS 做成 ready-to-use，代码路径清晰，后续集成和 debug 容易。

## 概念边界

- “完整 PCD”当前指 3 个 camera view 的点云 merge，不包含 3D generative prior。
- 当前阶段不需要处理 generative model、tracking、inverse physics 本身；我们的产物是稳定 RGB-D / PCD 输入。
- 下游 Heqian / inverse physics 会使用我们的 reconstructed PCD，因此我们需要给出清楚的速度、质量、失败模式结论。

## Workstream 1: Real-Time PCD Demo

目标：做一个能展示最终 FPS 的 real-time PCD demo。

需要回答的问题：

- RGB + native depth stream 在 3-camera 下最高稳定 FPS 是多少？
- RGB + IR stream 在 3-camera 下最高稳定 FPS 是多少？
- 实时显示、深度渲染、点云构建、USB/camera 本身分别占多少瓶颈？
- 最终 demo 是展示完整 scene PCD，还是只展示 object-mask PCD？

当前工程结论：

- Viewer 应使用“每相机采集线程 + 最新帧缓冲 + 主线程显示”的结构，避免主线程显示/渲染阻塞采集。
- Native depth viewer 和 FFS viewer 都需要 debug placeholder 模式：depth 区域只显示收到的 depth FPS，不做 colormap / depth render，用来区分采集瓶颈和渲染瓶颈。
- 如果 placeholder 模式能接近 30 FPS，而正常 render 低于 30 FPS，瓶颈主要在代码渲染/显示路径，不是 camera 或 USB。

下一步输出形式：

- 给出 native RGB-D 的采集 FPS、显示 FPS、depth render FPS。
- 给出 RGB+IR 的采集 FPS，作为 FFS 输入上限。
- 给出 real-time PCD 构建 FPS，包括 object-mask PCD 和 full merged PCD 两种。

## Workstream 2: Fast Foundation Stereo

目标：把 FFS 做成 ready-to-use 的 depth backend，并明确速度/质量 tradeoff。

默认配置建议：

- 没有特殊原因不要随意改动核心参数，优先保持 `scale=1.0`、`valid_iters=4` 作为 baseline。
- 需要对 PyTorch、single-engine TensorRT、double-engine TensorRT 做同一配置下的 FPS 对比。
- 需要测试 per-camera workers 和 shared worker / batch inference 的区别。

公开权重候选：

| config | 特点 | 参考速度/显存 |
|---|---|---|
| `23-36-37` | 精度最高，最慢 | 约 `49.4 / 23.4 ms` 或 `41.1 / 18.4 ms`，约 `653 MB` |
| `20-26-39` | 中间档 | 约 `43.6 / 19.4 ms` 或 `37.5 / 16.4 ms`，约 `651 MB` |
| `20-30-48` | 最快，精度最低 | 约 `38.4 / 16.6 ms` 或 `29.3 / 14.0 ms`，约 `646 MB` |

需要持续跟踪的问题：

- 本地编译的 TensorRT engine 只对应一个固定组合：权重 `.pth`、`scale`、`valid_iters`、precision、输入 shape / batch 约束。
- batch=3 编译失败时，需要区分是 dynamic shape / plugin / memory / engine profile 问题，还是 FFS 模型图本身不支持该 batch。
- 如果 confidence filtering 最终有效，TensorRT 版本需要导出 logits 或额外 confidence output，否则只能在 PyTorch 路径里做。

## Workstream 3: Confidence Filtering

目标：用 FFS classifier logits 生成 confidence proxy，过滤低可信 depth pixel，减少 floating artifacts。

当前实现的四种 confidence proxy：

- `margin`: `top1_softmax - top2_softmax`，越大越可信。
- `max_softmax`: 最大 softmax probability，越大越可信。
- `entropy`: 实际实现为 `1 - entropy / log(D)`，是反熵置信度，越大越可信。
- `variance`: disparity-bin softmax 分布的 inverse variance，再做 per-image robust normalization，越大越可信。

当前过滤方向：

- 代码逻辑是保留 `confidence >= threshold`，剔除 `confidence < threshold`。
- 所以报告里应该写“保留 `metric >= T`，过滤 `< T`”，不要写成“`metric >= T`: 过滤 X 点”。

重要结论：

- Logits filtering 没有整体失效；`margin` 和 `variance` 在 absolute threshold sweep 下已经明显生效。
- `max_softmax` 和 `entropy` 在 `0.01` 到 `0.25` 下几乎不动，是因为它们的数值分布本来偏高，不是代码没有执行。
- 如果我们说的 `1%, 5%, 10%...` 是“过滤最低百分比的点”，那当前 absolute threshold sweep 不等价；需要新增 percentile rejection sweep。
- `variance` 是 per-image p05/p95 归一化后的相对 confidence，不是跨相机/跨 round 稳定标定的绝对置信度。

已完成实验：

- 新 workflow：`scripts/harness/visual_compare_ffs_confidence_threshold_sweep_pcd.py`
- 结果目录：`result/ffs_confidence_threshold_sweep_object_pcd_frame_0000_erode1_phystwin/`
- 范围：static `round1/round2/round3`，`frame 0`
- thresholds：`0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50`
- 每个 threshold 一张 `6x3` Open3D panel：
  - rows: native depth, original FFS, margin, max_softmax, entropy, variance
  - columns: 3 个原始 FFS camera views
  - object mask 后再向内 erode `1px`
  - 所有显示结果都经过 PhysTwin-like radius-neighbor postprocess：`radius=0.01m`, `nb_points=40`

下一步建议：

- 保留 absolute threshold sweep，用于观察不同 metric 的自然数值尺度。
- 另做 percentile rejection sweep：过滤最低 `1%, 5%, 10%, 15%, 20%, 25%, 50%` 的 confidence 点。
- 对最终候选 metric 给出结论：是否减少 floating artifacts、是否过度制造 holes、是否稳定跨 round。

## Workstream 4: Depth To 3D PCD

目标：把 native depth 或 FFS depth 映射到可用的 3D object PCD。

当前处理策略：

- 只关注 object：从 object mask 内 unproject 点。
- depth clipping：只保留合理距离范围内的 depth。
- Open3D radius outlier filtering：删除一定半径内缺少邻居的孤立点。
- mask erosion：object mask 边缘向内缩 `1px`，丢弃最外层不稳定 mask/depth 边界点。

需要比较的 demo case：

- Case A: `RGB + native depth -> object PCD`
- Case B: `RGB + IR -> FFS depth -> object PCD`
- Case C: `native + FFS fused depth -> object PCD`

评价指标：

- fused object PCD 是否减少 floating artifacts。
- holes 是否可接受。
- native depth 缺失区域是否能被 FFS 合理补充。
- radius-neighbor postprocess 是否误删真实 object thin parts。

## Workstream 5: Static Benchmark

目标：把 static round 1-3 frame 0 固定为一个可复现实验集。

固定输入：

- `static/native_30_static_round1_20260410_235202`
- `static/ffs_30_static_round1_20260410_235202`
- `static/native_30_static_round2_20260414`
- `static/ffs_30_static_round2_20260414`
- `static/native_30_static_round3_20260414`
- `static/ffs_30_static_round3_20260414`

固定比较维度：

- native depth
- original FFS
- FFS + confidence filtering
- FFS / native / fused PCD after object mask
- PhysTwin-like postprocess 前后点数
- mask erosion 前后像素数

输出要求：

- 每个实验一个 result folder。
- 每个 panel 都要 proper label：round、frame、threshold、mask 是否 erode、是否 postprocess、metric 名称、点数。
- 每个 result folder 都要有 top-level `summary.json`，不要只留下图片。

## Workstream 6: Newton Robot Demo

目标：完成 Newton robot demo，并解释 simulation speed problem。

需要回答的问题：

- 当前 simulation speed 的瓶颈在 physics step、render、data transfer、Python loop，还是同步等待？
- 是否可以通过降低 render frequency、异步数据读取、减少 logging、batching step 等方式改善？
- demo 所需的最低可接受 sim FPS 是多少？

输出要求：

- 不只给 observation，要给结论和解释。
- 每个瓶颈结论后面要有证据：profile 数字、耗时分解或最小复现实验。

## 下周展示结构建议

每个主题按同一结构讲：

1. 问题：我们到底要验证什么。
2. 方法：怎么测，输入数据是什么，固定了哪些参数。
3. 结果：数字和图，不只放截图。
4. 解释：为什么会这样，瓶颈或 artifact 来自哪里。
5. 结论：这个方向是否可用，下一步改什么。

不要只展示 observation。每一点都要收束成一个明确判断：可用、不可用、需要更多实验，或者需要换方案。
