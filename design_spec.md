`data_process_sam3d` 的核心逻辑是：**query 初始化比较宽，offline 后处理非常严格，最后只从全序列可靠 controller tracks 里 FPS 选 30 个。**

**1. Query Points 怎么来**
在 [dense_track.py](/home/xinjie/single_proj_qqtt/data_process_sam3d/dense_track.py:66)：

- 只看 **first frame**。
- 读取该 camera 下 `mask/{cam}/*/0.png` 的所有 semantic masks。
- 把所有 mask 做 union。
- 从 union mask 的像素里取 query pixels。
- 坐标变成 `[t=0, x, y]`。
- 每个 camera 最多随机采样 `5000` 个 query。
- 然后交给 online CoTracker 跟踪。
- 保存时把 tracker 输出从 `xy` 翻成 `yx`，存到 `cotracker/{cam}.npz` 里。

也就是说，原始 query 初始化不是从 processed mask 来的，而是从 first-frame raw semantic union 来的。

**2. 第一层过滤：PCD depth valid**
在 [data_process_pcd.py](/home/xinjie/single_proj_qqtt/data_process_sam3d/data_process_pcd.py:97)：

- 每帧 depth lift 成 per-pixel 3D points。
- depth 有效条件是 camera-frame `z > 0.2` 且 `z < 1.5`。
- 输出 `points`, `colors`, `masks`。
- 这里的 `masks` 是 depth/range valid，不是 semantic mask。

**3. 第二层过滤：processed semantic mask**
在 [data_process_mask.py](/home/xinjie/single_proj_qqtt/data_process_sam3d/data_process_mask.py:42)：

- object/controller raw semantic mask 先和 depth-valid `masks[i]` 相与。
- 多 camera 的 object/controller points 合并成 point cloud。
- 对 object 和 controller 分别做 Open3D radius outlier removal：
  - `nb_points=40`
  - `radius=0.01`
- 被判成 outlier 的 3D 点对应像素会从 mask 里删掉。
- 最终保存 `mask/processed_masks.pkl`。

所以 processed mask = **semantic mask ∩ valid depth mask ∩ radius-outlier-cleaned mask**。

**4. 第三层过滤：tracking semantic validity**
在 [data_process_track.py](/home/xinjie/single_proj_qqtt/data_process_sam3d/data_process_track.py:37)：

- 先用 first frame processed mask 给每个 query 定身份：
  - first frame 在 object processed mask 里，就是 object track。
  - first frame 在 controller processed mask 里，就是 controller track。
- 后续每帧：
  - object track 如果当前像素不在当前 frame object processed mask，`visibility[t, q] = 0`。
  - controller track 如果当前像素不在当前 frame controller processed mask，`visibility[t, q] = 0`。
- visible 的点才会从 per-pixel PCD 里 lift 成 3D。
- invisible 的 3D point 保持 zero。

这一步已经把 tracker raw visible 和 processed semantic/depth validity 混在一起了：只要当前帧跑出 processed mask，就被当成不可见。

**5. 第四层过滤：motion consistency**
在 [data_process_track.py](/home/xinjie/single_proj_qqtt/data_process_sam3d/data_process_track.py:138)：

object 和 controller 都先算：

```text
motion[t] = point[t+1] - point[t]
motion_valid[t] = visibility[t] & visibility[t+1]
```

然后每个点找当前帧 3D 邻居：

```text
neighbor_dist = 0.01m
min neighbors = 5
motion similarity threshold = neighbor_dist / 2 = 0.005m
```

如果邻居太少，或者和多数邻居运动不一致，就把该 transition 判 invalid。

**controller 比 object 更严格。**

controller 会先做：

```python
mask = np.prod(controller_visibilities, axis=0)
```

也就是：**controller candidate 必须全序列每一帧都 visible**。任何一帧 semantic/depth/visibility 失败，这个 controller candidate 就从最终候选里全局删除。

然后 motion filter 里如果某个 controller 点某帧失败，也会：

```python
mask[j] = 0
```

所以 controller 是 once-fail-kill-track 的 offline 筛选。

**6. 最终 30 个 controller points 怎么选**
在 [data_process_track.py](/home/xinjie/single_proj_qqtt/data_process_sam3d/data_process_track.py:325)：

- 只取 `controller_mask == 1` 的 surviving controller tracks。
- 要求 surviving controller tracks 至少 30 个。
- 在 first-frame 3D positions 上做 Open3D `farthest_point_down_sample(30)`。
- 用 FPS 选出的 30 个 indices 保留完整时间轨迹：

```text
controller_points = T x 30 x 3
```

原版这里没有 recovery、没有 hold、没有重选。失败 track 在进入最终 30 之前已经被删掉了。

**一句话总结**
原始 `data_process_sam3d` 是 offline strict pipeline：

```text
first-frame raw mask union -> up to 5000 query points
-> tracker raw tracks
-> processed mask/depth/outlier filtering
-> per-frame semantic visibility
-> local 3D motion consistency
-> controller 全序列 surviving mask
-> first-frame FPS 选 30 个 controller targets
```

它的前提是：offline 全序列已经看完，可以先删掉坏 tracks，再从剩下的可靠 tracks 里选 30 个。因此原版的 `controller_points` 不是“30 个 query id 里有些丢了再补”，而是“已经离线筛过的一组全序列可靠 kinematic targets”。

---

# Demo 5：query points / object points / controller points 处理对比

这里专门比较三类点的处理，不讲宽泛架构。

原版 `data_process_sam3d` 是 offline pipeline。它能先看完整序列，再决定哪些 query 可靠、哪些 object points 保留、哪些 controller tracks 最终能进 30 个 controller targets。

Demo 5 / 我们实时版不具备这个前提。它必须边录边按 chunk 发布 `final_data.pkl`，所以它不能等完整序列结束后再筛点。它要先建立稳定 query schema，再让 object points 和 controller points 在后续 chunks 里保持同一套列拓扑。丢失时可以恢复、预测或标记质量，但不能随意换一批新点。

## 1. Query points

原版的 query points 来自 first frame 的 raw semantic masks。它读取 `mask/{cam}/*/0.png`，把 object/controller 等 semantic masks 做 union，然后从 union mask 里最多采样 5000 个 query。初始化坐标是 `[t=0, x, y]`，CoTracker 输出以后再把轨迹保存成 `yx` 格式到 `cotracker/{cam}.npz`。

这里 query points 一开始只是一个 raw union 候选池。它们还不是最终 object points 或 controller points。后面 `data_process_track.py` 会用 first-frame processed mask 给 query 定身份：落在 object processed mask 里就是 object track，落在 controller processed mask 里就是 controller track。因为原版是 offline，它可以看完整序列以后再删除不可靠 tracks；最终产品不需要承诺每个原始 query 都有稳定在线身份。

Demo 5 保留 first-frame union query 的思想，但它的约束更强。单相机 strict product 里用 object/controller union mask 采样，最多同样是 5000 个 query，常量是 `PHYSTWIN_DENSE_QUERY_POINTS = 5000`。运行时主格式是 `query_points_yx` 和 `tracks_yx`；写兼容 artifact 时再生成 `queries_txy`，从而保持 `cotracker/0.npz` / `tracking/0.npz` 兼容。

Demo 5 的关键变化是 query schema 必须跨 chunk 固定。第一 chunk 建立 `query_ids` 和 `query_semantic_labels`，后续 chunk 只能复用同一套 query ids 和 semantic labels，不能重新解释 query 身份。`query_schema_hash` 由 `query_ids`, `query_semantic_labels`, `object_sample_query_ids`, `controller_sample_query_ids` 计算，online chunks 必须通过这个 hash 保证拓扑连续。我们要保证我们第一个chunk得到的anchor points还在后面，这个没有问题。

所以 query points 的一句话区别是：原版把 query points 当作离线候选池；Demo 5 把 query points 当作在线拓扑基底。后续所有 object/controller sample 都必须引用稳定 query id。

Demo 5 里和 query schema 直接相关的字段是：

```text
query_ids
query_semantic_labels
query_schema_version = data_process_sam3d_realtime_query_schema_v1
query_schema_hash
object_sample_query_ids
controller_sample_query_ids
```

## 2. Object points

原版的 object points 来自 first frame 被判成 object 的 query tracks。每一帧里，一个 object query 只有同时满足 tracker visible、当前像素仍在 processed object mask 里、depth 有效，才会被 lift 成 3D object point。随后原版会做 local 3D motion consistency：邻域半径 0.01m，至少 5 个邻居，motion similarity threshold 是 0.005m。object 的 motion filter 比 controller 宽松；它不会像 controller 那样要求整段序列每一帧都 visible。

原版最终会对 object points 做 0.005m volume/grid 风格采样，得到压缩后的 `object_points`。因为它是 offline，所以它可以先看完整序列、先过滤，再决定最终 object sample。失败或不稳定的 object tracks 可以直接不进最终产品。

Demo 5 的前半段和原版保持一致：先通过 first-frame semantic label 找 object query candidates，再用 tracker visible、processed object mask、depth valid 得到每帧 object candidate 3D points，也跑同样的 chunk 内 motion filter。区别从采样开始。Demo 5 不能每个 chunk 都重采样一套新的 object points，否则 online optimizer 看到的 tensor 列身份会一直变。

因此 Demo 5 在第一 chunk 用 `StreamingObjectTrackSelector` 做一次 0.005m volume sample，并冻结这一批 `object_sample_query_ids`。这个 sample 的空间边界还会参考 `surface_points` / `interior_points`，让 object sample 和 shape prior 的空间范围一致。后续 chunks 继续输出同样列数、同样 query id 对应的 object columns。

后续 chunk 里，如果某个 object query 还能 direct 命中，就直接写它的轨迹。如果 direct 失败，Demo 5 会先尝试用邻近 direct object anchors 做 motion revive。这个 revive 用上一 chunk 的 last points 和当前 chunk 的 direct anchors 做局部运动插值，默认半径 0.011m，至少 2 个邻居，最多 4 个邻居。revive 成功时，该列 status 写成 `revived`。

如果 revive 也失败，Demo 5 不会换一个新的 physical query 填进这列。它会保留该 object column，用 last finite point/color 做占位，把 active query index 写成 `-1`，visibility/motion-valid 保持 false，status 写成 `missing`。这个占位不是说点真实可见，而是为了保持 online `object_points` 的 shape 和列身份稳定。

object points 的处理可以概括为：

```text
原版：object candidates -> offline filter -> 0.005m sample -> final object_points

Demo 5：object candidates -> chunk filter
      -> first chunk 0.005m sample and freeze object_sample_query_ids
      -> later chunks direct / revive / hold-missing
      -> keep object_points column topology stable
```

Demo 5 object trace 里最重要的字段是：

```text
object_points
object_colors
object_visibilities
object_motions_valid
object_sample_indices
object_selected_query_ids
object_sample_query_ids
object_track_query_indices
object_track_active_query_indices
object_track_status  # direct / revived / missing
```

## 3. Controller points

原版的 controller points 更严格。controller candidates 同样来自 first-frame processed mask 里被判为 controller 的 query tracks，但它要求每帧都能通过 tracker visible、processed controller mask、depth valid。随后它先做：

```text
controller_mask = np.prod(controller_visibilities, axis=0)
```

这意味着任何一帧不可见，这个 controller candidate 就会被全局删除。motion filter 里如果某个 controller 点某一帧失败，也会继续 kill 这个 candidate。最后，原版只从完整序列 surviving controller tracks 里按 first-frame 3D position 做 FPS，选出 30 个 `controller_points`。因此原版最终的 `controller_points = T x 30 x 3` 本质上是一批已经离线证明全序列可靠的 kinematic targets。

Demo 5 前面也跑 strict controller filter。它会得到 controller candidates、`controller_mask`、`controller_motions_valid`，并保留更细的诊断量：raw points、raw visibility、processed mask valid、depth valid、measurement valid。保留这些量是为了后续解释每个 anchor 为什么 direct 成功、为什么失败、能不能恢复。

Demo 5 的第一 chunk 会从 chunk 内 surviving / finite controller candidates 里 FPS 选 30 个，并把它们冻结为 `controller_sample_query_ids` / `controller_track_query_indices`。这一步对应原版的 FPS 选 30，但语义不同：原版是在完整序列筛完之后选；Demo 5 是在 online 开始时先选，然后后续负责维护这 30 列。

后续 chunks 里，Demo 5 不重新 FPS 选 30 个，也不拿新 query 替换旧 anchor。每个 controller column 先找原始 query id 的 direct observation。direct 必须同时通过 raw visible、depth valid、processed mask valid、measurement valid、motion valid。通过时写 `controller_track_mode = direct_valid`，confidence 是 1.0。

如果 primary raw point 可用，但 processed mask reject 或 motion consistency reject，Demo 5 不会立刻丢掉。它会拿 primary raw point 和 backup bundle 预测点比较；如果二者残差 <= 0.015m，就接受 raw point，写成 `mask_reject_primary_raw_accepted` 或 `motion_reject_residual_ok`。如果残差太大，就改用 bundle recovery。

backup bundle 是 Demo 5 为 streaming controller points 加的核心补偿。第一 chunk 里，每个 anchor 会在 0.03m 半径内找最多 12 个附近 backup query，至少需要 4 个候选才建立 bundle。后续某帧 direct 丢失时，如果 bundle 里有 >=3 个 support，就做 rigid recovery；如果只有 >=2 个 support，就做 translation recovery。recovery 成功时写对应的 `*_bundle_recovered` mode，并记录 support count、source query id 和 residual。

如果 tracker/depth/mask/motion 都无法提供可信恢复，Demo 5 仍然不换点。它会用 `previous_point + previous_velocity` 做预测占位，confidence=0.1，source query id 写 `-1`，visibility/motion-valid 为 false。这个 column 会被标成 missing 或 unrecoverable，后续质量状态可能变成 degraded/invalid。

Demo 5 的质量判断直接来自 confidence。mean confidence < 0.45、low-confidence ratio > 0.40，或者某个 anchor 连续 3 个 chunks 不可靠，会把 `track_process_status` 标成 `invalid`。这和原版不同：原版通常是在离线阶段直接失败或不给这个点；Demo 5 要把失败显式写进 online trace，让 consumer 知道这 30 列里哪些是 direct、哪些是 recovered、哪些只是预测占位。

controller points 的处理可以概括为：

```text
原版：controller candidates
    -> 全序列 visible
    -> 全序列 motion consistent
    -> surviving tracks
    -> FPS 选 30
    -> controller_points = T x 30 x 3

Demo 5：controller candidates per chunk
    -> chunk 内 strict visible/motion filter
    -> first chunk FPS 选 30 and freeze controller_sample_query_ids
    -> later chunks direct / raw-accept / bundle-recover / predict-missing
    -> controller_points 始终保持 T x 30 x 3
    -> 用 confidence/status/mode 解释每列质量
```

Demo 5 controller trace 里最重要的字段是：

```text
controller_points
controller_final_indices
controller_selected_query_ids
controller_sample_query_ids
controller_track_query_indices
controller_track_active_query_indices
controller_track_status       # direct / recovered / missing
controller_track_mode         # direct_valid / *_bundle_recovered / *_unrecoverable ...
controller_track_confidence
controller_filter_reason
controller_source_query_ids
controller_neighbor_query_ids
controller_neighbor_support_count
controller_neighbor_fit_residual
track_process_status          # normal / degraded / invalid
```

## 总结

三类点的差别可以压成一句话：

```text
query_points:
  原版是离线候选池；Demo 5 是跨 chunk 固定的在线 query schema。

object_points:
  原版可以离线筛完再采样；Demo 5 第一 chunk 采样后固定 object columns，
  后续只能 direct / revive / missing-hold，不能随意重采样换拓扑。

controller_points:
  原版先用全序列 strict 条件删坏 tracks，再 FPS 选 30；
  Demo 5 先选 30 个在线 anchor，再用 direct / bundle recovery / prediction
  维持这 30 列，并把质量写进 trace。
```
