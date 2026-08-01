# gaussian_align_demo

独立离线工具链：把 **TripoSplat 生成的 Gaussian Splat** 对齐到 demo_v6_2 的
**轨迹世界坐标系**（table-world），再沿录制的 `final_data.pkl` 物体轨迹驱动
Gaussian 逐帧变形并渲染视频。只读取 demo_v6_2 的产物，不修改、不接入其在线主链。

```
demo_v6_2 frame-0 case (RGB/mask/PCD/标定)
    → prepare_input      RGBA（alpha=object mask）
    → triposplat_driver  多 seed 生成（triposplat env）
    → seed_gallery       同步 turntable 网格 → 人工选 seed
    → align_gaussian     候选视角渲染 + SuperGlue + RANSAC-Umeyama → Sim(3)
    → refine_alignment   Nelder-Mead 深度锚定精配准
    → animate_trajectory frame-0 gate → 绑定 → 逐帧变形 → mp4
```

## 快速开始

```bash
# demo 环境（demo_2_max）里，仓库根目录：
python gaussian_align_demo/run_pipeline.py \
    --case-dir outputs/shape_prior_case/shape_prior_frame0 \
    --final-data outputs/data/final_data.pkl \
    --run-dir gaussian_align_demo/runs/myrun
# gallery 后会停下等选 seed：看 seed_gallery/seed_comparison_grid.mp4，然后
python -m gaussian_align_demo.seed_gallery --run-dir gaussian_align_demo/runs/myrun --select 1
python gaussian_align_demo/run_pipeline.py ... --stages align,refine,animate
```

环境：生成阶段用 `triposplat` conda env（权重在 `/home/xinjie/TripoSplat/ckpts/`），
其余阶段任意含 torch+gsplat+open3d 的环境（`demo_2_max`）。

## 输入契约（demo_v6_2 只读）

| 文件 | 内容 |
|---|---|
| `case/color/0/0.png` | frame-0 RGB（盘上就是 RGB） |
| `case/mask/0/{0,1}/0.png` | 0=object、1=controller，二值 0/255 |
| `case/pcd/0.npz` | `points (1,H,W,3)` **世界系米制**、`masks` 深度有效位 |
| `case/calibrate.pkl` | `[c2w]`，OpenCV 列向量约定 |
| `case/metadata.json` | `intrinsics=[ [3x3] ]`（无宽高，从 PNG 取） |
| `<base>/data/final_data.pkl` | `object_points (T,N,3)` 等；与 case **同一世界系** |

## 关键设计决定（含踩坑记录）

- **TripoSplat `run()` 同 seed 不可复现**：octree decoder 从全局 RNG 采样。
  driver 走分段 API（encode → sample_latent(seed) → decode），并在每次
  decode 前 `torch.manual_seed(decode_seed)` 固定。
- **TripoSplat PLY 语义**：`scale_*`=log σ、`opacity`=logit、`rot_*`=wxyz、
  保存时已应用 y-up→z-up；坐标是归一化单位盒，**非米制**——所以才需要 Sim(3)。
- **粗对齐用 RANSAC-Umeyama（3D-3D）而非 PnP+scale**：候选渲染有 expected
  depth（canonical 3D），真实侧有米制 PCD（world 3D），两侧都是 3D，闭式
  相似变换一步到位；PnP 重投影误差只作诊断。真实 run 上 8 个独立候选给出的
  scale 聚在 0.455–0.466，互相印证。
- **精配准用 Nelder-Mead 而非 autograd**：gsplat 前向确定、解析梯度数值正确，
  但光栅化 silhouette 的损失面在 <0.1 mm 尺度全是微观毛刺——无穷小梯度与宏观
  坡度方向相反（实测 Adam 单调发散）。宏观初始单纯形（3°/5 mm/2%）直接绕开。
  另外 scipy 对 x0=0 的默认初始单纯形是 2.5e-4，必须显式给 `initial_simplex`。
- **深度是 metric 锚**：mask-only 目标会把偏平物体转 ~29° 去填 2D 轮廓
  （IoU +0.11 但深度 9→51 mm）。深度 Huber 大权重 + 验收 gate 同时卡
  IoU 与深度不恶化 + 位姿增量 ≤3σ。
- **`demo_v6_2.shape_prior.match_pairs` 在 import 时全局关闭 autograd**
  （模块级 `torch.set_grad_enabled(False)`）——精配准阶段绝不 import 它。
- **demo_v6_2 的位姿工具是 PyTorch3D 行向量约定**（`sample_camera_poses` /
  `project_2d_to_3d`），与本工具的 OpenCV 列向量约定不兼容，只复用其
  SuperGlue matcher，不复用位姿数学（`tests/test_base_math.py` 锁投影往返）。
- **gsplat 1.5.3 要传 `packed=False`**：packed 默认路径的 backgrounds 形状
  断言与 (C,3) 输入不兼容（FuturePhysTwin 同样传 False）。
- **动画 = 增量 rollout**（移植 FuturePhysTwin `gs_render_dynamics` 数学）：
  关系图取 frame-0 K=16 近邻；每帧对每个 bone 做邻域 Procrustes（Kabsch +
  反射修正，退化回退 identity，无 ipdb 陷阱）；绑定索引 frame-0 冻结、权值
  逐帧刷新；位置 LBS 混合、bone 四元数半球对齐后加权混合**左乘**到高斯
  四元数（wxyz）。外观（颜色/不透明度/尺度）全程不动。
- **frame-0 一致性 gate**：`object_points[0]` 对 case 物体点云最近邻中位
  >2 cm 或质心 >5 cm 直接拒绝，绝不偷偷再做一次 ICP。

## 产物布局

```
runs/<id>/
├── input/            frame0_rgba.png + manifest
├── seeds/seed_XXX/   gaussian_{65536,262144}.ply + generation.json
├── seed_gallery/     seed_comparison_grid.mp4, turntable_*.mp4, seed_scores.csv
├── selected_seed.json
├── alignment/        sim3_coarse.json, coarse_aligned.ply, coarse_overlay.png,
│                     winner_matches.png, sim3_refined.json, refined_aligned.ply,
│                     refined_overlay.png, refinement_history.jsonl
└── motion/           binding.npz, metrics.json,
                      trajectory_fixed_camera.mp4  [录制RGB | 白底渲染 | blend+bones]
                      trajectory_orbit.mp4
```

## 已验证结果（runs/sloth_20260801，真实数据）

- 10 seeds × (65k+262k)，每 seed 采样 ~9.2 s；选 seed_001。
- 粗对齐：240 候选视角，winner 38 内点，RMS 9.8 mm，重投影中位 4.7 px。
- 精配准：IoU 0.758→0.766，深度中位 9.47→8.91 mm（增量 0.25°/1 mm/+1%）。
- 动画：325 帧×54 bones → 1945 帧 30 fps；四元数范数全程 1.0000；
  最大高斯位移 78 mm（与真实操控幅度一致）；固定机位 blend 中生成体
  全程咬合真实物体。

## 测试

```bash
python -m pytest gaussian_align_demo/tests/ -q   # 14 项：Sim3 协方差一致性、
# PLY 往返、投影往返、Umeyama 精确恢复、37% 外点 RANSAC、刚体运动精确传输等
```
