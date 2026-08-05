# demo_v7 — demo_v6_2 的 GUI 软件化

demo_v7 把 demo_v6_2 的命令行 demo 变成一个图形界面软件。**数据处理与 6.2
逐级一致**:分割/追踪/shape-prior/chunk/下游全部 import 复用 `demo_v6_2`
的代码,一行不改;PhysTwin_shen 只调用(`demo_v6_2.phystwin_shen_launch`
原样),不属于本侧工作。demo_v7 新增的只有三层:GUI、控制协议、以及一个
"按钮驱动"的相机服务状态机。

## 技术选型(定案)

- **PySide6(Qt6 Widgets)而非 Rust**。整条流水线本来就是 Python 子进程,
  Rust GUI 仍要跨 IPC 控制同样的进程——语言边界恰好落在改动最频繁的接缝
  (流水线控制)上,而收益为零:GUI 的真实工作是 30 Hz 贴图和按钮,Qt 的
  渲染无论绑定语言都是 C++;重计算全部在 CUDA 子进程。效率靠纪律保证:
  帧走 UDS 传 JPEG(cv2 C++ 编解码)、GUI 侧 latest-wins 只显示最新帧
  (与 6.2 live viewer 同一纪律)、禁止逐像素 Python 循环。
- **进程拓扑与 6.2 同构**:GUI+orchestrator 一个父进程(chunk 流会话、
  points.npz 触发 PhysTwin 均按 6.2 复用);相机服务是子进程
  (`CUDA_VISIBLE_DEVICES` 按 6.2 的 gpu.main_data_processing 配置),
  import 复用 `demo_v6_2.mdp` 的模型与阶段函数。
- **仅支持 Linux/Ubuntu(X11)**;fake-live 与 real camera 同一套界面与
  按钮,唯一差别是素材来源与"播放完毕"弹窗。

## 与 6.2 的流程差异(仅此三处,均为明确需求)

1. **frame-0 由按钮拍摄**,不再是就绪屏障自动指定;拍摄后可"重拍/确认"。
2. **warmup 阶段不做任何追踪**(EdgeTAM/tracker 均不启动):确认 frame-0
   后人和物体即可离开;后台只跑 frame-0 派生管线(SAM3.1 三 mask →
   PCD → shape-prior 全链:upscale/generate/align/sample,全部 6.2 原码)。
3. **正式开始前有摆位步骤**:实时画面上以 50% 透明度叠加 frame-0 的
   object/hand mask,操作者把物体和双手摆回原位;点击开始后,以摆位后的
   帧为"正式 frame 0",**用保存的 frame-0 SAM3.1 mask 种一个全新的
   EdgeTAM session**(6.2 同一 seeding 代码路径,prompt 换成保存的
   mask),随后的 lossless 管线、chunk 流、products、下游触发与 6.2
   完全一致。
4. **完全不使用 canonical-mesh 缓存**:shape prior 每次都从本次 frame-0
   现场全量生成(upscale → segment → generate → align → sample),不读也
   不写 `~/qqtt_shape_prior_cache`。实现即 v6.2 自身的禁用语义
   (`--shape-prior-object` 缺省 = 缓存关闭),由相机服务无条件强制,
   父进程转发什么都不影响。

## 界面流程

```
源选择(real / fake-live+case)                     ┌────────────────────┐
        ↓                                          │ 常驻右上角:RGB +   │
拍摄屏:实时大图 + [拍摄第一张] → [确认/重拍]        │ 深度实时缩略图      │
        ↓ 确认(人离开)                             │(源打开即持续显示)  │
warmup 屏:阶段进度条 + 日志;完成后 [查看结果]      └────────────────────┘
        ↓
结果屏(标签页):Masks / Shape-Prior(mesh 拖拽查看)/ 补点(三源分色)
        ↓ [进入摆位]
摆位屏:实时画面 + mask 半透明叠加 → [开始正式追踪]
        ↓
正式屏:中央 = 6.2 live-viewer 复合视图(彩虹点云交互),可切 RGB/深度;
       [停止录制](real)/[停止](fake);fake 播放完毕 → 弹窗 → 回到开始
```

## 运行

**双击打开(推荐)**:安装一次启动器,之后从应用菜单或桌面图标 "Demo v7"
直接点开(无终端;日志在 `~/.local/state/demo_v7/logs/`,启动失败会弹窗):

```bash
bash demo_v7/launcher/install.sh      # 装应用菜单项 + 桌面图标(幂等)
bash demo_v7/launcher/demo_v7.sh --check   # 环境自检(不开 GUI)
```

启动器做完整 `conda activate demo_2_max`(复刻一直以来的终端运行环境)并
钉死 env 解释器;repo 移动位置后重跑一次 install.sh 即可(.desktop 内是
绝对路径)。

终端方式:

```bash
conda activate demo_2_max
python demo_v7/app.py                 # 默认读 demo_v7/config/default.yaml
python demo_v7/app.py --source fake-live --fake-live-case data_collect/sloth_new_20260705_230611
```

无头自检(不开真 GUI):

```bash
python -m pytest demo_v7/tests -q
python demo_v7/tests/drive_fake_live.py   # 脚本化控制通道走完整状态机
```

## 架构

```
demo_v7/
├── app.py                    # QApplication 入口 + 源选择 + 主窗口
├── ipc/
│   ├── protocol.py           # 协议唯一事实源(状态/命令/事件/帧通道)
│   └── channel.py            # UDS JSON-lines 控制通道 + 二进制帧通道
├── service/
│   ├── camera_service.py     # 相机服务子进程入口(CUDA ns 同 6.2)
│   ├── staged_runtime.py     # 按钮驱动状态机(复用 6.2 stage/模型)
│   └── frame0_pipeline.py    # frame-0 派生管线(SAM3.1 → PCD → shape prior)
├── gui/
│   ├── main_window.py        # 屏幕栈 + 常驻相机 dock + 事件路由
│   ├── screens.py            # 六个屏幕
│   └── widgets.py            # numpy→QImage 视图、视频循环、进度时间线
├── orchestration/session.py  # 父进程:spawn 服务、chunk 会话、PhysTwin 触发
├── config/default.yaml       # 仅 v7 自有键;流水线配置全部读 6.2 的
└── tests/                    # 协议/状态机/无头 GUI 冒烟 + 脚本化端到端
```

- 控制与帧协议详见 `ipc/protocol.py`(两侧共同 import,别处不得定义)。
- 正式阶段的中央视图直接复用 `demo_v6_2.mdp.live_viewer.render_pair_frame`
  (纯函数),保证和 6.2 的实时可视化逐像素同源。
- 输出目录布局、artifact 名称、pipeline_status.jsonl 全部沿用 6.2。

## fake-live 语义(重要)

fake-live **只替换来源,不替换相机的持续输入特性**:流从源打开到关闭
永不暂停——拍摄第一张 = 抓当前帧快照(流继续走,确认屏钉住的是快照
png);摆位 overlay 叠在**实时流**上,用于实时判断双手/物体是否摆回
frame-0 位置;正式 frame-0 = seg 就绪那一刻的当前帧(与 v6.2 就绪屏障
语义一致)。正式开始前素材播完会自动从头继续(状态栏提示);正式期播完
弹窗结束并回到起点。
