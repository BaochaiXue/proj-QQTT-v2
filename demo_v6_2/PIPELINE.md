# Demo v6.2 流水线：23 个设计问题的代码答案

本文按设计评审中的 23 个问题，说明 Demo v6.2 每个阶段由哪个模块负责、
数据如何流动，以及出错时在哪里终止。每个“大步骤”都对应一个名称明确的
模块；相关模块的 docstring 也会重复说明自己负责的问题。

整体运行结构如下：`main.py` 是总控进程，它启动
`main_data_processing.py` 子进程。后者由多个 `mdp_demo_*` mixin 组成，负责
摄像头、分割、跟踪和 warm-up。总控进程持续读取摄像头进程写出的
`frames.jsonl`，通过 `chunk_*` 模块生成在线 chunk，同时通过
`shape_prior_*` 模块生成 SAM3D shape prior，最后启动一个下游消费者：
`visualize_track.py` 或 `phystwin_shen`。各进程还会把生命周期事件追加到
`pipeline_status.jsonl`，由 `pipeline_status.py` 统一定义事件格式。

## 摄像头与逐帧 I/O（Q1–Q6）

1. **摄像头在哪里启动？**
   `main.py:main()` 通过 `subprocess.Popen(main_data_processing_command)` 启动
   摄像头子进程。子进程内的 `MainDataProcessingDemo.run`
   （`mdp_demo_lifecycle.py`）根据输入模式选择
   `mdp_capture_source._start_realsense_pipeline`（真实 RealSense）或
   `RecordedRgbdFrameSource`（fake-live 录制回放）。

2. **摄像头线程在哪里创建？**
   `mdp_demo_lifecycle` 在启动运行线程时创建 daemon `threading.Thread`。
   其中包括 `mdp_demo_capture._capture_worker`、
   `_capture_recording_worker`，以及 masked EdgeTAM 等处理线程。

3. **进程和线程如何协作？**
   外层是多进程：总控、摄像头、shape-prior 阶段和下游消费者相互独立。
   摄像头进程内部使用 Python `threading`，并通过
   `OrderedPacketQueue`（按 seq 保序）、`LatestSlot`（只保留最新值）和
   `SameSeqPairer`（按相同 seq 配对）传递数据。前者和后者位于
   `mdp_pipeline_plumbing.py`，`LatestSlot` 位于 `utils/concurrency.py`。
   所有线程通过 `stop_event` 协调退出。

4. **RealSense RGB 和 depth 的 FPS 是多少？**
   所有启用的 RealSense stream 使用同一个 `int(args.fps)`，由
   `--camera-fps` 控制，默认是 **30 FPS**。发布和 chunk 时间线与采集频率
   分离，由 `--replay-fps` 控制，默认是 **5 FPS**；严格 5 FPS 运行时每隔
   `1 / replay_fps` 取一次最新输入帧。默认值来自 `config/default.yaml`。

5. **每帧在哪里读取？**
   `mdp_demo_capture` 的 capture worker 从 RealSense 取帧，或从录制数据中
   读取 RGB-D 引用，然后封装成 `mdp_packets.py` 中的 `FramePacket`，再写入
   `lossless_frame_queue` 或 `capture_slot`。

6. **frame id 和 timestamp 从哪里来？**
   `FramePacket.seq` 是流水线内部帧号。`source_timestamp_s` 和
   `source_frame_index` 在 fake-live 模式下来自原始录制元数据，在真实采集
   模式下来自摄像头，并随 prepared frame 一起保存。正式发布的均匀时间线
   使用 `frame_index × 1 / fps`；`source_*` 只负责记录来源。

## Warm-up（Q7–Q14）

7. **Warm-up 使用单帧还是一段视频？**
   使用单帧。`main_warmup.prepare_segmentation_warmup` 通过
   `mdp_demo_segwarmup._wait_for_first_frame` 取得第一帧，在这张 frame 0 上
   运行 SAM3.1，得到冻结身份所需的初始 mask，并用同一帧初始化 EdgeTAM。

8. **系统如何确认拿到的是 frame 0？**
   `_wait_for_first_frame` 从 sentinel seq `-1` 开始调用
   `capture_slot.get_latest_after(-1)`；第一张满足 `seq > -1` 的帧就是初始帧。
   lossless 模式则直接取 `lossless_frame_queue` 的队首。该帧只用于初始化
   EdgeTAM 一次。

9. **Warm-up 期间后续到达的帧如何处理？**
   这些帧仍会通过 EdgeTAM，且使用 `add_prompt=False`，因此跟踪器状态和
   preview 会继续前进；但正式 chunk gate 尚未打开，不会把这些帧写入正式
   chunk。`frames.jsonl` 在此期间只保留 warm-up frame 0 对应的锚点行。
   gate 规则由 `mdp_packets._formal_chunk_rows_gated` 和 `design_spec.md` 定义。

10. **Warm-up 最耗时的步骤是什么？**
    通常是 SAM3D shape-prior 链路，入口为
    `shape_prior_warmup.ShapePriorLocalClient.request_shape_prior`。顺序是：
    upscale → SAM3.1 segment → **SAM3D generate** → SuperGlue align → sample，
    其中 SAM3D generate 通常占用时间最多。每个阶段都会记录
    `shape_prior_*_ms`，代码没有写死固定耗时。

11. **Warm-up 完成后保留哪些状态和文件？**
    内存中保留 frame-0 `InitialMaskBundle`、`SegmentationWarmupState`，以及按
    hand_a / object / hand_b 初始化完成的 EdgeTAM session。磁盘上保留
    shape-prior case、`points.npz` 和 `final_mesh.glb`。

12. **Warm-up 状态如何校验？**
    校验发生在数据边界并直接失败：mask 尺寸必须等于输入帧尺寸；
    `_union_masks` 不接受空 mask 或形状不一致；
    `split_controller_hand_instances` 必须找到两只可分离的手；shape prior 不接受
    空 object mask、没有有效 depth 点或点数不足。相关实现位于
    `main_warmup.py` 和 `shape_prior_warmup.py`。

13. **Warm-up 出错后如何处理？**
    shape-prior 阶段以独立子进程运行并使用 `check=True`，任一阶段失败都会
    直接抛错。分割或 warm-up 线程异常会进入
    `mdp_demo_lifecycle._record_fatal_worker_error`：设置 `stop_event`、写入
    `fatal_error` 状态事件，并让摄像头进程以非零状态退出。shape-prior 终态
    失败时会解除 chunk 等待，使总控进程能够明确报告失败，而不是无限等待。

14. **正式时间线从哪里开始？**
    warm-up frame 0 占据发布 output frame 0 的锚点位置。shape prior 进入
    READY、chunk gate 打开后，第一张新处理的帧成为 output frame 1，并紧接在
    frame 0 后面。

## Chunk 组装、跟踪与过滤（Q15–Q19）

15. **Chunk 如何组装？**
    `chunk_data_stream.py` 同步累积 capture row 和 canonical prepared frame。
    数量达到 `chunk_size` 后关闭当前窗口。`chunk_window_builder.py` 直接读取
    prepared frame 已包含的 RGB、mask、世界坐标 PCD、track、visibility 和
    query points，并把逐帧数组沿新的时间维堆叠；这里不会再从 legacy sidecar
    重建数据。

16. **Chunk 大小在哪里配置？**
    `main_options.resolve_chunk_frame_count` 默认计算
    `round(replay_fps × chunk_seconds)`。默认值为 5 FPS × 7 秒 = **35 帧**，
    也可以通过 `--chunk-frame-count` 明确指定。最终大小写入
    `ChunkDataWriter.chunk_size`，并在 `chunk_data_output.py` 校验为正整数。

17. **Chunk 按时间还是按帧数关闭？**
    严格按**帧数**。当 buffer 长度等于 `chunk_size` 时关闭窗口，不根据
    wall-clock 时间猜测窗口边界。

18. **Chunk 组装后如何执行跟踪？**
    整个 session 只创建一个 `tracking.TrackingRuntime`。chunk 0 冻结 query
    identity、controller anchor 和 neighbor table；后续 chunk 复用这些状态。
    每个窗口依次调用 `tracking.build_window_observations` 和
    `TrackingRuntime.process_window`，执行冻结标签、逐帧 temporary-invalid 判定
    和 local-rigid anchor recovery。

19. **跟踪后还会做哪些过滤？**
    `tracking.motion_consistency` 执行动作一致性过滤：半径 0.01 m、至少 5 个
    邻居（包含自身）、相似阈值 0.005 m、至少 50% 邻居同意。depth-validity
    和 3D radius-outlier mask refinement 只在摄像头进程写 canonical prepared
    frame 时执行一次（`phystwin_strict_product.py`）。随后 `asap.py` 可填充无效
    object 点；`design_spec_v6_1.md` 明确要求保留其 silent-freeze 行为。

## 训练侧 schema、manifest 与读取起点（Q20–Q22）

20. **训练侧会收到什么数据？**
    每个窗口写成
    `online_data/chunks/chunk_{id:06d}.pkl`，记录 case、chunk id、
    `start_frame` / `end_frame`、source frame/timestamp，以及从 `data_keys.py`
    定义的 TIME_KEYS 切出的逐帧数组。必需数据包括 object points、colors、
    visibilities、motions-valid 和 controller points；可选数据包括 ASAP
    surface/interior points 与 recovery mask。

    同一 session 还会生成 RGB-D archive：
    `online_data/color/0/{k}.png`、`online_data/depth/0/{k}.npy`（uint16 mm）、
    `calibrate.pkl`、`metadata.json`、`enhance_metadata.json`，以及聚合后的
    `data/final_data.pkl`。实现分别位于 `chunk_data_output.py` 和
    `online_frame_archive.py`。

21. **Manifest 何时更新，如何保证读者不会看到半成品？**
    提交顺序是：先写完本 chunk 的 RGB-D archive；再原子写入 chunk pickle，
    更新 `online_data/manifest.json`；chunk 提交成功后才推进 archive 的
    `metadata.json` 和 `enhance_metadata.json`。因此，读者看到 manifest 中的
    新 committed chunk 时，对应帧文件已经存在；metadata 的 `frame_num` 也只
    统计已经提交成功的帧。manifest 状态从 `recording` 进入 `finished` 或
    `failed`。相关实现位于 `chunk_materialize.py`、`chunk_data_output.py`、
    `online_frame_archive.py` 和 `utils/atomic_io.py`。

22. **训练侧从什么时候开始读取？**
    `main.py._maybe_start_phystwin_shen` 在 `shape_prior/points.npz` 出现后启动
    `phystwin_shen/train_online_warp.py`。训练进程轮询
    `online_data/manifest.json`，从第一个 committed chunk 开始顺序读取；当
    manifest 进入 `finished` 且启用 stop-when-finished 时停止。启动逻辑位于
    `phystwin_shen_launch.py`。

## 在线流水线状态可视化（Q23）

23. **如何看到流水线当前在做什么，以及 warm-up 是否失败？**
    `visualize_track.py` 和 `viz_*` 模块负责显示 live input RGB 与
    `final_data` 输出。Demo v6.2 还让各进程把生命周期事件追加到
    `<base_path>/pipeline_status.jsonl`：

    - 总控进程写入 run start、chunk committed、downstream start、finish/fatal；
    - 摄像头进程写入 capture start、shape-prior submitted、warm-up ready 和
      fatal error；
    - shape-prior 各阶段写入自己的运行状态。

    viewer 通过 `viz_playback.run_side_by_side` 持续读取该文件，并调用
    `viz_panels.draw_pipeline_status` 绘制状态条。发生 warm-up 或 shape-prior
    错误时，状态条变红并显示错误信息。状态日志只用于观察运行过程；写入失败
    不会改变正式 chunk 和 RGB-D 产品。
