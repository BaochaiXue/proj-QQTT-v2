const modes = {
  overview: {
    title: "总览",
    summary:
      "三条线一起看: GPU 0 做 camera/perception/tracking，CPU 编排进程和 chunk，GPU 1 做 shape-prior worker 与当前 visualizer。",
    tags: ["warmup", "runtime", "shape", "actual", "io", "optional"],
  },
  shape: {
    title: "Shape prior warmup",
    summary:
      "只看 SAM3D shape prior: managed worker 先在 GPU 1 preload，主 runtime 首个有效 PCD 后提交一次请求，chunk writer 等结果进入 final_data。",
    tags: ["shape"],
  },
  runtime: {
    title: "其他 runtime warmup",
    summary:
      "只看非 shape-prior warmup: 相机/fake-live、FFS/SAM3.1/EdgeTAM/TapNext++/PCD filter 被热起来，目标是稳定写 frames.jsonl 和 prepared frame。",
    tags: ["runtime"],
  },
  actual: {
    title: "Actual run",
    summary:
      "只看实际运行: GPU 0 持续产出 capture 与 chunks，GPU 1 启动 visualize_track.py 做 side-by-side visualizer；realtime_phystwin 默认不启动。",
    tags: ["actual"],
  },
};

const lanes = [
  {
    id: "gpu0",
    title: "GPU 0: realtime camera / perception / tracking",
    subtitle: "默认 CUDA_VISIBLE_DEVICES=0，进程是 demo_v5_1/realtime_dense_track.py",
    nodes: [
      {
        n: "G0-1",
        kind: "warmup",
        tags: ["runtime"],
        title: "建立相机或 fake-live source",
        body:
          "live 模式启动 RealSense；fake-live 模式读取 recording 并按 replay_fps 模拟实时节奏。",
        source: "realtime_dense_track.py: run(), RecordedRgbdFrameSource",
      },
      {
        n: "G0-2",
        kind: "warmup",
        tags: ["runtime"],
        title: "深度路径 warmup",
        body:
          "native-realsense 直接使用 RGB-D；如果选择 ir-ffs，则创建 TensorRT FFS runner，并 warm up Numba IR-to-color align。",
        source: "realtime_dense_track.py: warm_up_numba_ffs_align()",
      },
      {
        n: "G0-3",
        kind: "warmup",
        tags: ["runtime"],
        title: "SAM3.1 第一帧 mask 初始化",
        body:
          "第一帧用 SAM3.1 生成 controller/object 初始 mask，之后交给 EdgeTAM streaming propagation。",
        source: "run_sam31_first_frame_mask_bundle()",
      },
      {
        n: "G0-4",
        kind: "warmup",
        tags: ["runtime"],
        title: "EdgeTAM 编译和 streaming session",
        body:
          "加载 HF EdgeTAM，应用 vision-reduce-overhead compile mode，创建 frame-by-frame session。",
        source: "_init_hf_model(), _seg_worker()",
      },
      {
        n: "G0-5",
        kind: "warmup",
        tags: ["runtime"],
        title: "TapNext++ query 初始化",
        body:
          "在 object/controller union mask 上采样 query，初始化 tracker，并进入严格同序 pair 输出。",
        source: "_ensure_tracker_queries(), adapter.initialize()",
      },
      {
        n: "G0-6",
        kind: "warmup",
        tags: ["shape", "runtime"],
        title: "首个有效 PCD 触发 shape-prior 请求",
        body:
          "当 mask、depth、PCD 和 table c2w 都齐备时，主进程组装 frame-0-like ShapePriorFrame0Request。",
        source: "_shape_prior_frame0_request_from_pcd_result()",
      },
      {
        n: "G0-7",
        kind: "run",
        tags: ["actual"],
        title: "持续输出 prepared PhysTwin frame",
        body:
          "每帧写 prepared_phystwin/*.npz，并 append frames.jsonl；这是 chunk writer 的实时输入流。",
        source: "HeadlessCaptureWriter.write_pcd()",
      },
      {
        n: "G0-8",
        kind: "io",
        tags: ["actual", "io"],
        title: "写 input RGB timeline",
        body:
          "side-by-side viewer 打开时，capture 也写 input_frames.jsonl 和 input_rgb/*.png，供左侧画面跟随。",
        source: "write_input_frame(), --write-input-rgb-timeline",
      },
    ],
  },
  {
    id: "cpu",
    title: "CPU / IO: orchestration and chunk bridge",
    subtitle: "main.py 管进程，realtime_data_process_track.py tail frames.jsonl",
    nodes: [
      {
        n: "C-1",
        kind: "warmup",
        tags: ["runtime", "shape"],
        title: "解析 default.yaml 与 CLI",
        body:
          "默认 realtime_gpu_mode=single，warmup_gpu_mode=dual，point_viewer=window，optimization=disabled。",
        source: "config/default.yaml, build_parser()",
      },
      {
        n: "C-2",
        kind: "warmup",
        tags: ["shape"],
        title: "先启动 managed shape-prior worker",
        body:
          "如果 shape_prior_warmup=true 且 worker_mode=managed，main.py 先起 worker，再起相机进程。",
        source: "_start_managed_shape_prior_worker()",
      },
      {
        n: "C-3",
        kind: "run",
        tags: ["actual"],
        title: "启动 camera subprocess",
        body:
          "主进程把 camera CUDA namespace 设为 0，并把 shape-prior endpoint 等参数传给 realtime_dense_track.py。",
        source: "build_camera_realtime_command()",
      },
      {
        n: "C-4",
        kind: "run",
        tags: ["actual"],
        title: "立即启动 side-by-side visualizer",
        body:
          "默认 layout=side-by-side，所以 viewer 在 camera 启动后立刻开，右侧先显示 waiting for first final_data chunk。",
        source: "point_viewer_start_policy()",
      },
      {
        n: "C-5",
        kind: "run",
        tags: ["actual", "io"],
        title: "tail frames.jsonl 并按窗口闭合",
        body:
          "stream_chunks_from_headless_capture() 读取 append-only frames.jsonl，默认每 35 帧关闭一个 chunk。",
        source: "stream_chunks_from_headless_capture()",
      },
      {
        n: "C-6",
        kind: "warmup",
        tags: ["shape", "actual"],
        title: "写 chunk 前等待 shape prior",
        body:
          "如果 require_shape_prior=true，chunk writer 在 final_data materialization 前等待 surface/interior points。",
        source: "_shape_points_for_chunk()",
      },
      {
        n: "C-7",
        kind: "io",
        tags: ["actual", "io"],
        title: "发布 chunk case 与 online stream",
        body:
          "每个窗口写 data_process chunk case，并更新 online_data/<case> 与 data/<case>/final_data.pkl。",
        source: "_write_chunk_from_rows(), ChunkedFinalDataWriter",
      },
      {
        n: "C-8",
        kind: "optional",
        tags: ["optional"],
        title: "可选 realtime_phystwin 分支",
        body:
          "只有 optimization_mode=continuous 时，第一 committed chunk 后才释放 worker 并启动 train_online_zero_then_first.py。",
        source: "_start_continuous_optimization()",
      },
    ],
  },
  {
    id: "gpu1",
    title: "GPU 1: shape prior and current visualizer",
    subtitle: "默认 CUDA_VISIBLE_DEVICES=1；worker 内部 cuda:0 映射到 physical GPU 1",
    nodes: [
      {
        n: "G1-1",
        kind: "warmup",
        tags: ["shape"],
        title: "worker preload x4 upscaler",
        body:
          "StableDiffusionUpscalePipeline 先加载到 worker device；shape-prior crop 会被放大给 SAM3D。",
        source: "ShapePriorSam3DWorker.preload_models()",
      },
      {
        n: "G1-2",
        kind: "warmup",
        tags: ["shape"],
        title: "worker preload SAM3D inference",
        body:
          "加载 vendor/demo_runtime/sam-3d-objects 的 pipeline config，并准备 SAM3D Objects 推理。",
        source: "_load_inference(), Inference(config, compile=False)",
      },
      {
        n: "G1-3",
        kind: "warmup",
        tags: ["shape"],
        title: "绑定 ZeroMQ REP endpoint",
        body:
          "worker 输出 ready 后，主 runtime 的 ShapePriorRemoteClient 可以发送单帧 npz request。",
        source: "shape_prior_worker.py: socket.bind()",
      },
      {
        n: "G1-4",
        kind: "warmup",
        tags: ["shape"],
        title: "执行一次 SAM3D shape prior",
        body:
          "worker 裁剪 object，x4 upscale，运行 SAM3D mesh，按 observation depth 做 single-view 对齐。",
        source: "handle(), _canonical_points_from_sam3d()",
      },
      {
        n: "G1-5",
        kind: "io",
        tags: ["shape", "io"],
        title: "回传 surface / interior points",
        body:
          "结果打包为一个 npz response，主相机进程写 shape_prior/points.npz 和 metadata。",
        source: "pack_shape_prior_result(), write_shape_prior_result()",
      },
      {
        n: "G1-6",
        kind: "run",
        tags: ["actual"],
        title: "启动 visualize_track.py",
        body:
          "当前 actual run 的另一边是 point viewer。它读取 input timeline 和 final_data/online_data 做 side-by-side 显示。",
        source: "build_point_viewer_command(), visualize_track.py",
      },
      {
        n: "G1-7",
        kind: "run",
        tags: ["actual"],
        title: "左 RGB / 右 final_data 同步",
        body:
          "左侧跟 camera input；右侧选择与目标 latency 最匹配的 final_data frame，未出 chunk 时显示 waiting。",
        source: "InputReceiveTimeline, OutputStreamPlaybackCursor",
      },
      {
        n: "G1-8",
        kind: "optional",
        tags: ["optional"],
        title: "非默认: realtime_phystwin optimization",
        body:
          "continuous optimization 也默认用 GPU 1，但当前 default.yaml 是 disabled，因此不属于当前 actual run。",
        source: "optimization_mode: disabled",
      },
    ],
  },
];

function createNode(node) {
  const article = document.createElement("article");
  article.className = "flow-node";
  article.dataset.kind = node.kind;
  article.dataset.tags = node.tags.join(" ");

  const title = document.createElement("h3");
  const text = document.createElement("span");
  text.textContent = node.title;
  const index = document.createElement("span");
  index.className = "node-index";
  index.textContent = node.n;
  title.append(text, index);

  const body = document.createElement("p");
  body.textContent = node.body;

  const source = document.createElement("small");
  source.textContent = node.source;

  article.append(title, body, source);
  return article;
}

function renderLanes() {
  const grid = document.querySelector("#flow-grid");
  if (!grid) return;
  grid.replaceChildren();

  lanes.forEach((lane) => {
    const section = document.createElement("section");
    section.className = `lane lane-${lane.id}`;

    const header = document.createElement("div");
    header.className = "lane-header";
    const title = document.createElement("strong");
    title.textContent = lane.title;
    const subtitle = document.createElement("span");
    subtitle.textContent = lane.subtitle;
    header.append(title, subtitle);

    const body = document.createElement("div");
    body.className = "lane-body";
    lane.nodes.forEach((node) => body.append(createNode(node)));

    section.append(header, body);
    grid.append(section);
  });
}

function setMode(modeName) {
  const mode = modes[modeName] || modes.overview;
  document.querySelectorAll(".mode-button").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.mode === modeName);
  });

  const title = document.querySelector("#mode-title");
  const summary = document.querySelector("#mode-summary");
  if (title) title.textContent = mode.title;
  if (summary) summary.textContent = mode.summary;

  document.querySelectorAll(".flow-node").forEach((node) => {
    const tags = (node.dataset.tags || "").split(" ");
    const highlighted = mode.tags.some((tag) => tags.includes(tag));
    node.classList.toggle("is-highlighted", highlighted);
    node.classList.toggle("is-dimmed", modeName !== "overview" && !highlighted);
  });
}

function bindControls() {
  document.querySelectorAll(".mode-button").forEach((button) => {
    button.addEventListener("click", () => {
      setMode(button.dataset.mode || "overview");
    });
  });
}

renderLanes();
bindControls();
setMode("overview");
