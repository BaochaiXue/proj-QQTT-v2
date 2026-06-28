const lanes = [
  {
    id: "CPU/IO",
    title: "CPU / IO / process control",
    subtitle: "相机、case 文件、IPC、profile、visualizer drawing",
  },
  {
    id: "GPU0",
    title: "GPU0 / mask-gpu / main CUDA",
    subtitle: "FFS depth, SAM3.1/EdgeTAM, fusion, Open3D render",
  },
  {
    id: "GPU1",
    title: "GPU1 / cotracker-gpu",
    subtitle: "TAPNext++ 或 LiteTracker tracker 子进程",
  },
];

const nodes = [
  {
    id: "process-split",
    phase: "warmup",
    gpu: "CPU/IO",
    title: "启动进程并切 CUDA 视野",
    detail:
      "entrypoint 先用 --mask-gpu 设置主进程 CUDA_VISIBLE_DEVICES；tracker 配置单独记录 --cotracker-gpu。",
    why:
      "这样 GPU0 和 GPU1 的模型加载不会互相看见对方的显存，profile 也能按设备分开。",
    output: "main_cuda_visible_devices=0, cotracker_cuda_visible_devices=1",
  },
  {
    id: "ffs-warmup",
    phase: "warmup",
    gpu: "GPU0",
    title: "FFS TensorRT depth 预热",
    detail:
      "Demo 3.2/3.3 live path 使用 FFS TensorRT builderOptimizationLevel=5, static batch=3。",
    why:
      "让深度网络、TensorRT engine 和 batch=3 路径先完成冷启动。",
    output: "三相机 FFS depth maps",
  },
  {
    id: "sam-edgetam-warmup",
    phase: "warmup",
    gpu: "GPU0",
    title: "SAM3.1 first-frame init + EdgeTAM session",
    detail:
      "GPU0 产生 object=stuffed animal 与 controller=towel 的初始 masks，空 mask 默认 fail fast。",
    why:
      "后续 tracker query 和 trackable masks 都依赖第一帧 object/controller 语义。",
    output: "object_mask_by_camera, controller_mask_by_camera",
  },
  {
    id: "tracker-persistent-worker",
    phase: "warmup",
    gpu: "GPU1",
    title: "Tracker 子进程 ready",
    detail:
      "TAPNext++ 或 LiteTracker 在 GPU1 预加载模型；LiteTracker 支持 lazy query init，先 ready_to_receive_inputs。",
    why:
      "主进程可以持续发 latest-wins input，tracker 只处理最新有效组，避免队列堆积。",
    output: "tracker_process_ready=true",
  },
  {
    id: "open3d-first-render-gate",
    phase: "warmup",
    gpu: "GPU0",
    title: "Open3D warmup HUD + first render gate",
    detail:
      "Open3D HUD 从 active runtime pipeline 生成，render 默认等 tracker result 和 3D anchors。",
    why:
      "第一帧不是空 PCD，也不是没有 tracker marker 的半成品。",
    output: "tracking_overlay_first_render_group_id",
  },
  {
    id: "shape-snapshot",
    phase: "shape",
    gpu: "CPU/IO",
    title: "第一组 strict-source input 被 snapshot",
    detail:
      "复制 RGB-D、object/controller masks、intrinsics、c2w 和 camera_ids，source_group_id 写入 metadata。",
    why:
      "shape prior 后面可以离线跑，但输入仍精确来自 live 的第一组有效三相机 bundle。",
    output: "demo33_shape_prior_warmup/<run_id>/case",
  },
  {
    id: "shape-case-write",
    phase: "shape",
    gpu: "CPU/IO",
    title: "写 FuturePhysTwin-style case",
    detail:
      "写 color/mask/pcd/calibrate.pkl/processed_masks.pkl/track_process_data.pkl；controller 点会 cap。",
    why:
      "FuturePhysTwin route 需要它熟悉的目录结构，而不是直接吃 live memory。",
    output: "shape_prior_status=case_ready",
  },
  {
    id: "shape-route",
    phase: "shape",
    gpu: "GPU0",
    title: "after-teardown 跑五段 shape-prior route",
    detail:
      "默认 shape_prior_gpu=auto 解析到 mask GPU。detached worker 等 live pid 退出后再跑 image upscale、segmentation、SAM3D、align、sample。",
    why:
      "重模型不和 FFS/EdgeTAM/tracker live window 抢 24GB 显存。",
    output: "final_data.pkl",
  },
  {
    id: "shape-gpu1-released",
    phase: "shape",
    gpu: "GPU1",
    title: "GPU1 已释放或空闲",
    detail:
      "默认 after-teardown 策略下，tracker 子进程已经 stop，GPU1 不参与 shape-prior heavy route。",
    why:
      "这就是 shape prior 和普通 warmup 分开的关键：它不是 live tracker 的前置依赖。",
    output: "shape_prior_blocks_tracker_input=false",
  },
  {
    id: "shape-render-layer",
    phase: "shape",
    gpu: "GPU0",
    title: "灰色 canonical reference layer",
    detail:
      "读取 final_data.pkl 后拼 object_points[0] + surface_points + interior_points，只 attach 到 render packet。",
    why:
      "它只帮助观察形状先验，不改变 live fused PCD、mask、queries 或 tracker markers。",
    output: "shape_prior_render_layer_enabled=true",
  },
  {
    id: "capture-group",
    phase: "actual",
    gpu: "CPU/IO",
    title: "三相机 capture group",
    detail:
      "RealSense 产生 group_id 绑定的 RGB/IR/depth 输入，后续 stage 都围绕这个 source group 对齐。",
    why:
      "actual run 的稳定性来自严格同源，不靠 nearest-frame 或 stale reuse。",
    output: "group_id=N",
  },
  {
    id: "actual-depth-mask",
    phase: "actual",
    gpu: "GPU0",
    title: "FFS depth + EdgeTAM masks",
    detail:
      "FFS depth、semantic masks 和 trackable masks 保持在主进程路径，depth/intrinsics/c2w 用于 3D lift。",
    why:
      "GPU0 负责把三相机观察变成可融合、可跟踪、可渲染的 bundle。",
    output: "depth_by_camera + trackable masks",
  },
  {
    id: "tracker-input-ipc",
    phase: "actual",
    gpu: "CPU/IO",
    title: "latest-wins IPC 发布 tracker input",
    detail:
      "主进程发送 RGB + union/object/controller trackable masks；depth 和 c2w 不发给 tracker child。",
    why:
      "tracker 只需要 2D RGB/mask，3D lift 留在主进程做，减少 IPC payload 和 GPU 纠缠。",
    output: "TrackingInputLitePacket",
  },
  {
    id: "tracker-model",
    phase: "actual",
    gpu: "GPU1",
    title: "TAPNext++ / LiteTracker 推理",
    detail:
      "GPU1 对三视角 input 做 batch-views 或 serial update，返回 tracks、visibility 和 object/controller 分类。",
    why:
      "tracking 与 FFS/EdgeTAM 解耦，GPU0 可以继续处理下一组输入。",
    output: "TrackingResultLitePacket",
  },
  {
    id: "strict-bundle",
    phase: "actual",
    gpu: "CPU/IO",
    title: "strict-source bundle 检查",
    detail:
      "render 需要 RGB、FFS depth、mask source、tracker result、lift input 与 render packet 同 group。",
    why:
      "默认路径下 tracker result 没有同源 bundle 就不能作为新 rendered tracking frame。",
    output: "same-bundle invariant",
  },
  {
    id: "actual-render",
    phase: "actual",
    gpu: "GPU0",
    title: "3D lift + Open3D render",
    detail:
      "主进程把 visible tracks lift 到世界坐标，渲染 object/controller PCD、红/青 tracker markers，以及可选灰色 shape prior。",
    why:
      "最终画面同时显示 live observation 和 render-only shape reference。",
    output: "Open3D pointcloud frame",
  },
  {
    id: "visualizer-input",
    phase: "visualizer",
    gpu: "CPU/IO",
    title: "准备 visualizer 输入",
    detail:
      "使用 actual run 或 diagnostic run 产出的 video tensor、tracks、visibility 和可选 segm_mask。",
    why:
      "这是一条解释/复查路径，输入已经存在，不再启动完整 PhysTwin 推理训练栈。",
    output: "video, tracks, visibility",
  },
  {
    id: "visualizer-draw",
    phase: "visualizer",
    gpu: "CPU/IO",
    title: "Visualizer 画点线并保存 mp4",
    detail:
      "Visualizer.draw_tracks_on_video 将 tensor 转到 CPU numpy/PIL，按 rainbow/cool/optical_flow 配色画轨迹。",
    why:
      "actual run 的另一遍用它做 track 可视化，而不是跑 realtime_phystwin 主程序。",
    output: "diagnostic mp4",
  },
  {
    id: "visualizer-gpu0-note",
    phase: "visualizer",
    gpu: "GPU0",
    title: "GPU0 不重新跑 live depth/mask",
    detail:
      "visualizer pass 消费已有结果；除非上游另行生成 tensor，它本身不是 FFS/EdgeTAM live owner。",
    why:
      "避免把诊断展示误解成第二套 realtime pipeline。",
    output: "no new FFS/EdgeTAM ownership",
  },
  {
    id: "visualizer-gpu1-note",
    phase: "visualizer",
    gpu: "GPU1",
    title: "GPU1 不重新跑 tracker child",
    detail:
      "visualizer pass 读取 tracks/visibility，不重新初始化 TAPNext++ 或 LiteTracker 子进程。",
    why:
      "它验证和讲解已有轨迹，不改变 actual run 的 tracker 测量。",
    output: "no tracker subprocess",
  },
];

const phaseLabels = {
  warmup: "普通 warmup",
  shape: "shape prior",
  actual: "actual run",
  visualizer: "visualizer",
};

const laneContainer = document.querySelector("#flow-lanes");
const detailTitle = document.querySelector("#detail-title");
const detailBody = document.querySelector("#detail-body");
const detailMeta = document.querySelector("#detail-meta");
const phaseButtons = Array.from(document.querySelectorAll(".phase-button"));

let selectedId = null;
let activePhase = "all";

function visibleNodes() {
  if (activePhase === "all") {
    return nodes;
  }
  return nodes.filter((node) => node.phase === activePhase);
}

function renderLanes() {
  const shown = visibleNodes();
  laneContainer.innerHTML = "";

  lanes.forEach((lane) => {
    const section = document.createElement("section");
    section.className = "gpu-lane";
    section.dataset.lane = lane.id;

    const header = document.createElement("header");
    header.className = "lane-header";
    header.innerHTML = `
      <strong><span class="lane-dot" aria-hidden="true"></span>${lane.title}</strong>
      <span>${lane.subtitle}</span>
    `;

    const body = document.createElement("div");
    body.className = "lane-body";

    shown
      .filter((node) => node.gpu === lane.id)
      .forEach((node) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "step-node";
        button.dataset.phase = node.phase;
        button.dataset.gpu = node.gpu;
        button.dataset.id = node.id;
        if (node.id === selectedId) {
          button.classList.add("is-selected");
        }
        button.innerHTML = `
          <span class="phase-chip">${phaseLabels[node.phase]}</span>
          <h3>${node.title}</h3>
          <p>${node.detail}</p>
        `;
        button.addEventListener("click", () => selectNode(node.id));
        body.appendChild(button);
      });

    if (!body.children.length) {
      const empty = document.createElement("p");
      empty.className = "empty-lane";
      empty.textContent = "这个阶段没有主要节点占用这条泳道。";
      body.appendChild(empty);
    }

    section.append(header, body);
    laneContainer.appendChild(section);
  });
}

function selectNode(nodeId) {
  const node = nodes.find((item) => item.id === nodeId);
  if (!node) {
    return;
  }
  selectedId = node.id;
  detailTitle.textContent = node.title;
  detailBody.textContent = node.detail;
  detailMeta.innerHTML = "";

  [
    ["阶段", phaseLabels[node.phase]],
    ["主要泳道", node.gpu],
    ["为什么这样做", node.why],
    ["输出/状态", node.output],
  ].forEach(([label, value]) => {
    const row = document.createElement("div");
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = label;
    dd.textContent = value;
    row.append(dt, dd);
    detailMeta.appendChild(row);
  });

  document
    .querySelectorAll(".step-node")
    .forEach((item) => item.classList.toggle("is-selected", item.dataset.id === node.id));
}

function setPhase(phase) {
  activePhase = phase;
  const shown = visibleNodes();
  if (!shown.some((node) => node.id === selectedId)) {
    selectedId = shown.length ? shown[0].id : null;
  }
  phaseButtons.forEach((button) => {
    const active = button.dataset.phase === phase;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });
  renderLanes();
  if (selectedId) {
    selectNode(selectedId);
  }
}

phaseButtons.forEach((button) => {
  button.addEventListener("click", () => setPhase(button.dataset.phase));
});

setPhase("all");
