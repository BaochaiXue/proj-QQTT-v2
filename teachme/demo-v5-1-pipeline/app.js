const modes = {
  overview: {
    title: "思维导图总览",
    summary:
      "先看中心思维导图：shape-prior warmup、普通 runtime warmup、actual run、GPU routing、artifacts 和代码 ownership 分开理解。",
  },
  shape: {
    title: "Shape prior warmup",
    summary:
      "只看 shape prior：managed worker 先在 GPU 1 preload x4 upscaler 和 SAM3D，等 GPU 0 的首个有效 PCD 后提交一次请求。",
  },
  runtime: {
    title: "普通 runtime warmup",
    summary:
      "只看普通 warmup：GPU 0 准备 source、depth、PCD filter、table calibration、SAM3.1、EdgeTAM 和 tracker。",
  },
  actual: {
    title: "Actual run",
    summary:
      "只看实际运行：GPU 0 持续写 capture/chunk，GPU 1 启动 visualize_track.py；realtime_phystwin 默认不启动。",
  },
};

function setMode(modeName) {
  const nextMode = modes[modeName] ? modeName : "overview";
  const mode = modes[nextMode];
  document.body.dataset.mode = nextMode;

  document.querySelectorAll(".mode-button").forEach((button) => {
    const isActive = button.dataset.mode === nextMode;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-selected", String(isActive));
  });

  const title = document.querySelector("#mode-title");
  const summary = document.querySelector("#mode-summary");
  if (title) {
    title.textContent = mode.title;
  }
  if (summary) {
    summary.textContent = mode.summary;
  }

  document.querySelectorAll(".diagram-section[data-diagram]").forEach((section) => {
    section.classList.toggle(
      "is-focused",
      nextMode !== "overview" && section.dataset.diagram === nextMode,
    );
  });
}

function bindControls() {
  document.querySelectorAll(".mode-button").forEach((button) => {
    button.addEventListener("click", () => {
      setMode(button.dataset.mode || "overview");
    });
  });
}

bindControls();
setMode("overview");
