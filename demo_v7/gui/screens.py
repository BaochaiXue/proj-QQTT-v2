"""The six demo_v7 screens (README 界面流程), one QWidget each.

Screens are intentionally dumb: they render state pushed into them by
``MainWindow`` (already on the Qt main thread) and surface operator intent as
Qt signals. They never talk to the IPC session directly, so every screen can
be constructed and driven headlessly (QT_QPA_PLATFORM=offscreen) without a
camera service.

Frame routing contract: ``MainWindow`` forwards every decoded-channel JPEG to
``on_frame(channel, jpeg_bytes)`` of the current screen; each screen picks the
channels it cares about (Capture: CH_RGB, Reposition: CH_OVERLAY, Formal: the
operator-selected channel). Artifact paths come from EVT_ARTIFACTS payloads
(``{name: abs_path}``); name interpretation is heuristic by design because the
artifact layout is owned by the service side.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from demo_v7.ipc.protocol import (
    CH_COMPOSITE,
    CH_DEPTH,
    CH_GAUSSIAN,
    CH_OVERLAY,
    CH_RGB,
)
from demo_v7.gui.mesh_view import MeshOrbitView
from demo_v7.gui.widgets import CaptionedImage, ImageView, ProgressTimeline

_ARTIFACT_GRID_COLUMNS = 3

# Warm-up stage plan shown before any progress event arrives. The service's
# EVT_PROGRESS vocabulary is "sam31_masks"/"shape_prior_submit"/"shape_prior"
# (with "<sub-stage> finished" details from case-dir milestones)/
# "shape_prior_ready"; the shape_prior details are fanned out onto the sp:*
# sub-rows so every chain step is its own live row (owner rule 2026-08-06:
# progress is VISUAL — spinners + elapsed — while text logs go to
# stdout + the run's log file, never a GUI text box).
WARMUP_STAGE_PLAN: list[tuple[str, str]] = [
    ("sam31_masks", "SAM3.1 三 mask 分割"),
    ("shape_prior_submit", "提交 shape-prior 任务"),
    ("sp:upscale", "  ├ 超分(upscale)"),
    ("sp:generate", "  ├ 生成(generate)"),
    ("sp:align", "  ├ 对齐(align)"),
    ("sp:sample", "  └ 补点采样(sample)"),
    ("shape_prior_ready", "shape-prior 就绪"),
]
# Backend id -> generate-row label (dialog vocabulary; sam3d is the default).
_GENERATE_ROW_LABELS = {
    "sam3d": "  ├ 生成(SAM3D)",
    "trellis2": "  ├ 生成(TRELLIS.2)",
    "none": "  ├ 生成(无,已跳过)",
}
# The shape_prior milestone order used to chain sub-row spinners.
_SP_SUB_ORDER = ("sp:upscale", "sp:generate", "sp:align", "sp:sample")


def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in (".png", ".jpg", ".jpeg")


# 补点 sources: (set name, checkbox label, display color). The colors are
# deliberately solid + distinct — the point of the view is source attribution.
_SAMPLING_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("observed", "frame-0 观测点云", "#8ab4f8"),
    ("surface", "表面补点(候选)", "#81c995"),
    ("interior", "体内补点(候选)", "#f2a25c"),
)
_SAMPLING_COLORS_U8 = {
    "observed": (0x8A, 0xB4, 0xF8),
    "surface": (0x81, 0xC9, 0x95),
    "interior": (0xF2, 0xA2, 0x5C),
}


class _ArtifactGrid(QWidget):
    """Scroll-friendly grid of captioned artifact stills."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(8, 8, 8, 8)
        self._grid.setSpacing(8)
        self._count = 0

    def clear(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._count = 0

    def add_images(self, paths: dict[str, str]) -> None:
        for name in sorted(paths):
            path = paths[name]
            if not _is_image(path):
                continue
            cell = CaptionedImage(name, path, parent=self)
            row, col = divmod(self._count, _ARTIFACT_GRID_COLUMNS)
            self._grid.addWidget(cell, row, col)
            self._count += 1


def _wrap_scroll(widget: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(widget)
    return scroll


class CaptureScreen(QWidget):
    """拍摄屏: live big view + 拍摄第一张 / 确认 / 重拍.

    While a frame-0 candidate is pending, live CH_RGB frames are ignored so a
    real camera (which keeps streaming) cannot overwrite the frozen candidate.
    On top of that, once the candidate png from EVT_ARTIFACTS(kind=frame0)
    arrives the big view is pinned to that exact image (the frozen live frame
    may already be one frame newer), and only 重拍 (back to live preview)
    releases the pin.
    """

    captureRequested = Signal()
    confirmRequested = Signal()
    retakeRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pending = False
        self._pinned = False
        self._view = ImageView("等待相机画面…", parent=self)
        self._hint = QLabel("请把物体和双手摆好,然后拍摄第一张。", self)
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._capture_btn = QPushButton("拍摄第一张", self)
        self._confirm_btn = QPushButton("确认", self)
        self._retake_btn = QPushButton("重拍", self)
        self._capture_btn.clicked.connect(self.captureRequested.emit)
        self._confirm_btn.clicked.connect(self.confirmRequested.emit)
        self._retake_btn.clicked.connect(self.retakeRequested.emit)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self._capture_btn)
        buttons.addWidget(self._retake_btn)
        buttons.addWidget(self._confirm_btn)
        buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addWidget(self._view, 1)
        layout.addWidget(self._hint)
        layout.addLayout(buttons)
        self.set_pending(False)

    def set_pending(self, pending: bool) -> None:
        """Toggle between live-preview mode and frozen-candidate mode."""
        self._pending = pending
        if not pending:
            # 重拍 (or any return to live preview) resumes live frames.
            self._pinned = False
        self._capture_btn.setVisible(not pending)
        self._confirm_btn.setVisible(pending)
        self._retake_btn.setVisible(pending)
        if pending:
            self._hint.setText("已拍摄候选 frame-0:确认后开始 warm-up,或重拍。")
        else:
            self._hint.setText("请把物体和双手摆好,然后拍摄第一张。")

    def set_busy(self, busy: bool) -> None:
        """Disable buttons while a command's ack is outstanding-ish states."""
        for btn in (self._capture_btn, self._confirm_btn, self._retake_btn):
            btn.setEnabled(not busy)

    def show_candidate(self, paths: dict[str, str]) -> None:
        """Pin the frozen frame-0 candidate from ARTIFACT_KIND_FRAME0 paths.

        Preference: the FRAME0_PENDING "candidate" png, then any rgb/color
        still, then any image. ``setImagePath`` cancels the queued live frame;
        the pin keeps later live CH_RGB frames out until 确认/重拍 resolves.
        """
        candidate: str | None = None
        rgb_like: str | None = None
        first: str | None = None
        for name in sorted(paths):
            path = paths[name]
            if not _is_image(path):
                continue
            lowered = (name + " " + path).lower()
            if candidate is None and "candidate" in lowered:
                candidate = path
            if rgb_like is None and ("rgb" in lowered or "color" in lowered):
                rgb_like = path
            if first is None:
                first = path
        best = candidate or rgb_like or first
        if best is not None and self._view.setImagePath(best):
            self._pinned = True

    def on_frame(self, channel: str, jpeg_bytes: bytes) -> None:
        if channel == CH_RGB and not self._pending and not self._pinned:
            self._view.setFrame(jpeg_bytes)


class WarmupScreen(QWidget):
    """warmup 屏: 人可离开 banner + live stage timeline + 查看结果.

    Pure visual progress: every chain step is a timeline row with a spinner
    and live elapsed while running, settling to ✓/✗ + duration. No text-log
    widget — event lines go to stdout and the run's log file (main_window).
    """

    viewResultsRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        banner = QLabel("frame-0 已确认:人和物体现在可以离开画面。", self)
        banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        banner.setStyleSheet(
            "background-color: #1e3a2f; color: #81c995; font-size: 18px;"
            " font-weight: bold; padding: 12px; border-radius: 6px;"
        )
        self._backend: str | None = None
        self._timeline = ProgressTimeline(self)
        self._timeline.setStages(WARMUP_STAGE_PLAN)
        self._results_btn = QPushButton("查看结果", self)
        self._results_btn.setEnabled(False)
        self._results_btn.clicked.connect(self.viewResultsRequested.emit)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self._results_btn)
        buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addWidget(banner)
        layout.addWidget(self._timeline)
        layout.addStretch(1)
        layout.addLayout(buttons)

    def set_shape_prior_backend(self, backend: str) -> None:
        """Rename the generate row for the run's backend (hello-ack echo)."""
        self._backend = str(backend)
        label = _GENERATE_ROW_LABELS.get(self._backend)
        if label is not None:
            self._timeline.setRowLabel("sp:generate", label)

    def on_progress(
        self, stage: str, detail: str, ok: bool, elapsed_ms: float | None
    ) -> None:
        """Fan service progress onto the visual rows + chain the spinners."""
        skipped = "跳过" in detail or "skipped" in detail or "disabled" in detail
        if stage == "sam31_masks":
            self._timeline.report(stage, detail, ok=ok, elapsed_ms=elapsed_ms)
            if ok:
                self._timeline.begin("shape_prior_submit")
        elif stage == "shape_prior_submit":
            self._timeline.report(stage, detail, ok=ok, elapsed_ms=elapsed_ms)
            if ok and not skipped:
                self._timeline.begin("sp:upscale")
        elif stage == "shape_prior":
            if skipped:
                for key in _SP_SUB_ORDER:
                    self._timeline.report(key, "跳过", ok=True)
            else:
                # Milestone details: "<sub-stage> finished" (no sample
                # milestone — sp:sample settles on shape_prior_ready).
                sub = f"sp:{detail.split(' ', 1)[0]}"
                if sub in _SP_SUB_ORDER:
                    self._timeline.report(sub, ok=ok, elapsed_ms=elapsed_ms)
                    if ok:
                        index = _SP_SUB_ORDER.index(sub)
                        if index + 1 < len(_SP_SUB_ORDER):
                            self._timeline.begin(_SP_SUB_ORDER[index + 1])
                else:
                    self._timeline.report(stage, detail, ok=ok, elapsed_ms=elapsed_ms)
        elif stage == "shape_prior_ready":
            if ok and not skipped:
                self._timeline.report("sp:sample")
            self._timeline.report(stage, detail, ok=ok, elapsed_ms=elapsed_ms)
        else:
            self._timeline.report(stage, detail, ok=ok, elapsed_ms=elapsed_ms)
        if not ok:
            # The chain is dead (retake or fatal follows): freeze the board
            # honestly — the failed row keeps its ✗, siblings stop spinning.
            self._timeline.stopAll()
        if stage == "shape_prior_ready" and ok:
            self._results_btn.setEnabled(True)

    def reset(self) -> None:
        """Fresh timeline for a new run (after 回到开始/重拍 cycles)."""
        self._results_btn.setEnabled(False)
        self._timeline.clear()
        self._timeline.setStages(WARMUP_STAGE_PLAN)
        if self._backend is not None:
            self.set_shape_prior_backend(self._backend)
        self._timeline.begin("sam31_masks")


class ReviewScreen(QWidget):
    """结果屏: Masks / Shape Prior tabs + 进入摆位.

    Shape Prior tab embeds an interactive mesh view (drag=旋转, 滚轮=缩放,
    双击=复位; Open3D offscreen — no video wait) with a 对齐后/原始 mesh
    switcher; render/alignment stills land below it (对齐叠加图 per README).
    """

    repositionRequested = Signal()
    regenGaussianRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tabs = QTabWidget(self)
        self._masks_grid = _ArtifactGrid()
        self._tabs.addTab(_wrap_scroll(self._masks_grid), "Masks")
        self._mesh_view = MeshOrbitView("等待 mesh(生成/对齐完成后可拖拽查看)…")
        self._mesh_pick = QComboBox()
        self._mesh_pick.addItem("对齐后 mesh")
        self._mesh_pick.addItem("原始生成 mesh")
        self._mesh_pick.currentIndexChanged.connect(self._on_mesh_pick)
        self._mesh_paths: dict[int, str] = {}
        mesh_bar = QHBoxLayout()
        mesh_bar.addWidget(QLabel("网格:"))
        mesh_bar.addWidget(self._mesh_pick)
        mesh_bar.addWidget(QLabel("拖拽=旋转 滚轮=缩放 双击=复位"))
        mesh_bar.addStretch(1)
        self._prior_grid = _ArtifactGrid()
        prior_page = QWidget()
        prior_layout = QVBoxLayout(prior_page)
        prior_layout.addLayout(mesh_bar)
        prior_layout.addWidget(self._mesh_view, 1)
        prior_layout.addWidget(self._prior_grid)
        self._tabs.addTab(prior_page, "Shape Prior")
        # 补点 tab: frame-0 observed object points vs shape-prior surface /
        # interior fill (candidates.npz), each source a solid color + toggle.
        self._sampling_view = MeshOrbitView("等待补点数据(warmup 完成后可查看)…")
        self._sampling_paths: dict[str, str] = {}
        self._sampling_loaded = False
        # Generation backend for this service run (hello-ack echo); "none"
        # switches the 补点 view to observed-only. Survives reset(): the
        # backend is fixed for the service's lifetime (回到开始 relaunches).
        self._shape_prior_backend: str | None = None
        self._sampling_checks: dict[str, QCheckBox] = {}
        sampling_bar = QHBoxLayout()
        for key, label, color in _SAMPLING_SOURCES:
            check = QCheckBox(label)
            check.setChecked(True)
            check.setStyleSheet(f"color: {color}; font-weight: bold;")
            check.toggled.connect(
                lambda on, name=key: self._sampling_view.setSourceVisible(name, on)
            )
            self._sampling_checks[key] = check
            sampling_bar.addWidget(check)
        sampling_bar.addWidget(QLabel("拖拽=旋转 滚轮=缩放 双击=复位"))
        sampling_bar.addStretch(1)
        sampling_page = QWidget()
        sampling_layout = QVBoxLayout(sampling_page)
        sampling_layout.addLayout(sampling_bar)
        sampling_layout.addWidget(self._sampling_view, 1)
        self._tabs.addTab(sampling_page, "补点")
        # Gaussian tab: TripoSplat 拣选 — turntable contact sheet + world
        # overlay stills, generation status, and 换seed re-roll.
        self._gaussian_status = QLabel("等待 gaussian 生成…", self)
        self._gaussian_status.setStyleSheet("color: #9aa0a6;")
        self._gaussian_regen_btn = QPushButton("换 seed 重新生成", self)
        self._gaussian_regen_btn.setEnabled(False)
        self._gaussian_regen_btn.clicked.connect(self.regenGaussianRequested.emit)
        gaussian_bar = QHBoxLayout()
        gaussian_bar.addWidget(self._gaussian_status, 1)
        gaussian_bar.addWidget(self._gaussian_regen_btn)
        self._gaussian_grid = _ArtifactGrid()
        gaussian_page = QWidget()
        gaussian_layout = QVBoxLayout(gaussian_page)
        gaussian_layout.addLayout(gaussian_bar)
        gaussian_layout.addWidget(_wrap_scroll(self._gaussian_grid), 1)
        self._tabs.addTab(gaussian_page, "Gaussian")
        self._reposition_btn = QPushButton("进入摆位", self)
        self._reposition_btn.clicked.connect(self.repositionRequested.emit)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self._reposition_btn)
        buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addWidget(self._tabs, 1)
        layout.addLayout(buttons)

    def set_shape_prior_backend(self, backend: str) -> None:
        """Adapt the tabs to the run's generation backend (hello-ack echo).

        Backend "none" produces no mesh and no candidates.npz: the mesh view
        says so instead of waiting forever, and the 补点 view renders the
        observed frame-0 points alone (they ARE the tracking structure in
        that mode).
        """
        self._shape_prior_backend = str(backend)
        if self._shape_prior_backend == "none":
            self._mesh_view.setPlaceholderText(
                "本次运行未生成 shape prior(backend=none)。"
            )
            self._maybe_build_sampling_view()

    def set_mask_artifacts(self, paths: dict[str, str]) -> None:
        self._masks_grid.clear()
        self._masks_grid.add_images(paths)

    def set_shape_prior_artifacts(self, paths: dict[str, str]) -> None:
        # Interactive mesh view: 对齐后 (final_mesh.glb) preferred, 原始生成
        # (object.glb) as the second combo entry. Videos are ignored — the
        # mesh is inspectable the moment the glb lands, no render wait.
        for name in sorted(paths):
            path = paths[name]
            lowered = name.lower()
            if lowered.endswith("candidates_npz") or path.lower().endswith(
                "candidates.npz"
            ):
                self._sampling_paths["candidates"] = path
                continue
            if not path.lower().endswith((".glb", ".obj", ".ply", ".gltf")):
                continue
            if "final" in lowered:
                self._mesh_paths[0] = path
            else:
                self._mesh_paths.setdefault(1, path)
        self._show_picked_mesh()
        self._maybe_build_sampling_view()
        self._prior_grid.add_images(paths)

    def set_frame0_artifacts(self, paths: dict[str, str]) -> None:
        """Frame-0 kind artifacts: keep the observed-object-points npz."""
        for name, path in paths.items():
            if "object_points" in name.lower():
                self._sampling_paths["observed"] = path
        self._maybe_build_sampling_view()

    def _maybe_build_sampling_view(self) -> None:
        """Load the 补点 view once its npz sources are on disk.

        Normally needs BOTH the observed points and candidates.npz; under
        backend "none" no candidates ever come, so the observed cloud renders
        alone (surface/interior legends show 0).
        """
        if self._sampling_loaded:
            return
        observed = self._sampling_paths.get("observed")
        candidates = self._sampling_paths.get("candidates")
        observed_only = candidates is None and self._shape_prior_backend == "none"
        if observed is None or (candidates is None and not observed_only):
            return
        try:
            import numpy as np  # noqa: PLC0415

            observed_npz = np.load(observed)
            empty = np.zeros((0, 3), dtype=np.float32)
            if observed_only:
                surface_points, interior_points = empty, empty
            else:
                candidates_npz = np.load(candidates)
                surface_points = candidates_npz["raw_surface_points"]
                interior_points = candidates_npz["raw_interior_points"]
            sets = {
                "observed": (
                    observed_npz["object_xyz_m"],
                    _SAMPLING_COLORS_U8["observed"],
                ),
                "surface": (
                    surface_points,
                    _SAMPLING_COLORS_U8["surface"],
                ),
                "interior": (
                    interior_points,
                    _SAMPLING_COLORS_U8["interior"],
                ),
            }
            self._sampling_view.setPointSets(sets)
            for key, label, _color in _SAMPLING_SOURCES:
                count = int(np.asarray(sets[key][0]).reshape(-1, 3).shape[0])
                self._sampling_checks[key].setText(f"{label} ({count})")
            self._sampling_loaded = True
        except Exception:
            # Missing/partial files stay pending; a later artifact event or
            # tab visit retries via the next call.
            pass

    def set_alignment_artifacts(self, paths: dict[str, str]) -> None:
        self._prior_grid.add_images(paths)

    def set_gaussian_artifacts(self, paths: dict[str, str]) -> None:
        """A generation (or re-roll) landed: refresh stills, enable 换seed."""
        self._gaussian_grid.clear()
        self._gaussian_grid.add_images(paths)
        self._gaussian_regen_btn.setEnabled(True)

    def set_gaussian_progress(self, detail: str, ok: bool) -> None:
        self._gaussian_status.setText(detail if ok else f"失败: {detail}")
        if not ok:
            # A failed roll still leaves the previous artifacts usable.
            self._gaussian_regen_btn.setEnabled(
                self._gaussian_grid._count > 0  # noqa: SLF001 (own widget)
            )

    def _on_mesh_pick(self, index: int) -> None:
        self._show_picked_mesh()

    def _show_picked_mesh(self) -> None:
        index = int(self._mesh_pick.currentIndex())
        path = self._mesh_paths.get(index) or self._mesh_paths.get(1 - index)
        if path is not None:
            self._mesh_view.setMeshPath(path)

    def reset(self) -> None:
        self._masks_grid.clear()
        self._prior_grid.clear()
        self._gaussian_grid.clear()
        self._gaussian_status.setText("等待 gaussian 生成…")
        self._gaussian_regen_btn.setEnabled(False)
        self._mesh_paths.clear()
        self._mesh_view.clear()
        self._sampling_paths.clear()
        self._sampling_loaded = False
        self._sampling_view.clear()
        for key, label, _color in _SAMPLING_SOURCES:
            check = self._sampling_checks[key]
            check.setChecked(True)
            check.setText(label)


class RepositionScreen(QWidget):
    """摆位屏: CH_OVERLAY live view + 开始正式追踪."""

    startFormalRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        hint = QLabel(
            "请把物体和双手摆回 frame-0 的位置(与半透明 mask 对齐),然后开始正式追踪。",
            self,
        )
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._view = ImageView("等待叠加画面…", parent=self)
        self._start_btn = QPushButton("开始正式追踪", self)
        self._start_btn.clicked.connect(self.startFormalRequested.emit)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self._start_btn)
        buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addWidget(self._view, 1)
        layout.addLayout(buttons)

    def on_frame(self, channel: str, jpeg_bytes: bytes) -> None:
        if channel == CH_OVERLAY:
            self._view.setFrame(jpeg_bytes)


class FormalScreen(QWidget):
    """正式屏: big view (复合/RGB/深度 switcher) + stats row + 停止.

    The switcher only changes which channel feeds the big view; the persistent
    RGB/depth dock keeps living regardless (MainWindow owns the dock).
    """

    stopRequested = Signal()

    _CHANNEL_CHOICES: list[tuple[str, str]] = [
        ("复合", CH_COMPOSITE),
        ("高斯", CH_GAUSSIAN),
        ("RGB", CH_RGB),
        ("深度", CH_DEPTH),
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._selected_channel = CH_COMPOSITE
        self._view = ImageView("等待复合视图…", parent=self)
        self._channel_group = QButtonGroup(self)
        switcher = QHBoxLayout()
        switcher.addStretch(1)
        for label, channel in self._CHANNEL_CHOICES:
            button = QPushButton(label, self)
            button.setCheckable(True)
            button.setChecked(channel == self._selected_channel)
            button.clicked.connect(
                lambda _checked=False, ch=channel: self._select_channel(ch)
            )
            self._channel_group.addButton(button)
            switcher.addWidget(button)
        switcher.addStretch(1)
        self._channel_group.setExclusive(True)
        self._stats = QLabel("等待统计…", self)
        self._stats.setStyleSheet("color: #9aa0a6; font-family: monospace;")
        self._stop_btn = QPushButton("停止", self)
        self._stop_btn.clicked.connect(self.stopRequested.emit)
        bottom = QHBoxLayout()
        bottom.addWidget(self._stats, 1)
        bottom.addWidget(self._stop_btn)
        layout = QVBoxLayout(self)
        layout.addLayout(switcher)
        layout.addWidget(self._view, 1)
        layout.addLayout(bottom)

    def _select_channel(self, channel: str) -> None:
        self._selected_channel = channel
        self._view.clear()

    def selected_channel(self) -> str:
        return self._selected_channel

    def on_frame(self, channel: str, jpeg_bytes: bytes) -> None:
        if channel == self._selected_channel:
            self._view.setFrame(jpeg_bytes)

    def set_stats(self, payload: dict) -> None:
        """Render one EVT_FORMAL_STATS payload ({"seq","fps":{...},"latency_ms"})."""
        parts: list[str] = []
        seq = payload.get("seq")
        if seq is not None:
            parts.append(f"帧 {seq}")
        latency = payload.get("latency_ms")
        if isinstance(latency, (int, float)):
            parts.append(f"延迟 {latency:.0f} ms")
        fps = payload.get("fps")
        if isinstance(fps, dict) and fps:
            fps_text = "  ".join(
                f"{name}:{value:.1f}"
                for name, value in sorted(fps.items())
                if isinstance(value, (int, float))
            )
            if fps_text:
                parts.append("fps  " + fps_text)
        self._stats.setText("  |  ".join(parts) if parts else "等待统计…")


class FinishedScreen(QWidget):
    """结束屏: run dir + 回到开始."""

    restartRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = QLabel("本次运行已结束。", self)
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setStyleSheet("font-size: 20px; font-weight: bold;")
        self._run_dir = QLabel("", self)
        self._run_dir.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._run_dir.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._run_dir.setStyleSheet("font-family: monospace; color: #9aa0a6;")
        restart_btn = QPushButton("回到开始", self)
        restart_btn.clicked.connect(self.restartRequested.emit)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(restart_btn)
        buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addStretch(1)
        layout.addWidget(self._title)
        layout.addWidget(self._run_dir)
        layout.addLayout(buttons)
        layout.addStretch(1)

    def set_run_dir(self, run_dir: str | None) -> None:
        self._run_dir.setText(f"输出目录: {run_dir}" if run_dir else "输出目录未知")

    def set_title(self, text: str) -> None:
        self._title.setText(text)
