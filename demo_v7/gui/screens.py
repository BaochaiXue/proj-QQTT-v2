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
    QSplitter,
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
from demo_v7.gui.i18n import tr
from demo_v7.gui.widgets import (
    CaptionedImage,
    ImageView,
    InfoDot,
    ProgressTimeline,
)

_ARTIFACT_GRID_COLUMNS = 3

# Warm-up stage plan shown before any progress event arrives. The service's
# EVT_PROGRESS vocabulary is "sam31_masks"/"shape_prior_submit"/"shape_prior"
# (with "<sub-stage> finished" details from case-dir milestones)/
# "shape_prior_ready"; the shape_prior details are fanned out onto the sp:*
# sub-rows so every chain step is its own live row (owner rule 2026-08-06:
# progress is VISUAL — spinners + elapsed — while text logs go to
# stdout + the run's log file, never a GUI text box).
WARMUP_STAGE_PLAN: list[tuple[str, tuple[str, str]]] = [
    ("sam31_masks", ("SAM3.1 三 mask 分割", "SAM3.1 3-mask segmentation")),
    ("shape_prior_submit", ("提交 shape-prior 任务", "Submit shape-prior job")),
    ("sp:upscale", ("  ├ 超分(upscale)", "  ├ Upscale (SD ×4)")),
    ("sp:generate", ("  ├ 生成(generate)", "  ├ Generate")),
    ("sp:align", ("  ├ 对齐(align)", "  ├ Align")),
    ("sp:sample", ("  └ 补点采样(sample)", "  └ Point sampling")),
    ("shape_prior_ready", ("shape-prior 就绪", "Shape prior ready")),
    # Rides in parallel with the chain (camera GPU); completes last —
    # alignment parks until the chain is READY.
    ("gaussian", ("Gaussian 生成(TripoSplat)", "Gaussian generation (TripoSplat)")),
]
# Backend id -> generate-row label (dialog vocabulary; sam3d is the default).
_GENERATE_ROW_LABELS = {
    "sam3d": ("  ├ 生成(SAM3D)", "  ├ Generate (SAM3D)"),
    "trellis2": ("  ├ 生成(TRELLIS.2)", "  ├ Generate (TRELLIS.2)"),
    "none": ("  ├ 生成(无,已跳过)", "  ├ Generate (none, skipped)"),
}
# The honest upscale-row label when the run disabled the SD ×4 stage (the
# passthrough still crops + emits the same milestone, so the row stays).
_UPSCALE_OFF_LABEL = ("  ├ 裁剪(无超分)", "  ├ Crop only (no upscale)")
# The shape_prior milestone order used to chain sub-row spinners.
_SP_SUB_ORDER = ("sp:upscale", "sp:generate", "sp:align", "sp:sample")


def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in (".png", ".jpg", ".jpeg")


# 补点 sources: (set name, checkbox label, display color). The colors are
# deliberately solid + distinct — the point of the view is source attribution.
_SAMPLING_SOURCES: tuple[tuple[str, tuple[str, str], str], ...] = (
    ("observed", ("frame-0 观测点云", "frame-0 observed points"), "#8ab4f8"),
    ("surface", ("表面补点(候选)", "surface fill (candidates)"), "#81c995"),
    ("interior", ("体内补点(候选)", "interior fill (candidates)"), "#f2a25c"),
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
        self._view = ImageView(tr("等待相机画面…", "Waiting for the camera feed…"), parent=self)
        self._hint = QLabel(
            tr(
                "请把物体和双手摆好,然后拍摄第一张。",
                "Position the object and both hands, then capture frame-0.",
            ),
            self,
        )
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._capture_btn = QPushButton(tr("拍摄第一张", "Capture frame-0"), self)
        self._confirm_btn = QPushButton(tr("确认", "Confirm"), self)
        self._retake_btn = QPushButton(tr("重拍", "Retake"), self)
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
            self._hint.setText(
                tr(
                    "已拍摄候选 frame-0:确认后开始 warm-up,或重拍。",
                    "Candidate frame-0 captured: confirm to start warm-up, or retake.",
                )
            )
        else:
            self._hint.setText(
                tr(
                    "请把物体和双手摆好,然后拍摄第一张。",
                    "Position the object and both hands, then capture frame-0.",
                )
            )

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
        banner = QLabel(
            tr(
                "frame-0 已确认:人和物体现在可以离开画面。",
                "frame-0 confirmed: people and the object may leave the view.",
            ),
            self,
        )
        banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        banner.setStyleSheet(
            "background-color: #1e3a2f; color: #81c995; font-size: 18px;"
            " font-weight: bold; padding: 12px; border-radius: 6px;"
        )
        self._backend: str | None = None
        self._upscale: bool | None = None
        self._gaussian_backend: str | None = None
        self._timeline = ProgressTimeline(self)
        self._timeline.setStages(
            [(key, tr(*pair)) for key, pair in WARMUP_STAGE_PLAN]
        )
        self._results_btn = QPushButton(tr("查看结果", "View results"), self)
        self._results_btn.setEnabled(False)
        self._results_btn.clicked.connect(self.viewResultsRequested.emit)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self._results_btn)
        buttons.addStretch(1)
        banner_info = InfoDot(
            "warmup 在做什么:对确认的 frame-0 依次跑 SAM3.1 分割 → 生成 "
            "shape prior mesh → 对齐到观测点云 → 体积补点采样。每行是一个"
            "阶段的实时进度;完成后到结果页检查。",
            "What warmup does: on the confirmed frame-0 it runs SAM3.1 "
            "segmentation → shape-prior mesh generation → alignment to the "
            "observed points → volumetric point sampling. Each row is one "
            "stage's live progress; inspect the outputs on the results "
            "screen when done.",
            self,
        )
        banner_row = QHBoxLayout()
        banner_row.addWidget(banner, 1)
        banner_row.addWidget(banner_info)
        layout = QVBoxLayout(self)
        layout.addLayout(banner_row)
        layout.addWidget(self._timeline)
        layout.addStretch(1)
        layout.addLayout(buttons)

    def set_shape_prior_backend(self, backend: str) -> None:
        """Rename the generate row for the run's backend (hello-ack echo)."""
        self._backend = str(backend)
        pair = _GENERATE_ROW_LABELS.get(self._backend)
        if pair is not None:
            self._timeline.setRowLabel("sp:generate", tr(*pair))

    def set_shape_prior_upscale(self, enabled: bool) -> None:
        """Honest upscale row label (hello-ack echo): off = crop only."""
        self._upscale = bool(enabled)
        if not self._upscale:
            self._timeline.setRowLabel("sp:upscale", tr(*_UPSCALE_OFF_LABEL))

    def set_gaussian_backend(self, backend: str) -> None:
        """Settle the gaussian row when the run disabled the feature."""
        self._gaussian_backend = str(backend)
        if self._gaussian_backend == "none":
            self._timeline.report(
                "gaussian", tr("已关闭(源选择)", "off (source-select)"), ok=True
            )

    def on_gaussian_progress(self, detail: str, ok: bool) -> None:
        """Drive the gaussian row (NOT via on_progress: the feature is
        fail-soft, so its ✗ must never stopAll() the live chain rows)."""
        if not ok:
            self._timeline.report("gaussian", detail, ok=False)
        elif detail.startswith("gaussian 就绪"):
            self._timeline.report("gaussian", detail, ok=True)
        else:
            # 生成中/采样 n/N/等待对齐: keep the spinner, refresh the detail
            # (begin() preserves the first call's start for the elapsed).
            self._timeline.begin("gaussian", detail)

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
                    self._timeline.report(key, tr("跳过", "skipped"), ok=True)
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
        self._timeline.setStages(
            [(key, tr(*pair)) for key, pair in WARMUP_STAGE_PLAN]
        )
        if self._backend is not None:
            self.set_shape_prior_backend(self._backend)
        if self._upscale is not None:
            self.set_shape_prior_upscale(self._upscale)
        if self._gaussian_backend is not None:
            self.set_gaussian_backend(self._gaussian_backend)
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
        self._mesh_view = MeshOrbitView(
            tr(
                "等待 mesh(生成/对齐完成后可拖拽查看)…",
                "Waiting for mesh (drag to inspect once generated/aligned)…",
            )
        )
        self._mesh_pick = QComboBox()
        self._mesh_pick.addItem(tr("对齐后 mesh", "Aligned mesh"))
        self._mesh_pick.addItem(tr("原始生成 mesh", "Raw generated mesh"))
        self._mesh_pick.currentIndexChanged.connect(self._on_mesh_pick)
        self._mesh_paths: dict[int, str] = {}
        mesh_bar = QHBoxLayout()
        mesh_bar.addWidget(QLabel(tr("网格:", "Mesh:")))
        mesh_bar.addWidget(
            InfoDot(
                "对齐后 mesh:生成的网格经对齐阶段(位姿+尺度+非刚性变形)"
                "贴合到 frame-0 观测,是下游补点/物理用的版本;原始生成 mesh:"
                "生成器的标准姿态原始输出。",
                "Aligned mesh: the generated mesh after the alignment stage "
                "(pose + scale + non-rigid deformation) fitted to the "
                "frame-0 observation — the version downstream sampling/"
                "physics uses. Raw generated mesh: the generator's original "
                "canonical-pose output.",
            )
        )
        mesh_bar.addWidget(self._mesh_pick)
        mesh_bar.addWidget(
            QLabel(tr("拖拽=旋转 滚轮=缩放 双击=复位", "drag=rotate wheel=zoom double-click=reset"))
        )
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
        self._sampling_view = MeshOrbitView(
            tr(
                "等待补点数据(warmup 完成后可查看)…",
                "Waiting for sampling data (viewable after warmup)…",
            )
        )
        self._sampling_paths: dict[str, str] = {}
        self._sampling_loaded = False
        # Generation backend for this service run (hello-ack echo); "none"
        # switches the 补点 view to observed-only. Survives reset(): the
        # backend is fixed for the service's lifetime (回到开始 relaunches).
        self._shape_prior_backend: str | None = None
        # Gaussian generator for this run (hello-ack echo, same lifetime
        # semantics); "none" turns the Gaussian tab into a static notice.
        self._gaussian_backend: str | None = None
        self._sampling_checks: dict[str, QCheckBox] = {}
        sampling_bar = QHBoxLayout()
        for key, pair, color in _SAMPLING_SOURCES:
            check = QCheckBox(tr(*pair))
            check.setChecked(True)
            check.setStyleSheet(f"color: {color}; font-weight: bold;")
            check.toggled.connect(
                lambda on, name=key: self._sampling_view.setSourceVisible(name, on)
            )
            self._sampling_checks[key] = check
            sampling_bar.addWidget(check)
        sampling_bar.addWidget(
            InfoDot(
                "补点(point sampling):蓝=frame-0 观测到的物体表面点;绿=按 "
                "mesh 表面采样的候选点;橙=mesh 体积内部的候选点。补点填充相机"
                "看不到的部分,组成下游物理跟踪的完整点结构。",
                "Point sampling: blue = object points observed at frame-0; "
                "green = candidates sampled on the mesh surface; orange = "
                "candidates inside the mesh volume. Sampling fills in what "
                "the camera cannot see, forming the complete point structure "
                "for downstream physics tracking.",
            )
        )
        sampling_bar.addWidget(
            QLabel(tr("拖拽=旋转 滚轮=缩放 双击=复位", "drag=rotate wheel=zoom double-click=reset"))
        )
        sampling_bar.addStretch(1)
        sampling_page = QWidget()
        sampling_layout = QVBoxLayout(sampling_page)
        sampling_layout.addLayout(sampling_bar)
        sampling_layout.addWidget(self._sampling_view, 1)
        self._tabs.addTab(sampling_page, tr("补点", "Sampling"))
        # Gaussian tab: TripoSplat 拣选 — turntable contact sheet + world
        # overlay stills, generation status, and 换seed re-roll.
        self._gaussian_status = QLabel(
            tr("等待 gaussian 生成…", "Waiting for gaussian generation…"), self
        )
        self._gaussian_status.setStyleSheet("color: #9aa0a6;")
        self._gaussian_regen_btn = QPushButton(
            tr("换 seed 重新生成", "Regenerate with a new seed"), self
        )
        self._gaussian_regen_btn.setEnabled(False)
        self._gaussian_regen_btn.clicked.connect(self.regenGaussianRequested.emit)
        gaussian_bar = QHBoxLayout()
        gaussian_bar.addWidget(
            InfoDot(
                "3D gaussians(高斯泼溅):TripoSplat 从 frame-0 生成、对齐到"
                "场景,正式期在「高斯」频道实时渲染。生成有随机性 —— 对结果"
                "不满意可换 seed 重新生成,不影响 mesh 链。",
                "3D gaussians (splats): generated by TripoSplat from "
                "frame-0 and aligned into the scene; the formal phase "
                "renders them live in the Gaussian channel. Generation is "
                "stochastic — re-roll with a new seed if unhappy; the mesh "
                "chain is unaffected.",
            )
        )
        gaussian_bar.addWidget(self._gaussian_status, 1)
        gaussian_bar.addWidget(self._gaussian_regen_btn)
        # Draggable 3D inspection (owner ask): the aligned splats as a
        # per-point-colored cloud in the same orbit widget the mesh uses —
        # 拖拽=旋转 滚轮=缩放 双击=复位; combo flips 世界系/生成系.
        self._gaussian_ply_paths: dict[int, str] = {}
        self._gaussian_view = MeshOrbitView(
            tr("等待 gaussian 生成…", "Waiting for gaussian generation…")
        )
        self._gaussian_view_pick = QComboBox()
        self._gaussian_view_pick.addItem(tr("对齐后(世界系)", "Aligned (world)"))
        self._gaussian_view_pick.addItem(tr("原始(生成系)", "Raw (canonical)"))
        self._gaussian_view_pick.currentIndexChanged.connect(
            lambda _index: self._show_picked_gaussian()
        )
        gaussian_view_bar = QHBoxLayout()
        gaussian_view_bar.addWidget(self._gaussian_view_pick)
        gaussian_view_bar.addWidget(
            QLabel(
                tr(
                    "拖拽=旋转 滚轮=缩放 双击=复位",
                    "drag=rotate wheel=zoom double-click=reset",
                )
            )
        )
        gaussian_view_bar.addStretch(1)
        gaussian_view_page = QWidget()
        gaussian_view_layout = QVBoxLayout(gaussian_view_page)
        gaussian_view_layout.setContentsMargins(0, 0, 0, 0)
        gaussian_view_layout.addLayout(gaussian_view_bar)
        gaussian_view_layout.addWidget(self._gaussian_view, 1)
        self._gaussian_grid = _ArtifactGrid()
        gaussian_split = QSplitter(Qt.Orientation.Vertical)
        gaussian_split.addWidget(gaussian_view_page)
        gaussian_split.addWidget(_wrap_scroll(self._gaussian_grid))
        gaussian_split.setStretchFactor(0, 3)
        gaussian_split.setStretchFactor(1, 1)
        gaussian_page = QWidget()
        gaussian_layout = QVBoxLayout(gaussian_page)
        gaussian_layout.addLayout(gaussian_bar)
        gaussian_layout.addWidget(gaussian_split, 1)
        self._tabs.addTab(gaussian_page, "Gaussian")
        self._reposition_btn = QPushButton(tr("进入摆位", "Enter repositioning"), self)
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
                tr(
                    "本次运行未生成 shape prior(backend=none)。",
                    "This run generated no shape prior (backend=none).",
                )
            )
            self._maybe_build_sampling_view()

    def set_gaussian_backend(self, backend: str) -> None:
        """Adapt the Gaussian tab to the run's generator (hello-ack echo)."""
        self._gaussian_backend = str(backend)
        if self._gaussian_backend == "none":
            self._apply_gaussian_disabled_notice()

    def _apply_gaussian_disabled_notice(self) -> None:
        self._gaussian_status.setText(
            tr(
                "本次运行未生成 gaussian(已在源选择关闭)。",
                "Gaussians are off for this run (source-select choice).",
            )
        )
        self._gaussian_regen_btn.setEnabled(False)

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
            for key, pair, _color in _SAMPLING_SOURCES:
                count = int(np.asarray(sets[key][0]).reshape(-1, 3).shape[0])
                self._sampling_checks[key].setText(f"{tr(*pair)} ({count})")
            self._sampling_loaded = True
        except Exception:
            # Missing/partial files stay pending; a later artifact event or
            # tab visit retries via the next call.
            pass

    def set_alignment_artifacts(self, paths: dict[str, str]) -> None:
        self._prior_grid.add_images(paths)

    def set_gaussian_artifacts(self, paths: dict[str, str]) -> None:
        """A generation (or re-roll) landed: refresh stills + 3D view."""
        self._gaussian_grid.clear()
        self._gaussian_grid.add_images(paths)
        self._gaussian_regen_btn.setEnabled(True)
        if "world_ply" in paths:
            self._gaussian_ply_paths[0] = str(paths["world_ply"])
        if "ply" in paths:
            self._gaussian_ply_paths[1] = str(paths["ply"])
        self._show_picked_gaussian()

    def _show_picked_gaussian(self) -> None:
        index = int(self._gaussian_view_pick.currentIndex())
        path = self._gaussian_ply_paths.get(index) or self._gaussian_ply_paths.get(
            1 - index
        )
        if path is None:
            return
        try:
            import numpy as np

            from demo_v7.service.gaussian_utils import load_gaussian_ply

            splats = load_gaussian_ply(path)
            keep = splats.opacities > 0.3  # drop floaters (same as align)
            points = splats.means[keep]
            colors = (np.clip(splats.colors[keep], 0.0, 1.0) * 255.0).astype(
                np.uint8
            )
            self._gaussian_view.setColoredCloud(points, colors)
        except Exception as exc:
            self._gaussian_view.setPlaceholderText(
                tr("gaussian 预览加载失败", "gaussian preview failed") + f": {exc}"
            )

    def set_gaussian_progress(self, detail: str, ok: bool) -> None:
        self._gaussian_status.setText(
            detail if ok else tr("失败", "Failed") + f": {detail}"
        )
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
        self._gaussian_ply_paths.clear()
        self._gaussian_view.clear()
        if self._gaussian_backend == "none":
            self._apply_gaussian_disabled_notice()
        else:
            self._gaussian_status.setText(
                tr("等待 gaussian 生成…", "Waiting for gaussian generation…")
            )
            self._gaussian_regen_btn.setEnabled(False)
        self._mesh_paths.clear()
        self._mesh_view.clear()
        self._sampling_paths.clear()
        self._sampling_loaded = False
        self._sampling_view.clear()
        for key, pair, _color in _SAMPLING_SOURCES:
            check = self._sampling_checks[key]
            check.setChecked(True)
            check.setText(tr(*pair))


class RepositionScreen(QWidget):
    """摆位屏: CH_OVERLAY live view + 开始正式追踪."""

    startFormalRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        hint = QLabel(
            tr(
                "请把物体和双手摆回 frame-0 的位置(与半透明 mask 对齐),然后开始正式追踪。",
                "Move the object and hands back to the frame-0 pose (match the translucent masks), then start formal tracking.",
            ),
            self,
        )
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._view = ImageView(tr("等待叠加画面…", "Waiting for the overlay view…"), parent=self)
        self._start_btn = QPushButton(tr("开始正式追踪", "Start formal tracking"), self)
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

    _CHANNEL_CHOICES: list[tuple[tuple[str, str], str]] = [
        (("复合", "Composite"), CH_COMPOSITE),
        (("高斯", "Gaussian"), CH_GAUSSIAN),
        (("RGB", "RGB"), CH_RGB),
        (("深度", "Depth"), CH_DEPTH),
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._selected_channel = CH_COMPOSITE
        self._view = ImageView(tr("等待复合视图…", "Waiting for the composite view…"), parent=self)
        self._channel_group = QButtonGroup(self)
        switcher = QHBoxLayout()
        switcher.addStretch(1)
        for pair, channel in self._CHANNEL_CHOICES:
            button = QPushButton(tr(*pair), self)
            button.setCheckable(True)
            button.setChecked(channel == self._selected_channel)
            button.clicked.connect(
                lambda _checked=False, ch=channel: self._select_channel(ch)
            )
            self._channel_group.addButton(button)
            switcher.addWidget(button)
        switcher.addWidget(
            InfoDot(
                "大画面显示的频道:复合=跟踪可视化叠加(物体/双手);高斯="
                "gaussian 实时渲染;RGB=原始彩色;深度=深度伪彩。右侧停靠栏的 "
                "RGB/深度小窗始终保留。",
                "Which channel the big view shows: Composite = tracking "
                "overlay (object/hands); Gaussian = live gaussian render; "
                "RGB = raw color; Depth = colorized depth. The docked "
                "RGB/depth thumbnails stay on regardless.",
            )
        )
        switcher.addStretch(1)
        self._channel_group.setExclusive(True)
        self._stats = QLabel(tr("等待统计…", "Waiting for stats…"), self)
        self._stats.setStyleSheet("color: #9aa0a6; font-family: monospace;")
        self._stop_btn = QPushButton(tr("停止", "Stop"), self)
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
            parts.append(tr("帧", "frame") + f" {seq}")
        latency = payload.get("latency_ms")
        if isinstance(latency, (int, float)):
            parts.append(tr("延迟", "latency") + f" {latency:.0f} ms")
        fps = payload.get("fps")
        if isinstance(fps, dict) and fps:
            fps_text = "  ".join(
                f"{name}:{value:.1f}"
                for name, value in sorted(fps.items())
                if isinstance(value, (int, float))
            )
            if fps_text:
                parts.append("fps  " + fps_text)
        self._stats.setText(
            "  |  ".join(parts) if parts else tr("等待统计…", "Waiting for stats…")
        )


class FinishedScreen(QWidget):
    """结束屏: run dir + 回到开始."""

    restartRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = QLabel(tr("本次运行已结束。", "This run has finished."), self)
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setStyleSheet("font-size: 20px; font-weight: bold;")
        self._run_dir = QLabel("", self)
        self._run_dir.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._run_dir.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._run_dir.setStyleSheet("font-family: monospace; color: #9aa0a6;")
        restart_btn = QPushButton(tr("回到开始", "Back to start"), self)
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
        self._run_dir.setText(
            tr("输出目录", "Output dir") + f": {run_dir}"
            if run_dir
            else tr("输出目录未知", "Output dir unknown")
        )

    def set_title(self, text: str) -> None:
        self._title.setText(text)
