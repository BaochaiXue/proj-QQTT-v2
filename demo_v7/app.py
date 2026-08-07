#!/usr/bin/env python3
"""demo_v7 GUI entrypoint: source selection + session spawn + main window.

Usage (see ``demo_v7/README.md``)::

    python demo_v7/app.py
    python demo_v7/app.py --source fake-live --fake-live-case data_collect/<case>

Responsibilities kept here (and only here):

- argv parsing (``--source`` / ``--fake-live-case`` / ``--base-path``); when
  ``--source`` is omitted, a modal SourceSelect dialog asks for
  真实相机 (real) vs fake-live + case folder.
- constructing ``OrchestratorSession`` (imported lazily so this file — and the
  headless GUI smoke that runs with a stub session — never needs the service
  stack) and handing it to ``MainWindow``.
- the 回到开始 loop: tear the window+session down, re-ask the source, relaunch.

``QT_QPA_PLATFORM`` is respected as-is (never overridden), so
``QT_QPA_PLATFORM=offscreen`` drives the whole GUI headlessly.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

# Keep this repo at the front of the import path when the script is launched
# from another working directory (same bootstrap as demo_v6_2/main.py); must
# run before any ``demo_v7.*`` import when executed as a script.
_BOOTSTRAP_REPO_ROOT_STR = str(Path(__file__).resolve().parents[1])
if _BOOTSTRAP_REPO_ROOT_STR in sys.path:
    sys.path.remove(_BOOTSTRAP_REPO_ROOT_STR)
sys.path.insert(0, _BOOTSTRAP_REPO_ROOT_STR)

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from demo_v7.gui import i18n
from demo_v7.gui.i18n import tr
from demo_v7.gui.main_window import MainWindow, shutdown_session_on_thread
from demo_v7.gui.widgets import InfoDot
from demo_v7.service.backend_options import (  # import-light (stdlib only)
    BACKEND_NONE,
    BACKEND_SAM3D,
    BACKEND_TRELLIS2,
    DEFAULT_SHAPE_PRIOR_BACKEND,
    SHAPE_PRIOR_BACKENDS,
    normalize_backend,
    normalize_upscale,
)
from demo_v7.service.gaussian_options import (  # import-light (stdlib only)
    DEFAULT_GAUSSIAN_BACKEND,
    GAUSSIAN_BACKENDS,
    GAUSSIAN_NONE,
    GAUSSIAN_TRIPOSPLAT,
    normalize_gaussian_backend,
)

SOURCE_REAL = "real"
SOURCE_FAKE_LIVE = "fake-live"

# Combo order + GUI label (zh, en) pairs for the generation backend;
# translated at dialog (re)build time, never at import (i18n contract).
_BACKEND_LABELS: tuple[tuple[str, tuple[str, str]], ...] = (
    (BACKEND_TRELLIS2, ("TRELLIS.2(默认)", "TRELLIS.2 (default)")),
    (BACKEND_SAM3D, ("SAM3D", "SAM3D")),
    (BACKEND_NONE, ("无(不生成 shape prior)", "None (no shape prior)")),
)
# Combo order + GUI label (zh, en) pairs for the gaussian generator; one
# real model today (TripoSplat), but the slot is a first-class run option so
# future generators drop in without GUI surgery.
_GAUSSIAN_LABELS: tuple[tuple[str, tuple[str, str]], ...] = (
    (GAUSSIAN_TRIPOSPLAT, ("TripoSplat(默认)", "TripoSplat (default)")),
    (GAUSSIAN_NONE, ("无(不生成 gaussian)", "None (no gaussians)")),
)
# Language combo entries: id -> native display name (never translated).
_LANGUAGE_LABELS: tuple[tuple[str, str], ...] = (
    (i18n.LANG_ZH, "简体中文"),
    (i18n.LANG_EN, "English"),
)

# session.shutdown() can block on a dying service / chunk-stream tail; the
# GUI waits at most this long (daemon thread keeps draining), then continues.
_SHUTDOWN_WAIT_S = 20.0
_SHUTDOWN_POLL_MS = 100

# Dark-neutral default styling (contract: DESIGN_CONTRACTS.md gui section).
_APP_STYLESHEET = """
QWidget { background-color: #202124; color: #e8eaed; font-size: 14px; }
QMainWindow, QDialog { background-color: #202124; }
QPushButton {
    background-color: #3c4043; border: 1px solid #5f6368;
    border-radius: 4px; padding: 6px 18px;
}
QPushButton:hover { background-color: #4a4e52; }
QPushButton:disabled { color: #5f6368; border-color: #3c4043; }
QPushButton:checked { background-color: #1a73e8; border-color: #1a73e8; }
QPlainTextEdit, QLineEdit {
    background-color: #16181c; border: 1px solid #3c4043; border-radius: 3px;
}
QTabWidget::pane { border: 1px solid #3c4043; }
QTabBar::tab { background: #2a2d31; padding: 6px 16px; }
QTabBar::tab:selected { background: #3c4043; }
QDockWidget { titlebar-close-icon: none; }
QScrollArea { border: none; }
QStatusBar { color: #9aa0a6; }
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="demo_v7 GUI (PySide6) for demo_v6_2.")
    parser.add_argument(
        "--source",
        choices=(SOURCE_REAL, SOURCE_FAKE_LIVE),
        default=None,
        help="Camera source; omit to pick interactively at startup.",
    )
    parser.add_argument(
        "--fake-live-case",
        type=Path,
        default=None,
        help="Raw data_collect case folder for fake-live replay.",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=None,
        help="Run output base path (defaults to the v7/v6.2 config default).",
    )
    parser.add_argument(
        "--shape-prior-backend",
        choices=SHAPE_PRIOR_BACKENDS,
        default=None,
        help=(
            "Shape-prior generation backend; omit to pick interactively "
            "(config default preselected)."
        ),
    )
    parser.add_argument(
        "--shape-prior-upscale",
        choices=("on", "off"),
        default=None,
        help=(
            "Upscale (SD x4) stage: on keeps the v6.2 chain, off uses the "
            "crop-only passthrough (faster warmup); omit to pick "
            "interactively (config default preselected)."
        ),
    )
    parser.add_argument(
        "--gaussian-backend",
        choices=GAUSSIAN_BACKENDS,
        default=None,
        help=(
            "Gaussian-splats generator; omit to pick interactively "
            "(config default preselected)."
        ),
    )
    parser.add_argument(
        "--language",
        choices=i18n.LANGUAGES,
        default=None,
        help=(
            "GUI language (zh = 简体中文, en = English); omit to pick "
            "interactively (config default preselected)."
        ),
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help=(
            "Record the run (real camera) into this directory as a "
            "data_collect-format fake-live case; must not exist or be empty."
        ),
    )
    return parser


def config_default_source_and_case() -> tuple[str, Path | None]:
    """Dialog defaults from config: v7 session.* -> v6.2 input.* fallback.

    Light imports only (yaml + the v6.2 config module) so the dialog opens
    instantly; the heavy service stack still loads lazily after 开始.
    """
    import yaml

    from demo_v6_2.orchestration.main_config import (
        DEFAULT_FAKE_LIVE_CASE,
        DEFAULT_INPUT_SOURCE,
    )

    session_cfg: dict[str, Any] = {}
    try:
        loaded = yaml.safe_load(
            (Path(__file__).parent / "config" / "default.yaml").read_text()
        )
        if isinstance(loaded, dict) and isinstance(loaded.get("session"), dict):
            session_cfg = loaded["session"]
    except Exception:
        pass
    source_v62 = str(session_cfg.get("source") or DEFAULT_INPUT_SOURCE)
    source = SOURCE_REAL if source_v62 == "live" else SOURCE_FAKE_LIVE
    case = session_cfg.get("fake_live_case") or DEFAULT_FAKE_LIVE_CASE
    return source, Path(case) if case else None


def config_default_shape_prior_backend() -> str:
    """Dialog default for the generation backend (config, then sam3d)."""
    import yaml

    try:
        loaded = yaml.safe_load(
            (Path(__file__).parent / "config" / "default.yaml").read_text()
        )
        if isinstance(loaded, dict) and isinstance(loaded.get("session"), dict):
            return normalize_backend(loaded["session"].get("shape_prior_backend"))
    except Exception:
        pass
    return DEFAULT_SHAPE_PRIOR_BACKEND


def config_default_shape_prior_upscale() -> bool:
    """Dialog default for the upscale toggle (config, then on)."""
    import yaml

    try:
        loaded = yaml.safe_load(
            (Path(__file__).parent / "config" / "default.yaml").read_text()
        )
        if isinstance(loaded, dict) and isinstance(loaded.get("session"), dict):
            return normalize_upscale(loaded["session"].get("shape_prior_upscale"))
    except Exception:
        pass
    return True


def config_default_gaussian_backend() -> str:
    """Dialog default for the gaussian generator (config, then triposplat)."""
    import yaml

    try:
        loaded = yaml.safe_load(
            (Path(__file__).parent / "config" / "default.yaml").read_text()
        )
        if isinstance(loaded, dict) and isinstance(loaded.get("session"), dict):
            return normalize_gaussian_backend(
                loaded["session"].get("gaussian_backend")
            )
    except Exception:
        pass
    return DEFAULT_GAUSSIAN_BACKEND


def config_default_language() -> str:
    """Dialog default for the GUI language (config, then zh)."""
    import yaml

    try:
        loaded = yaml.safe_load(
            (Path(__file__).parent / "config" / "default.yaml").read_text()
        )
        if isinstance(loaded, dict) and isinstance(loaded.get("session"), dict):
            return i18n.normalize_language(loaded["session"].get("language"))
    except Exception:
        pass
    return i18n.DEFAULT_LANGUAGE


class SourceSelectDialog(QDialog):
    """Modal 源选择: language + camera source + backend + upscale.

    The language combo takes effect immediately (``i18n.set_language`` +
    ``_retranslate`` on this dialog); the accepted choice then applies to
    the whole GUI because ``MainWindow`` is only constructed afterwards.
    """

    def __init__(
        self,
        *,
        default_case: Path | None = None,
        default_source: str = SOURCE_FAKE_LIVE,
        default_backend: str = DEFAULT_SHAPE_PRIOR_BACKEND,
        default_upscale: bool = True,
        default_gaussian: str = DEFAULT_GAUSSIAN_BACKEND,
        default_language: str | None = None,
        default_record_dir: Path | None = None,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        i18n.set_language(
            default_language if default_language is not None else i18n.language()
        )
        self.setModal(True)
        # Language row first: switching it retranslates this dialog live.
        self._language_combo = QComboBox(self)
        for lang_id, native_name in _LANGUAGE_LABELS:
            self._language_combo.addItem(native_name, lang_id)
        lang_index = self._language_combo.findData(i18n.language())
        if lang_index >= 0:
            self._language_combo.setCurrentIndex(lang_index)
        self._language_combo.currentIndexChanged.connect(self._on_language_changed)
        self._language_label = QLabel(self)
        language_row = QHBoxLayout()
        language_row.addWidget(self._language_label)
        language_row.addWidget(self._language_combo, 1)
        # Hover ⓘ per option (what it is / what it is used for); tooltips
        # follow the live language switch via _retranslate.
        self._source_info = InfoDot(
            "真实相机:连接 RealSense 实时采集。fake-live 回放:用 data_collect "
            "里录好的素材从头重放整条流水线 —— 相机从不暂停,只是数据源是录像,"
            "适合无相机调试与复现实验。",
            "Real camera: live RealSense capture. fake-live replay: re-runs "
            "the whole pipeline from a pre-recorded data_collect case — the "
            "camera never pauses, only the source is a recording; useful for "
            "camera-free debugging and reproducible runs.",
            self,
        )
        self._case_info = InfoDot(
            "fake-live 模式读取的素材目录(data_collect/<case>):一次录制的"
            "彩色/深度帧与标定数据。",
            "The recording folder fake-live replays (data_collect/<case>): "
            "one capture session's color/depth frames plus calibration.",
            self,
        )
        self._backend_info = InfoDot(
            "决定 warmup 阶段如何生成物体的 shape prior(先验网格):<br>"
            "• TRELLIS.2 — microsoft/TRELLIS.2-4B,几何/纹理更精细,对齐更贴合"
            "(默认);<br>"
            "• SAM3D — v6.2 原生成器;<br>"
            "• 无 — 跳过生成:没有 mesh 和补点,ASAP/PhysTwin 下游关闭,"
            "仅用观测点云跟踪。<br>启动后不可更改(回到开始可重选)。",
            "How the warmup builds the object's shape prior (mesh):<br>"
            "• TRELLIS.2 — microsoft/TRELLIS.2-4B, finer geometry/texture, "
            "tighter alignment (default);<br>"
            "• SAM3D — the original v6.2 generator;<br>"
            "• None — skip generation: no mesh, no point filling, ASAP/"
            "PhysTwin downstream disabled, tracking uses observed points "
            "only.<br>Fixed once started (Back-to-start to change).",
            self,
        )
        self._upscale_info = InfoDot(
            "开启:生成前先把 frame-0 的物体裁剪用 Stable Diffusion ×4 超分,"
            "给生成器更清晰的条件图(v6.2 原链路)。关闭:直接用原分辨率裁剪 "
            "—— warmup 少一步超分推理,更快;生成器的输入图更小。",
            "On: before generation the frame-0 object crop is upscaled ×4 "
            "with Stable Diffusion for a sharper conditioning image (the "
            "v6.2 chain). Off: the original-resolution crop is used directly "
            "— one less inference step, faster warmup, smaller conditioning "
            "input.",
            self,
        )
        self._gaussian_info = InfoDot(
            "是否生成物体的 3D gaussians(高斯泼溅):TripoSplat 从 frame-0 "
            "生成并对齐到场景,正式期可在「高斯」频道实时渲染;选「无」跳过。"
            "与 mesh 链并行运行;shape prior 为「无」时强制关闭(它依赖链路的"
            "掩码图与世界对齐)。",
            "Whether to build 3D gaussians (splats) of the object: TripoSplat "
            "generates them from frame-0 and aligns them to the scene; the "
            "formal phase can render them live in the Gaussian channel. "
            "Runs in parallel with the mesh chain; forced off when the shape "
            "prior is None (it needs the chain's masked image and world "
            "alignment).",
            self,
        )
        self._info_dots = (
            self._source_info,
            self._case_info,
            self._backend_info,
            self._upscale_info,
            self._gaussian_info,
        )
        self._real_radio = QRadioButton(self)
        self._fake_radio = QRadioButton(self)
        if default_source == SOURCE_FAKE_LIVE:
            self._fake_radio.setChecked(True)
        else:
            self._real_radio.setChecked(True)
        self._case_edit = QLineEdit(self)
        if default_case is not None:
            self._case_edit.setText(str(default_case))
        self._browse_btn = QPushButton(self)
        self._browse_btn.clicked.connect(self._browse_case)
        self._case_label = QLabel(self)
        case_row = QHBoxLayout()
        case_row.addWidget(self._case_label)
        case_row.addWidget(self._case_info)
        case_row.addWidget(self._case_edit, 1)
        case_row.addWidget(self._browse_btn)
        # Shape-prior 生成后端 (sam3d / trellis2 / none): decided here because
        # the service prewarms the chosen backend's worker at spawn — it
        # cannot change without a 回到开始 relaunch.
        self._backend_combo = QComboBox(self)
        for backend_id, _pair in _BACKEND_LABELS:
            self._backend_combo.addItem("", backend_id)
        index = self._backend_combo.findData(default_backend)
        if index >= 0:
            self._backend_combo.setCurrentIndex(index)
        self._backend_label = QLabel(self)
        backend_row = QHBoxLayout()
        backend_row.addWidget(self._backend_label)
        backend_row.addWidget(self._backend_info)
        backend_row.addWidget(self._backend_combo, 1)
        # 上采样 (SD x4) toggle: off swaps the stage for the crop-only
        # passthrough — faster warmup, generation conditions on the
        # original-resolution crop. Backend "none" ignores it (no chain).
        self._upscale_check = QCheckBox(self)
        self._upscale_check.setChecked(bool(default_upscale))
        # Gaussian 生成模型 (triposplat / none): one real model today, but a
        # first-class run option like the shape-prior backend. Runs on the
        # camera GPU in parallel with the mesh chain; shape prior "none"
        # forces it off (needs the chain's masked image + world alignment).
        self._gaussian_combo = QComboBox(self)
        for gaussian_id, _pair in _GAUSSIAN_LABELS:
            self._gaussian_combo.addItem("", gaussian_id)
        gaussian_index = self._gaussian_combo.findData(default_gaussian)
        if gaussian_index >= 0:
            self._gaussian_combo.setCurrentIndex(gaussian_index)
        self._gaussian_label = QLabel(self)
        gaussian_row = QHBoxLayout()
        gaussian_row.addWidget(self._gaussian_label)
        gaussian_row.addWidget(self._gaussian_info)
        gaussian_row.addWidget(self._gaussian_combo, 1)
        # 录制 (real camera only): tee the whole run into a data_collect-
        # format directory that later replays as a fake-live case.
        self._record_check = QCheckBox(self)
        self._record_check.setChecked(False)
        self._record_info = InfoDot(
            "把本次真实相机运行的每一帧 RGB-D(含相机内参/时间戳)录制成 "
            "data_collect 格式的目录 —— 之后可以在本对话框选「fake-live 回放」"
            "直接重放这次运行。目录须为空;录制在后台线程写盘,不影响流水线。",
            "Record every RGB-D frame of this real-camera run (with camera "
            "intrinsics and timestamps) into a data_collect-format directory "
            "— it can later be replayed via the fake-live option in this "
            "dialog. The directory must be empty; writing happens on a "
            "background thread and never stalls the pipeline.",
            self,
        )
        from datetime import datetime  # noqa: PLC0415 (dialog-open timestamp)

        self._record_edit = QLineEdit(self)
        if default_record_dir is not None:
            # CLI --record-dir seeds the interactive dialog (checked +
            # path); otherwise a fresh timestamped default, unchecked.
            self._record_edit.setText(str(default_record_dir))
            self._record_check.setChecked(True)
        else:
            self._record_edit.setText(
                f"data_collect/record_{datetime.now():%Y%m%d_%H%M%S}"
            )
        self._record_browse_btn = QPushButton(self)
        self._record_browse_btn.clicked.connect(self._browse_record_dir)
        record_row = QHBoxLayout()
        record_row.addWidget(self._record_check)
        record_row.addWidget(self._record_info)
        record_row.addWidget(self._record_edit, 1)
        record_row.addWidget(self._record_browse_btn)
        # Only meaningful for the real camera; fake-live already IS a case.
        self._real_radio.toggled.connect(self._update_record_enabled)
        buttons = QDialogButtonBox(self)
        self._start_btn = buttons.addButton(
            "", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self._quit_btn = buttons.addButton(
            "", QDialogButtonBox.ButtonRole.RejectRole
        )
        self._start_btn.setDefault(True)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(language_row)
        self._source_label = QLabel(self)
        source_row = QHBoxLayout()
        source_row.addWidget(self._source_label)
        source_row.addWidget(self._source_info)
        source_row.addStretch(1)
        layout.addLayout(source_row)
        layout.addWidget(self._real_radio)
        layout.addWidget(self._fake_radio)
        layout.addLayout(case_row)
        layout.addLayout(backend_row)
        upscale_row = QHBoxLayout()
        upscale_row.addWidget(self._upscale_check)
        upscale_row.addWidget(self._upscale_info)
        upscale_row.addStretch(1)
        layout.addLayout(upscale_row)
        layout.addLayout(gaussian_row)
        layout.addLayout(record_row)
        layout.addWidget(buttons)
        self._error = QLabel("", self)
        self._error.setStyleSheet("color: #f28b82;")
        layout.addWidget(self._error)
        self._update_record_enabled()
        self._retranslate()
        self.resize(560, 300)

    def _update_record_enabled(self) -> None:
        real = self._real_radio.isChecked()
        for widget in (
            self._record_check,
            self._record_edit,
            self._record_browse_btn,
        ):
            widget.setEnabled(real)

    def _browse_record_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self,
            tr("选择录制保存目录", "Select the recording destination"),
        )
        if chosen:
            self._record_edit.setText(chosen)

    def _on_language_changed(self) -> None:
        i18n.set_language(str(self._language_combo.currentData()))
        self._error.setText("")
        self._retranslate()

    def _retranslate(self) -> None:
        """Apply the current language to every string in this dialog."""
        self.setWindowTitle(tr("demo_v7 — 源选择", "demo_v7 — Source Select"))
        self._language_label.setText(tr("界面语言:", "Language:"))
        self._source_label.setText(
            tr("请选择相机来源:", "Select the camera source:")
        )
        self._real_radio.setText(
            tr("真实相机(RealSense)", "Real camera (RealSense)")
        )
        self._fake_radio.setText(tr("fake-live 回放", "fake-live replay"))
        self._case_label.setText(tr("回放素材:", "Replay case:"))
        self._case_edit.setPlaceholderText(
            tr("data_collect/<case> 目录", "data_collect/<case> directory")
        )
        self._browse_btn.setText(tr("浏览…", "Browse…"))
        self._backend_label.setText(
            tr("Shape prior 生成:", "Shape prior generator:")
        )
        for i, (_backend_id, pair) in enumerate(_BACKEND_LABELS):
            self._backend_combo.setItemText(i, tr(*pair))
        self._upscale_check.setText(
            tr(
                "上采样(SD ×4 超分;关闭可加速 warmup)",
                "Upscale (SD ×4; disable for a faster warmup)",
            )
        )
        self._gaussian_label.setText(
            tr("Gaussian 生成:", "Gaussian generator:")
        )
        for i, (_gaussian_id, pair) in enumerate(_GAUSSIAN_LABELS):
            self._gaussian_combo.setItemText(i, tr(*pair))
        self._record_check.setText(
            tr("录制(可作 fake-live 素材)", "Record (reusable as fake-live)")
        )
        self._record_edit.setPlaceholderText(
            tr("录制保存目录(须为空)", "Recording directory (must be empty)")
        )
        self._record_browse_btn.setText(tr("浏览…", "Browse…"))
        self._record_info.retranslate()
        self._start_btn.setText(tr("开始", "Start"))
        self._quit_btn.setText(tr("退出", "Quit"))
        for dot in self._info_dots:
            dot.retranslate()

    def _browse_case(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self,
            tr("选择 data_collect case 目录", "Select a data_collect case directory"),
        )
        if chosen:
            self._case_edit.setText(chosen)
            self._fake_radio.setChecked(True)

    def _validate_and_accept(self) -> None:
        if self._fake_radio.isChecked() and not self._case_edit.text().strip():
            self._error.setText(
                tr(
                    "fake-live 需要选择一个素材目录。",
                    "fake-live needs a replay case directory.",
                )
            )
            return
        if self._real_radio.isChecked() and self._record_check.isChecked():
            record_text = self._record_edit.text().strip()
            if not record_text:
                self._error.setText(
                    tr("录制需要一个保存目录。", "Recording needs a directory.")
                )
                return
            # Same resolution the session applies (relative -> repo root),
            # so this pre-check and the session's check see the same path.
            target = Path(record_text).expanduser()
            if not target.is_absolute():
                target = Path(_BOOTSTRAP_REPO_ROOT_STR) / target
            try:
                dirty = target.exists() and (
                    not target.is_dir() or any(target.iterdir())
                )
            except OSError:
                dirty = True
            if dirty:
                self._error.setText(
                    tr(
                        "录制目录已存在且非空(或不是目录),请换一个。",
                        "Recording path exists and is not an empty "
                        "directory; pick another.",
                    )
                )
                return
        self.accept()

    def selection(
        self,
    ) -> tuple[str, Path | None, str, bool, str, str, Path | None]:
        """(source, case, backend, upscale, gaussian, language, record_dir)."""
        backend = str(self._backend_combo.currentData())
        upscale = bool(self._upscale_check.isChecked())
        gaussian = str(self._gaussian_combo.currentData())
        lang = str(self._language_combo.currentData())
        if self._fake_radio.isChecked():
            return (
                SOURCE_FAKE_LIVE,
                Path(self._case_edit.text().strip()),
                backend,
                upscale,
                gaussian,
                lang,
                None,
            )
        record_dir = (
            Path(self._record_edit.text().strip())
            if self._record_check.isChecked() and self._record_edit.text().strip()
            else None
        )
        return SOURCE_REAL, None, backend, upscale, gaussian, lang, record_dir


def create_session(
    source: str,
    fake_live_case: Path | None,
    base_path: Path | None,
    shape_prior_backend: str | None = None,
    shape_prior_upscale: bool | str | None = None,
    gaussian_backend: str | None = None,
    record_dir: Path | None = None,
) -> Any:
    """Build an OrchestratorSession (lazy import; see module docstring).

    The GUI vocabulary is real/fake-live; the session/v6.2 side names the real
    camera source ``live`` (demo_v6_2 --input-source choices), so normalize
    here — the GUI is the only place the word "real" exists.
    """
    from demo_v7.orchestration.session import OrchestratorSession

    kwargs: dict[str, Any] = {
        "source": "live" if source == SOURCE_REAL else source,
    }
    if fake_live_case is not None:
        kwargs["fake_live_case"] = fake_live_case
    if base_path is not None:
        kwargs["base_path"] = base_path
    if shape_prior_backend is not None:
        kwargs["shape_prior_backend"] = shape_prior_backend
    if shape_prior_upscale is not None:
        kwargs["shape_prior_upscale"] = shape_prior_upscale
    if gaussian_backend is not None:
        kwargs["gaussian_backend"] = gaussian_backend
    if record_dir is not None:
        kwargs["record_dir"] = record_dir
    return OrchestratorSession(**kwargs)


class AppController(QObject):
    """Owns the (dialog -> session -> window) lifecycle incl. 回到开始."""

    def __init__(self, app: QApplication, args: argparse.Namespace) -> None:
        super().__init__()
        self._app = app
        self._args = args
        self._window: MainWindow | None = None
        self._session: Any = None
        self._shutdown_thread: threading.Thread | None = None
        self._record_dir_seeded = False
        # One-time language init (CLI wins over config); afterwards the
        # dialog owns the choice and 回到开始 keeps the last selection.
        i18n.set_language(args.language or config_default_language())

    def start(self) -> bool:
        """First launch; returns False when the user cancelled the dialog."""
        if self._args.source is not None:
            return self._launch(
                self._args.source,
                self._args.fake_live_case,
                self._args.shape_prior_backend,
                self._args.shape_prior_upscale,
                self._args.gaussian_backend,
                self._args.record_dir,
            )
        return self._ask_and_launch()

    def _ask_and_launch(self) -> bool:
        cfg_source, cfg_case = config_default_source_and_case()
        dialog = SourceSelectDialog(
            default_case=self._args.fake_live_case or cfg_case,
            default_source=cfg_source,
            default_backend=(
                self._args.shape_prior_backend
                or config_default_shape_prior_backend()
            ),
            default_upscale=(
                normalize_upscale(self._args.shape_prior_upscale)
                if self._args.shape_prior_upscale is not None
                else config_default_shape_prior_upscale()
            ),
            default_gaussian=(
                self._args.gaussian_backend or config_default_gaussian_backend()
            ),
            # CLI --record-dir seeds only the first dialog: after 回到开始 the
            # dir is typically non-empty (run 1 recorded into it), so later
            # dialogs fall back to a fresh timestamped default.
            default_record_dir=(
                self._args.record_dir if not self._record_dir_seeded else None
            ),
        )
        self._record_dir_seeded = True
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        source, case, backend, upscale, gaussian, _language, record_dir = (
            dialog.selection()
        )
        # Language already applied globally by the dialog's live switch.
        return self._launch(source, case, backend, upscale, gaussian, record_dir)

    def _launch(
        self,
        source: str,
        fake_live_case: Path | None,
        shape_prior_backend: str | None = None,
        shape_prior_upscale: bool | str | None = None,
        gaussian_backend: str | None = None,
        record_dir: Path | None = None,
    ) -> bool:
        try:
            self._session = create_session(
                source,
                fake_live_case,
                self._args.base_path,
                shape_prior_backend,
                shape_prior_upscale,
                gaussian_backend,
                record_dir,
            )
            # start() spawns the camera service and connects both sockets; it
            # blocks up to connect_timeout_s and self-cleans on failure.
            self._session.start()
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox

            self._session = None
            QMessageBox.critical(
                None,
                tr("启动失败", "Startup failed"),
                tr("无法启动相机服务:", "Could not start the camera service: ")
                + str(exc),
            )
            return False
        self._window = MainWindow(self._session)
        self._window.restartRequested.connect(
            self._on_restart, Qt.ConnectionType.QueuedConnection
        )
        self._window.show()
        return True

    def _on_restart(self) -> None:
        """回到开始: tear down off the main thread, then re-ask the source."""
        self._await_shutdown(self._begin_teardown(), self._resume_restart)

    def _resume_restart(self) -> None:
        if not self._ask_and_launch():
            self._app.quit()

    def _begin_teardown(self) -> threading.Thread | None:
        """Detach + hide the window; start the session shutdown off-thread.

        The window disappears immediately; the blocking ``session.shutdown()``
        runs on a daemon thread (or is already running via the window's own
        ``closeEvent``). Returns the newest shutdown thread — also remembered
        so a later final ``teardown`` can bounded-join it — or None when
        nothing was ever launched. Idempotent with the service already dead.
        """
        thread: threading.Thread | None = None
        if self._window is not None:
            self._window.detach_session()
            self._window.setEnabled(False)
            self._window.hide()
            self._window.close()
            # If the user closed the window directly, closeEvent already
            # started the shutdown thread; reuse it instead of racing a
            # second (idempotent, immediately-returning) shutdown() call.
            thread = self._window.session_shutdown_thread()
            self._window.deleteLater()
            self._window = None
        session = self._session
        self._session = None
        if session is not None and thread is None:
            thread = shutdown_session_on_thread(session)
        if thread is not None:
            self._shutdown_thread = thread
        return self._shutdown_thread

    def _await_shutdown(
        self, thread: threading.Thread | None, on_done: Callable[[], None]
    ) -> None:
        """Invoke ``on_done`` once ``thread`` finishes, polling on a QTimer.

        Bounded: after _SHUTDOWN_WAIT_S the continuation runs anyway
        (force-continue; the daemon thread keeps draining in the background).
        """
        if thread is None or not thread.is_alive():
            on_done()
            return
        deadline = time.monotonic() + _SHUTDOWN_WAIT_S
        timer = QTimer(self)
        timer.setInterval(_SHUTDOWN_POLL_MS)

        def _poll() -> None:
            if thread.is_alive() and time.monotonic() < deadline:
                return
            timer.stop()
            timer.deleteLater()
            on_done()

        timer.timeout.connect(_poll)
        timer.start()

    def teardown(self) -> None:
        """Final teardown (no event loop left): bounded-join the shutdown."""
        thread = self._begin_teardown()
        if thread is not None:
            thread.join(_SHUTDOWN_WAIT_S)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("demo_v7")
    app.setStyleSheet(_APP_STYLESHEET)
    controller = AppController(app, args)
    if not controller.start():
        return 0
    try:
        status = app.exec()
    finally:
        controller.teardown()
    # Skip interpreter teardown: Open3D's OffscreenRenderer (mesh/补点 views)
    # aborts in its Filament destructors at exit ("Trying to destroy
    # nonexistent resource" -> terminate), which would turn every clean close
    # into an exit-code-134 error dialog from the launcher. Session/service
    # cleanup is already done (teardown above); os._exit just bypasses the
    # native destructor lottery — the standard Qt-app workaround.
    import os  # noqa: PLC0415

    os._exit(int(status))


if __name__ == "__main__":
    raise SystemExit(main())
