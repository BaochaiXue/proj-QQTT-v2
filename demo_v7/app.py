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

from demo_v7.gui.main_window import MainWindow, shutdown_session_on_thread
from demo_v7.service.backend_options import (  # import-light (stdlib only)
    BACKEND_NONE,
    BACKEND_SAM3D,
    BACKEND_TRELLIS2,
    DEFAULT_SHAPE_PRIOR_BACKEND,
    SHAPE_PRIOR_BACKENDS,
    normalize_backend,
)

SOURCE_REAL = "real"
SOURCE_FAKE_LIVE = "fake-live"

# Combo order + GUI labels for the shape-prior generation backend.
_BACKEND_LABELS: tuple[tuple[str, str], ...] = (
    (BACKEND_SAM3D, "SAM3D(默认)"),
    (BACKEND_TRELLIS2, "TRELLIS.2"),
    (BACKEND_NONE, "无(不生成 shape prior)"),
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


class SourceSelectDialog(QDialog):
    """Modal 源选择: 真实相机 vs fake-live + case folder."""

    def __init__(
        self,
        *,
        default_case: Path | None = None,
        default_source: str = SOURCE_FAKE_LIVE,
        default_backend: str = DEFAULT_SHAPE_PRIOR_BACKEND,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("demo_v7 — 源选择")
        self.setModal(True)
        self._real_radio = QRadioButton("真实相机(RealSense)", self)
        self._fake_radio = QRadioButton("fake-live 回放", self)
        if default_source == SOURCE_FAKE_LIVE:
            self._fake_radio.setChecked(True)
        else:
            self._real_radio.setChecked(True)
        self._case_edit = QLineEdit(self)
        self._case_edit.setPlaceholderText("data_collect/<case> 目录")
        if default_case is not None:
            self._case_edit.setText(str(default_case))
        browse_btn = QPushButton("浏览…", self)
        browse_btn.clicked.connect(self._browse_case)
        case_row = QHBoxLayout()
        case_row.addWidget(QLabel("回放素材:", self))
        case_row.addWidget(self._case_edit, 1)
        case_row.addWidget(browse_btn)
        # Shape-prior 生成后端 (sam3d / trellis2 / none): decided here because
        # the service prewarms the chosen backend's worker at spawn — it
        # cannot change without a 回到开始 relaunch.
        self._backend_combo = QComboBox(self)
        for backend_id, label in _BACKEND_LABELS:
            self._backend_combo.addItem(label, backend_id)
        index = self._backend_combo.findData(default_backend)
        if index >= 0:
            self._backend_combo.setCurrentIndex(index)
        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Shape prior 生成:", self))
        backend_row.addWidget(self._backend_combo, 1)
        buttons = QDialogButtonBox(self)
        start_btn = buttons.addButton("开始", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton("退出", QDialogButtonBox.ButtonRole.RejectRole)
        start_btn.setDefault(True)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("请选择相机来源:", self))
        layout.addWidget(self._real_radio)
        layout.addWidget(self._fake_radio)
        layout.addLayout(case_row)
        layout.addLayout(backend_row)
        layout.addWidget(buttons)
        self._error = QLabel("", self)
        self._error.setStyleSheet("color: #f28b82;")
        layout.addWidget(self._error)
        self.resize(560, 250)

    def _browse_case(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "选择 data_collect case 目录")
        if chosen:
            self._case_edit.setText(chosen)
            self._fake_radio.setChecked(True)

    def _validate_and_accept(self) -> None:
        if self._fake_radio.isChecked() and not self._case_edit.text().strip():
            self._error.setText("fake-live 需要选择一个素材目录。")
            return
        self.accept()

    def selection(self) -> tuple[str, Path | None, str]:
        """Return (source, fake_live_case, backend) after ``exec`` accepted."""
        backend = str(self._backend_combo.currentData())
        if self._fake_radio.isChecked():
            return SOURCE_FAKE_LIVE, Path(self._case_edit.text().strip()), backend
        return SOURCE_REAL, None, backend


def create_session(
    source: str,
    fake_live_case: Path | None,
    base_path: Path | None,
    shape_prior_backend: str | None = None,
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

    def start(self) -> bool:
        """First launch; returns False when the user cancelled the dialog."""
        if self._args.source is not None:
            return self._launch(
                self._args.source,
                self._args.fake_live_case,
                self._args.shape_prior_backend,
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
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        source, case, backend = dialog.selection()
        return self._launch(source, case, backend)

    def _launch(
        self,
        source: str,
        fake_live_case: Path | None,
        shape_prior_backend: str | None = None,
    ) -> bool:
        try:
            self._session = create_session(
                source,
                fake_live_case,
                self._args.base_path,
                shape_prior_backend,
            )
            # start() spawns the camera service and connects both sockets; it
            # blocks up to connect_timeout_s and self-cleans on failure.
            self._session.start()
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox

            self._session = None
            QMessageBox.critical(None, "启动失败", f"无法启动相机服务:{exc}")
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
