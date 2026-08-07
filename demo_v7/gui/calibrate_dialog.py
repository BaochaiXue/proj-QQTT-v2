"""可视化桌面标定对话框 (in-GUI ChArUco table calibration).

The one-shot CLI (``cameras_calibrate_table.py``) is blind: it grabs a single
frame headlessly and either succeeds or dies. This dialog shows the SAME
strict estimation live: the camera view with the tool's own diagnostic
overlay, corner counts and reprojection error per frame, and a capture
button that is enabled only while the current frame passes the strict gate.
Saving writes exactly the file set the CLI writes, through the same qqtt
writer functions.

The camera runs in a child process (``calibrate_stream_worker.py``): an
in-process reopen of a stopped RealSense reliably never delivers a first
frame, process death is the only full device release, and the repo's GUIs
never open cameras in-process anyway. This dialog is only the QProcess
client: JSON lines in, preview JPEG (atomic-replaced) reloaded per frame.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from demo_v7.gui.i18n import tr

_WORKER_SCRIPT = Path(__file__).resolve().parent / "calibrate_stream_worker.py"

# Workers that ignored QUIT (stuck in a wedged camera call). Their QProcess
# objects must outlive the dialog: destroying a QProcess SIGKILLs only the
# main child, which would strand CameraSystem's camera-holding child
# processes — the worker's own process-group fallback (<=15s) reaps the
# whole group, we just keep the handle alive until then.
_LINGERING_WORKERS: list[QProcess] = []


def _linger(process: QProcess) -> None:
    process.setParent(None)
    _LINGERING_WORKERS.append(process)
    process.finished.connect(lambda *_: _LINGERING_WORKERS.remove(process))


class CalibrationDialog(QDialog):
    """Live visual table calibration; accept() means a calibration was saved."""

    def __init__(
        self,
        *,
        serial: str | None,
        output_path: Path,
        diagnostic_path: Path | None,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self._output_path = output_path
        self.saved = False
        self._latest_ok = False
        self._stdout_buffer = b""
        self.setModal(True)
        self.setWindowTitle(tr("桌面标定", "Table calibration"))

        self._view = QLabel(self)
        self._view.setMinimumSize(848, 477)
        self._view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._view.setStyleSheet("background: #202124;")
        self._view.setText(tr("正在打开相机…", "Opening the camera…"))

        self._status = QLabel(self)
        self._status.setWordWrap(True)
        self._hint = QLabel(
            tr(
                "把 ChArUco 标定板平放在桌面上,等指标变绿后点「拍摄并保存」。",
                "Place the ChArUco board flat on the table; capture once the "
                "metrics turn green.",
            ),
            self,
        )
        self._hint.setWordWrap(True)

        self._capture_btn = QPushButton(
            tr("拍摄并保存", "Capture && save"), self
        )
        self._capture_btn.setEnabled(False)
        self._capture_btn.clicked.connect(self._capture)
        self._cancel_btn = QPushButton(tr("取消", "Cancel"), self)
        self._cancel_btn.clicked.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self._capture_btn)
        buttons.addWidget(self._cancel_btn)
        layout = QVBoxLayout(self)
        layout.addWidget(self._view, 1)
        layout.addWidget(self._status)
        layout.addWidget(self._hint)
        layout.addLayout(buttons)

        self._preview_dir = tempfile.TemporaryDirectory(prefix="v7_calibrate_")
        argv = [
            "-u",
            str(_WORKER_SCRIPT),
            "--output", str(output_path),
            "--preview-dir", self._preview_dir.name,
        ]
        if diagnostic_path is not None:
            argv.extend(["--diagnostic", str(diagnostic_path)])
        if serial:
            argv.extend(["--serial", str(serial)])
        self._process: QProcess | None = QProcess(self)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.finished.connect(self._on_process_finished)
        self._process.start(sys.executable, argv)

    # ------------------------------------------------------------------
    def _on_stdout(self) -> None:
        if self._process is None:
            return
        self._stdout_buffer += bytes(self._process.readAllStandardOutput())
        *lines, self._stdout_buffer = self._stdout_buffer.split(b"\n")
        for raw in lines:
            raw = raw.strip()
            if not raw.startswith(b"{"):
                continue  # camera library chatter on merged pipes
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            self._on_event(event)

    def _on_event(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "frame":
            pixmap = QPixmap(str(event.get("preview_path", "")))
            if not pixmap.isNull():
                self._view.setPixmap(
                    pixmap.scaled(
                        self._view.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            self._latest_ok = bool(event.get("ok"))
            if self._latest_ok:
                self._status.setStyleSheet("color: #81c995;")
                self._status.setText(
                    tr("检测到标定板 · ", "Board detected · ")
                    + f"{event.get('corner_count', 0)} "
                    + tr("角点", "corners")
                    + f" ({event.get('corner_fraction', 0.0):.0%}) · "
                    + tr("重投影 ", "reprojection ")
                    + f"{event.get('reprojection_error_px', 0.0):.3f}px"
                )
            else:
                self._status.setStyleSheet("color: #f28b82;")
                self._status.setText(
                    tr("未通过: ", "Not ready: ")
                    + str(event.get("message", ""))[-200:]
                )
            self._capture_btn.setEnabled(self._latest_ok)
        elif kind == "saved":
            self.saved = True
            self.accept()
        elif kind == "fatal":
            self._show_camera_error(str(event.get("message", "")))

    def _show_camera_error(self, message: str) -> None:
        self._status.setStyleSheet("color: #f28b82;")
        self._status.setText(tr("相机错误: ", "Camera error: ") + message[-300:])
        self._view.setText(tr("相机不可用", "Camera unavailable"))
        self._capture_btn.setEnabled(False)

    def _on_process_finished(self, exit_code: int, _status: Any) -> None:
        if self.saved:
            return
        if exit_code != 0 and not self._status.text():
            self._show_camera_error(f"worker exit {exit_code}")

    def _capture(self) -> None:
        if self._process is None or not self._latest_ok:
            return
        # Freeze the UI on the capture decision; the worker saves its latest
        # accepted frame, emits "saved" and exits — accept() follows.
        self._capture_btn.setEnabled(False)
        self._cancel_btn.setEnabled(False)
        self._process.write(b"CAPTURE\n")

    def done(self, result: int) -> None:  # noqa: N802 — Qt override
        process, self._process = self._process, None
        if process is not None:
            process.write(b"QUIT\n")
            # Normal exit: camera stop after init is ~1s, init itself ~5s.
            if not process.waitForFinished(8000):
                _linger(process)  # wedged: the worker group-kills itself
        self._preview_dir.cleanup()
        super().done(result)
