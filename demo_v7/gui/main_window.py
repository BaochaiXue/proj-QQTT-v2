"""demo_v7 main window: screen stack + persistent camera dock + event routing.

Threading contract: ``OrchestratorSession`` invokes its event/frame handlers on
its own socket-reader threads. This window registers plain ``Signal.emit``
bridges as those handlers, so every payload hops to the Qt main thread via a
queued connection before any widget is touched — no widget method here is ever
called off the main thread.

Session contract (owned by ``demo_v7/orchestration/session.py``, duck-typed
here so the GUI can run against a stub in headless smoke tests):

- ``send_command(cmd: dict) -> None`` (fire-and-forget; acks arrive as events)
- ``set_on_event(cb: Callable[[dict], None]) -> None``
- ``set_on_frame(cb: Callable[[FrameHeader, bytes], None]) -> None``
- ``shutdown() -> None`` (idempotent teardown)

Command sending is guarded: the control socket may not be connected yet during
service startup, so CMD_HELLO retries on a timer until the first event arrives
and every send failure surfaces in the status bar instead of raising.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Protocol

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QDockWidget,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from demo_v7.ipc import protocol
from demo_v7.ipc.protocol import FrameHeader
from demo_v7.gui.screens import (
    CaptureScreen,
    FinishedScreen,
    FormalScreen,
    RepositionScreen,
    ReviewScreen,
    WarmupScreen,
)
from demo_v7.gui.widgets import ImageView

_HELLO_RETRY_MS = 1000
_HELLO_MAX_TRIES = 120


class SessionLike(Protocol):
    """The slice of OrchestratorSession the GUI depends on."""

    def send_command(self, cmd: dict) -> None: ...

    def set_on_event(self, cb: Callable[[dict], None]) -> None: ...

    def set_on_frame(self, cb: Callable[[FrameHeader, bytes], None]) -> None: ...

    def shutdown(self) -> None: ...


def shutdown_session_on_thread(session: SessionLike) -> threading.Thread:
    """Run the blocking session teardown on a daemon thread.

    Sends CMD_SHUTDOWN (best-effort) then calls ``session.shutdown()`` — both
    idempotent and safe against a service that is already dead — so the Qt
    main thread never blocks on service teardown. Callers bounded-join the
    returned thread and force-continue if it is stuck.
    """

    def _worker() -> None:
        try:
            session.send_command({"cmd": protocol.CMD_SHUTDOWN})
        except Exception:
            pass
        try:
            session.shutdown()
        except Exception:
            pass

    thread = threading.Thread(
        target=_worker, name="v7-session-shutdown", daemon=True
    )
    thread.start()
    return thread


class MainWindow(QMainWindow):
    """QStackedWidget of the six screens + always-on RGB/depth dock."""

    restartRequested = Signal()

    # IPC -> main-thread bridges (auto connection == queued across threads).
    _eventArrived = Signal(dict)
    _frameArrived = Signal(object, object)  # (FrameHeader, jpeg bytes)

    def __init__(
        self, session: SessionLike, *, run_dir: str | None = None
    ) -> None:
        super().__init__()
        self._session = session
        self._run_dir = run_dir
        self._session_detached = False
        self._shutdown_thread: threading.Thread | None = None
        self._state = protocol.STATE_STARTING
        self._saw_any_event = False
        self._event_log_path: Path | None = None
        self.setWindowTitle("demo_v7 — 实时物理孪生")
        self.resize(1440, 900)

        self._capture = CaptureScreen()
        self._warmup = WarmupScreen()
        self._review = ReviewScreen()
        self._reposition = RepositionScreen()
        self._formal = FormalScreen()
        self._finished = FinishedScreen()
        self._stack = QStackedWidget(self)
        for screen in (
            self._capture,
            self._warmup,
            self._review,
            self._reposition,
            self._formal,
            self._finished,
        ):
            self._stack.addWidget(screen)
        self.setCentralWidget(self._stack)
        self._state_to_screen: dict[str, QWidget] = {
            protocol.STATE_STARTING: self._capture,
            protocol.STATE_PREVIEW: self._capture,
            protocol.STATE_FRAME0_PENDING: self._capture,
            protocol.STATE_WARMUP: self._warmup,
            protocol.STATE_REVIEW: self._review,
            protocol.STATE_REPOSITION: self._reposition,
            protocol.STATE_FORMAL: self._formal,
            protocol.STATE_FINISHED: self._finished,
            protocol.STATE_FATAL: self._finished,
        }

        self._build_camera_dock()
        self._wire_screens()

        self._eventArrived.connect(self._on_event)
        self._frameArrived.connect(self._on_frame)
        session.set_on_event(self._eventArrived.emit)
        session.set_on_frame(
            lambda header, payload: self._frameArrived.emit(header, payload)
        )

        # HELLO until the control channel answers (service may still be
        # binding its sockets); any received event stops the retries.
        self._hello_tries = 0
        self._hello_timer = QTimer(self)
        self._hello_timer.setInterval(_HELLO_RETRY_MS)
        self._hello_timer.timeout.connect(self._try_hello)
        self._hello_timer.start()
        self._try_hello()
        self.statusBar().showMessage("正在连接相机服务…")

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _build_camera_dock(self) -> None:
        dock = QDockWidget("实时相机", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        content = QWidget(dock)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        self._dock_rgb = ImageView("RGB", hint_size=(320, 180))
        self._dock_depth = ImageView("深度", hint_size=(320, 180))
        layout.addWidget(QLabel("RGB", content))
        layout.addWidget(self._dock_rgb, 1)
        layout.addWidget(QLabel("深度", content))
        layout.addWidget(self._dock_depth, 1)
        content.setMaximumWidth(360)
        dock.setWidget(content)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _wire_screens(self) -> None:
        self._capture.captureRequested.connect(
            lambda: self._send({"cmd": protocol.CMD_CAPTURE_FRAME0})
        )
        self._capture.retakeRequested.connect(
            lambda: self._send({"cmd": protocol.CMD_RETAKE_FRAME0})
        )
        self._capture.confirmRequested.connect(
            lambda: self._send({"cmd": protocol.CMD_CONFIRM_FRAME0})
        )
        self._warmup.viewResultsRequested.connect(
            lambda: self._send({"cmd": protocol.CMD_ENTER_REVIEW})
        )
        self._review.repositionRequested.connect(
            lambda: self._send({"cmd": protocol.CMD_BEGIN_REPOSITION})
        )
        self._review.regenGaussianRequested.connect(
            lambda: self._send({"cmd": protocol.CMD_REGEN_GAUSSIAN})
        )
        self._reposition.startFormalRequested.connect(
            lambda: self._send({"cmd": protocol.CMD_START_FORMAL})
        )
        self._formal.stopRequested.connect(
            lambda: self._send({"cmd": protocol.CMD_STOP_FORMAL})
        )
        self._finished.restartRequested.connect(self.restartRequested.emit)

    # ------------------------------------------------------------------
    # Command path
    # ------------------------------------------------------------------
    def _send(self, cmd: dict) -> None:
        try:
            self._session.send_command(cmd)
        except Exception as exc:  # pragma: no cover - depends on live session
            self.statusBar().showMessage(f"命令发送失败 ({cmd.get('cmd')}): {exc}")

    def _try_hello(self) -> None:
        if self._saw_any_event or self._hello_tries >= _HELLO_MAX_TRIES:
            self._hello_timer.stop()
            return
        self._hello_tries += 1
        try:
            self._session.send_command({"cmd": protocol.CMD_HELLO})
        except Exception:
            # Control socket not up yet; keep retrying on the timer.
            pass

    # ------------------------------------------------------------------
    # Event / frame routing (main thread; hopped via signals)
    # ------------------------------------------------------------------
    def _on_frame(self, header: FrameHeader, payload: bytes) -> None:
        channel = header.channel
        if channel == protocol.CH_RGB:
            self._dock_rgb.setFrame(payload)
        elif channel == protocol.CH_DEPTH:
            self._dock_depth.setFrame(payload)
        current = self._stack.currentWidget()
        on_frame = getattr(current, "on_frame", None)
        if on_frame is not None:
            on_frame(channel, payload)

    def _log_event(self, event: dict) -> None:
        """Archive one service event: stdout + <run_dir>/v7_gui_events.log.

        Owner rule 2026-08-06: the GUI shows progress visually; the textual
        event stream lives on the command line and in a per-run log file.
        High-frequency formal stats are skipped (they'd swamp the file and
        already reach the status bar). Best-effort: logging never breaks the
        event path.
        """
        name = str(event.get("event", "?"))
        if name == protocol.EVT_FORMAL_STATS:
            return
        parts = [f"[{name}]"]
        for key in ("cmd", "state", "stage", "detail", "kind", "where",
                    "message", "error"):
            value = event.get(key)
            if value not in (None, ""):
                parts.append(f"{key}={value}")
        if event.get("ok") is False:
            parts.append("ok=False")
        if event.get("elapsed_ms") is not None:
            parts.append(f"elapsed_ms={event['elapsed_ms']:.0f}")
        if name == protocol.EVT_ARTIFACTS:
            parts.append(f"paths={sorted(dict(event.get('paths') or {}))}")
        line = (
            time.strftime("%H:%M:%S") + " " + " ".join(str(p) for p in parts)
        )
        print(f"[v7-gui] {line}", flush=True)
        try:
            if self._event_log_path is None:
                run_dir = self._resolve_run_dir()
                if run_dir is None:
                    return
                self._event_log_path = Path(run_dir) / "v7_gui_events.log"
            with self._event_log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:
            pass

    def _on_event(self, event: dict) -> None:
        self._saw_any_event = True
        self._log_event(event)
        name = event.get("event")
        if name == protocol.EVT_ACK:
            self._on_ack(event)
        elif name == protocol.EVT_STATE:
            self._apply_state(str(event.get("state", "")), event.get("detail"))
        elif name == protocol.EVT_PROGRESS:
            stage = str(event.get("stage", ""))
            detail = str(event.get("detail", ""))
            ok = bool(event.get("ok", True))
            if stage == "gaussian":
                # TripoSplat generation runs while the operator sits on the
                # REVIEW screen; its status lives in the Gaussian tab.
                self._review.set_gaussian_progress(detail, ok)
            else:
                self._warmup.on_progress(stage, detail, ok, event.get("elapsed_ms"))
        elif name == protocol.EVT_ARTIFACTS:
            self._on_artifacts(event)
        elif name == protocol.EVT_ERROR:
            message = f"错误 [{event.get('where', '?')}]: {event.get('message', '')}"
            self.statusBar().showMessage(message)
        elif name == protocol.EVT_REPLAY_EXHAUSTED:
            if event.get("wrapped", False):
                # Pre-formal wrap: the stream keeps running (a camera never
                # stops) — informational only, no modal in the operator flow.
                self.statusBar().showMessage(
                    "fake-live 素材已播放完毕,已从头继续播放。", 10000
                )
            else:
                QMessageBox.information(
                    self, "回放结束", "fake-live 素材已播放完毕,正式追踪已停止。"
                )
        elif name == protocol.EVT_FORMAL_STATS:
            self._formal.set_stats(event)

    def _on_ack(self, event: dict) -> None:
        cmd = event.get("cmd")
        if not event.get("ok", False):
            self.statusBar().showMessage(
                f"命令被拒绝 ({cmd}): {event.get('error', '未知错误')}"
            )
            return
        if cmd == protocol.CMD_HELLO:
            state = event.get("state")
            if isinstance(state, str) and state:
                self._apply_state(state, None)
            source_kind = event.get("source_kind")
            backend = event.get("shape_prior_backend")
            # Truthful echo of the 上采样 toggle (service-resolved; absent in
            # older acks -> treated as the on default and not annotated).
            no_upscale = event.get("shape_prior_upscale") is False
            if source_kind:
                suffix = f" | prior:{backend}" if backend else ""
                if suffix and no_upscale and backend != "none":
                    suffix += "(无超分)"
                self.setWindowTitle(
                    f"demo_v7 — 实时物理孪生 [{source_kind}{suffix}]"
                )
            if isinstance(backend, str) and backend:
                # Review screen adapts its Shape Prior/补点 tabs (backend
                # "none" renders the observed points alone, no candidates);
                # warmup screen renames its generate row.
                self._review.set_shape_prior_backend(backend)
                self._warmup.set_shape_prior_backend(backend)

    def _on_artifacts(self, event: dict) -> None:
        kind = event.get("kind")
        paths_raw = event.get("paths")
        if not isinstance(paths_raw, dict):
            return
        paths = {str(k): str(v) for k, v in paths_raw.items()}
        if kind == protocol.ARTIFACT_KIND_FRAME0:
            self._capture.show_candidate(paths)
            # The observed-object-points npz for the 补点 review view also
            # ships under the frame0 kind.
            self._review.set_frame0_artifacts(paths)
        elif kind == protocol.ARTIFACT_KIND_MASKS:
            self._review.set_mask_artifacts(paths)
        elif kind == protocol.ARTIFACT_KIND_SHAPE_PRIOR:
            self._review.set_shape_prior_artifacts(paths)
        elif kind == protocol.ARTIFACT_KIND_ALIGNMENT:
            self._review.set_alignment_artifacts(paths)
        elif kind == protocol.ARTIFACT_KIND_GAUSSIAN:
            self._review.set_gaussian_artifacts(paths)

    def _apply_state(self, state: str, detail: Any) -> None:
        if state not in self._state_to_screen:
            return
        previous = self._state
        self._state = state
        self._capture.set_pending(state == protocol.STATE_FRAME0_PENDING)
        if state == protocol.STATE_WARMUP and previous != protocol.STATE_WARMUP:
            self._warmup.reset()
        if state == protocol.STATE_FINISHED:
            self._finished.set_title("本次运行已结束。")
            self._finished.set_run_dir(self._resolve_run_dir())
        if state == protocol.STATE_FATAL:
            self._finished.set_title("运行失败(fatal)。")
            self._finished.set_run_dir(self._resolve_run_dir())
            QMessageBox.critical(
                self,
                "致命错误",
                f"相机服务遇到不可恢复的错误:{detail or '详见日志'}",
            )
        self._stack.setCurrentWidget(self._state_to_screen[state])
        label = str(detail) if detail else ""
        self.statusBar().showMessage(f"状态: {state} {label}".rstrip())

    def _resolve_run_dir(self) -> str | None:
        if self._run_dir:
            return str(self._run_dir)
        for attr in ("run_dir", "run_output_dir", "base_path"):
            value = getattr(self._session, attr, None)
            if value:
                return str(value)
        return None

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------
    def detach_session(self) -> None:
        """Stop using the session (app controller owns its shutdown)."""
        self._session_detached = True
        self._hello_timer.stop()
        try:
            self._session.set_on_event(lambda event: None)
            self._session.set_on_frame(lambda header, payload: None)
        except Exception:
            pass

    def session_shutdown_thread(self) -> threading.Thread | None:
        """The daemon shutdown thread started by ``closeEvent``, if any."""
        return self._shutdown_thread

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        if not self._session_detached:
            # Blocking session.shutdown() must never run on the Qt main
            # thread: detach handlers, disable the window (it hides on close
            # anyway) and tear the service down on a daemon thread. The app
            # controller bounded-joins this thread on final teardown.
            session = self._session
            self.detach_session()
            self.setEnabled(False)
            self._shutdown_thread = shutdown_session_on_thread(session)
        super().closeEvent(event)
