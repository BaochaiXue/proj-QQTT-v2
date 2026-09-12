"""Lifecycle/teardown regression tests (audit fixes, 2026-09-11).

Each test pins a behaviour that was previously wrong:
- FORMAL was announced (and the capture producer released) even when the
  readiness wait ended because a worker went fatal;
- ``shutdown()`` wrote STAGE_RUN_FINISHED ok=true for a wedged chunk thread,
  a service that had to be killed, or a service that reported FATAL;
- a chunk-stream failure was latched but never told the GUI, and a late
  FINISHED could still paint the failed run green;
- a dropped control link swallowed commands silently;
- the recorder reported a clean close after its drain timed out;
- artifacts emitted during a control-link gap were lost to a GUI that
  attached (or re-attached) afterwards.
"""

from __future__ import annotations

import subprocess
import threading
import time

import pytest

from demo_v7.ipc import protocol


class _StatusStub:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, str, bool]] = []

    def emit(self, stage, detail="", *, ok=True, **kwargs) -> None:
        self.emitted.append((str(stage), str(detail), bool(ok)))


class _ControlStub:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def send_event(self, event: dict) -> None:
        self.events.append(dict(event))


class _PreloadStub:
    def __init__(self, ready: bool) -> None:
        self._ready = ready
        self.calls = 0

    def wait_frame0_consumers_ready(self, timeout: float) -> bool:
        self.calls += 1
        return self._ready


def _bare_runtime(*, ready: bool):
    """A StagedRuntime carrying only what the readiness commit touches."""
    from demo_v7.runtime.mdp.plumbing import FatalErrorLatch
    from demo_v7.service.staged_runtime import StagedRuntime

    runtime = object.__new__(StagedRuntime)
    runtime.stop_event = threading.Event()
    runtime._formal_go = threading.Event()
    runtime.control = _ControlStub()
    runtime.preload = _PreloadStub(ready)
    runtime.fatal = FatalErrorLatch(
        status=_StatusStub(), stop_event=runtime.stop_event
    )
    return runtime


class TestFormalReadinessCommit:
    @pytest.mark.parametrize(
        "ready, stop, deadline_offset_s, expect_formal, expect_fatal",
        [
            # The control row: the only way out that MAY announce FORMAL.
            pytest.param(True, False, 5.0, True, False, id="ready"),
            pytest.param(False, True, 5.0, False, False, id="stop_event"),
            # Deadline in the past: fatal recorded, still no FORMAL.
            pytest.param(False, False, -1.0, False, True, id="deadline"),
        ],
    )
    def test_stop_event_during_wait_never_announces_formal(
        self, ready, stop, deadline_offset_s, expect_formal, expect_fatal
    ) -> None:
        """The readiness loop also exits through its CONDITION, not just
        its break: guards the re-check at staged_runtime.py:1512-1513
        (``if self.stop_event.is_set() or self.fatal.snapshot() is not
        None: return``). Without it, teardown — or the deadline fatal,
        which sets stop_event too — fell through to ``_formal_go.set()``
        + ``_announce_state(STATE_FORMAL)``: the producer was released
        and the GUI shown the formal screen for a dying run."""
        runtime = _bare_runtime(ready=ready)
        if stop:
            runtime.stop_event.set()
        runtime._commit_formal_after_readiness(
            deadline_s=time.perf_counter() + deadline_offset_s
        )
        assert runtime._formal_go.is_set() is expect_formal
        assert [e["state"] for e in runtime.control.events] == (
            [protocol.STATE_FORMAL] if expect_formal else []
        )
        assert (runtime.fatal.snapshot() is not None) is expect_fatal

    def test_fatal_during_wait_never_announces_formal(self) -> None:
        """The regression: a worker dying mid-readiness used to still
        release the producer and tell the GUI FORMAL. The fatal half of
        the staged_runtime.py:1512 re-check, raced from another thread the
        way a real worker records it."""
        runtime = _bare_runtime(ready=False)

        def _go_fatal() -> None:
            time.sleep(0.05)
            runtime.fatal.record("worker", RuntimeError("boom"))

        threading.Thread(target=_go_fatal, daemon=True).start()
        runtime._commit_formal_after_readiness(
            deadline_s=time.perf_counter() + 5.0
        )
        assert not runtime._formal_go.is_set()
        assert runtime.control.events == []


class _FakeProc:
    """An already-exited service handle (stop_process is a no-op on it).

    ``drain_timeout=True`` makes the CMD_SHUTDOWN drain wait blow its
    deadline, the way a service that never honoured the shutdown does.
    """

    def __init__(self, code=None, *, drain_timeout: bool = False) -> None:
        self._code = code
        self._drain_timeout = bool(drain_timeout)
        # A high, unused pid: stop_process probes the GROUP first, and an
        # invalid (<=1) id would raise EINVAL instead of "no such group".
        self.pid = 4194303
        self.returncode = code

    def poll(self):
        return self._code

    def wait(self, timeout=None):
        if self._drain_timeout:
            raise subprocess.TimeoutExpired(cmd="camera-service", timeout=timeout)
        return self._code


class _StuckThread:
    def is_alive(self) -> bool:
        return True

    def join(self, timeout=None) -> None:
        return None


def _session(tmp_path):
    from demo_v7.orchestration.session import OrchestratorSession

    session = OrchestratorSession(
        source="fake-live",
        fake_live_case="data_collect/fake",
        base_path=tmp_path / "run",
    )
    session._status = _StatusStub()
    return session


class TestShutdownHonesty:
    @pytest.mark.parametrize(
        "preload, expect_ok, detail_fragment",
        [
            # The control row: nothing wrong, so it really is FINISHED.
            pytest.param(lambda s: None, True, "", id="clean"),
            pytest.param(lambda s: setattr(s, "_chunk_thread", _StuckThread()),
                         False, "did not drain", id="wedged_chunk_thread"),
            pytest.param(lambda s: setattr(s, "_service_state", protocol.STATE_FATAL),
                         False, "reported fatal", id="service_state_fatal"),
            pytest.param(lambda s: setattr(s, "_service", _FakeProc(code=2)),
                         False, "exit code 2", id="service_exit_2"),
            # NEGATIVE code: the case the `!= 0` (not `> 0`) check exists
            # for — a SIGTERM'd/SIGKILL'd service reports -15/-9.
            pytest.param(lambda s: setattr(s, "_service", _FakeProc(code=-15)),
                         False, "exit code -15", id="service_signal_killed"),
            pytest.param(
                lambda s: setattr(s, "_service", _FakeProc(code=0, drain_timeout=True)),
                False, "did not exit on shutdown", id="service_drain_timed_out"),
        ],
    )
    def test_wedged_chunk_thread_is_not_a_clean_finish(
        self, tmp_path, preload, expect_ok, detail_fragment
    ) -> None:
        """One table over the shutdown()->reasons->status.emit path in
        demo_v7/orchestration/session.py: :1125-1129 (chunk thread alive
        past its join timeout), :1130-1133 (service ignored CMD_SHUTDOWN
        and had to be terminated), :1137 (any non-zero exit, negative
        included) and :1139-1142 (service reported FATAL). Previously
        only ``_chunk_error`` counted, so every failing row below still
        wrote STAGE_RUN_FINISHED ok=true."""
        from demo_v7.runtime.pipeline_status import (
            STAGE_FATAL,
            STAGE_RUN_FINISHED,
        )

        session = _session(tmp_path)
        preload(session)
        session.shutdown(chunk_join_timeout_s=0.01)
        stage, detail, ok = session._status.emitted[-1]
        assert ok is expect_ok
        assert stage == (STAGE_RUN_FINISHED if expect_ok else STAGE_FATAL)
        assert detail_fragment in detail


class TestChunkFailureReporting:
    def test_failure_reaches_gui_and_stops_producers(self, tmp_path, monkeypatch):
        """A chunk-stream failure used to be latched and nothing else:
        the GUI kept showing a live run and the producers kept feeding a
        consumer that no longer existed. Guards all three halves of the
        35fbf5f handler in demo_v7/orchestration/session.py: :901-907
        (the ``_handle_event`` EVT_ERROR emit), :910-914 (CMD_STOP_FORMAL)
        and :915 (``_stop_phystwin()``)."""
        import demo_v7.orchestration.session as session_mod
        from demo_v7.service import arap_rescue

        session = _session(tmp_path)
        session._service_state = protocol.STATE_FORMAL
        events = []
        commands = []
        stopped = []
        session.set_on_event(events.append)
        monkeypatch.setattr(session, "send_command", commands.append)
        monkeypatch.setattr(session, "_stop_phystwin", lambda: stopped.append(True))
        monkeypatch.setattr(arap_rescue, "patch_arap_factorize_rescue", lambda: None)
        monkeypatch.setattr(arap_rescue, "patch_asap_island_cleanup", lambda: None)
        failure = RuntimeError("chunk materialization failed")

        def fail_stream(*args, **kwargs):
            raise failure

        monkeypatch.setattr(session_mod, "ChunkStreamSession", fail_stream)
        session._run_chunk_stream()

        assert session.chunk_error is failure
        assert events == [
            {
                "event": protocol.EVT_ERROR,
                "where": "chunk_stream",
                "message": str(failure),
            }
        ]
        assert commands == [{"cmd": protocol.CMD_STOP_FORMAL}]
        assert stopped == [True]

    def test_gui_failure_survives_service_finish(self, monkeypatch):
        """Capture can finish draining AFTER the parent's chunk stream
        died; its FINISHED/HELLO events must not paint the failed run
        green again. Guards demo_v7/gui/main_window.py:326-327 (escalate
        a chunk/control/frames EVT_ERROR to STATE_FATAL) and :424-425
        (``if self._state == protocol.STATE_FATAL: return`` — FATAL is
        sticky). One ``where`` suffices: the chunk_stream/frames_link
        members of the literal tuple at :326 add no mutant."""
        monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication, QMessageBox

        from demo_v7.gui.main_window import MainWindow

        app = QApplication.instance() or QApplication([])
        dialogs = []
        monkeypatch.setattr(
            QMessageBox, "critical", lambda *args: dialogs.append(args[-1])
        )

        class Session:
            def send_command(self, command):
                pass

            def set_on_event(self, callback):
                pass

            def set_on_frame(self, callback):
                pass

        window = MainWindow(Session())
        try:
            window._on_event(
                {
                    "event": protocol.EVT_ERROR,
                    "where": "control_link",
                    "message": "chunk materialization failed",
                }
            )
            window._on_event(
                {"event": protocol.EVT_STATE, "state": protocol.STATE_FINISHED}
            )
            window._on_event(
                {
                    "event": protocol.EVT_ACK,
                    "cmd": protocol.CMD_HELLO,
                    "ok": True,
                    "state": protocol.STATE_FORMAL,
                }
            )
            app.processEvents()
            assert window._state == protocol.STATE_FATAL
            assert window._stack.currentWidget() is window._finished
            assert len(dialogs) == 1
            assert "chunk materialization failed" in dialogs[0]
        finally:
            window.detach_session()
            window.close()


def test_terminal_link_error_cannot_report_success(tmp_path):
    """A dead link is terminal even when the last state seen was
    FINISHED. Guards demo_v7/orchestration/session.py:695-696 (latch
    ``_terminal_failure`` in ``_note_link_error``), :809-810 (raise it
    from ``wait_for_state`` BEFORE the state check) and :1121-1122 (count
    it as a shutdown reason) — all 35fbf5f."""
    from demo_v7.runtime.pipeline_status import STAGE_FATAL

    session = _session(tmp_path)
    events = []
    session.set_on_event(events.append)
    session._service_state = protocol.STATE_FINISHED
    session._note_link_error("control", "reconnect timed out")
    assert events[-1]["where"] == "control_link"
    with pytest.raises(RuntimeError, match="reconnect timed out"):
        session.wait_for_state(protocol.STATE_FINISHED, timeout_s=0.1)
    session.shutdown(chunk_join_timeout_s=0.01)
    stage, detail, ok = session._status.emitted[-1]
    assert stage == STAGE_FATAL and not ok
    assert "reconnect timed out" in detail

    # The sibling latch on the same pre-check (session.py:805-808): a
    # FINISHED service must not let ``wait_for_state`` return success over
    # a chunk stream that died. Both checks precede
    # ``if self._service_state in targets: return``.
    chunk_session = _session(tmp_path)
    chunk_session._service_state = protocol.STATE_FINISHED
    chunk_session._chunk_error = RuntimeError("chunk materialization failed")
    with pytest.raises(RuntimeError, match="chunk materialization failed"):
        chunk_session.wait_for_state(protocol.STATE_FINISHED, timeout_s=0.1)


def test_link_exit_during_shutdown_is_expected(tmp_path):
    """demo_v7/orchestration/session.py:686-688 — the ``if
    self._shutdown_done: return`` guard at the top of
    ``_note_link_error``. Without it the service's perfectly normal exit
    during shutdown latched a terminal failure, so EVERY clean run ended
    STAGE_FATAL."""
    session = _session(tmp_path)
    events = []
    session.set_on_event(events.append)
    session._shutdown_done = True
    session._note_link_error("control", "service exited")
    assert events == []
    assert session._terminal_failure is None


class TestControlCommandDelivery:
    def test_send_on_dead_socket_raises(self, tmp_path) -> None:
        """A command that never left the socket must not look delivered:
        demo_v7/ipc/channel.py:227-230 raises ConnectionError where the
        pre-d84b608 code did ``except OSError: pass``, so a button pressed
        while the control link was reconnecting silently did nothing."""
        import socket

        from demo_v7.ipc.channel import ControlClient

        server_path = tmp_path / "ctl.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(server_path))
        server.listen(1)
        client = ControlClient(server_path, on_event=lambda _e: None)
        conn, _ = server.accept()
        try:
            conn.close()
            server.close()
            with pytest.raises(ConnectionError):
                for _ in range(5):  # first send may land in the send buffer
                    client.send_command({"cmd": protocol.CMD_HELLO})
                    time.sleep(0.01)
        finally:
            try:
                client.close()
            except Exception:
                pass


class TestRecorderCloseHonesty:
    def test_stuck_writer_reports_incomplete(self, tmp_path, monkeypatch) -> None:
        """demo_v7/service/recorder.py:189-206 — the ``if
        self._worker.is_alive():`` error latch. The join is bounded, so
        before this branch existed ``close()`` returned a clean summary
        for a writer that was still appending: a truncated fake-live case
        was reported as a good recording."""
        import demo_v7.service.recorder as recorder_mod
        from demo_v7.service.recorder import FakeLiveCaseRecorder

        monkeypatch.setattr(recorder_mod, "_CLOSE_DRAIN_TIMEOUT_S", 0.05)
        rec = FakeLiveCaseRecorder(tmp_path / "case")
        blocked = threading.Event()

        class _NeverFinishes:
            def is_alive(self) -> bool:
                return True

            def join(self, timeout=None) -> None:
                blocked.set()

        rec._worker = _NeverFinishes()
        summary = rec.close()
        assert blocked.is_set()
        assert summary["error"] is not None


class TestHelloArtifactSnapshot:
    """A GUI attaching late (or after a control-link gap) must still learn
    the artifacts already produced — events are dropped with no retention
    while no client is connected."""

    def _runtime(self):
        from demo_v7.ipc import protocol as proto
        from demo_v7.service.staged_runtime import StagedRuntime

        runtime = object.__new__(StagedRuntime)
        runtime._artifacts_sent = {}
        runtime._artifacts_lock = threading.Lock()
        runtime.control = _ControlStub()
        return runtime, proto

    def test_emitted_artifacts_accumulate_and_merge_per_kind(self) -> None:
        """The producer half of bf6033a: demo_v7/service/staged_runtime.py
        :374-377 remembers every emitted artifact, MERGING per kind
        instead of overwriting (a second frame0 emit must not lose the
        first one's entries), and :378-380 still sends it live."""
        runtime, proto = self._runtime()
        runtime._emit_artifacts(proto.ARTIFACT_KIND_FRAME0, {"candidate": "a.png"})
        runtime._emit_artifacts(proto.ARTIFACT_KIND_FRAME0, {"object_points": "b.npz"})
        runtime._emit_artifacts(proto.ARTIFACT_KIND_MASKS, {"object": "m.png"})
        assert runtime._artifacts_sent[proto.ARTIFACT_KIND_FRAME0] == {
            "candidate": "a.png",
            "object_points": "b.npz",
        }
        assert runtime._artifacts_sent[proto.ARTIFACT_KIND_MASKS] == {"object": "m.png"}
        # Still sent live, unchanged.
        assert len(runtime.control.events) == 3

    def test_gui_replays_snapshot_from_hello_ack(self) -> None:
        """The consumer half of bf6033a: demo_v7/gui/main_window.py
        :395-399 replays the hello-ack artifact snapshot, including the
        ``and paths`` filter that skips kinds with no paths (an empty kind
        replayed as an artifact event resets the screen that owns it)."""
        pytest.importorskip("PySide6")
        from demo_v7.gui.main_window import MainWindow
        from demo_v7.ipc import protocol as proto

        replayed: list[dict] = []

        class _Screen:
            def __getattr__(self, _name):
                return lambda *a, **k: None

        class _Window:
            """Duck-typed stand-in: MainWindow is a Qt class and cannot be
            built with object.__new__, but _on_ack only needs these."""

            def __init__(self) -> None:
                self._on_artifacts = replayed.append
                self._apply_state = lambda *a, **k: None
                self.setWindowTitle = lambda *a, **k: None
                self.statusBar = lambda: _Screen()
                self._review = _Screen()
                self._warmup = _Screen()
                self._hello_synced = False

        window = _Window()
        MainWindow._on_ack(
            window,
            {
                "event": proto.EVT_ACK,
                "cmd": proto.CMD_HELLO,
                "ok": True,
                "artifacts": {
                    proto.ARTIFACT_KIND_MASKS: {"object": "m.png"},
                    proto.ARTIFACT_KIND_SHAPE_PRIOR: {},
                },
            },
        )
        assert replayed == [{"kind": proto.ARTIFACT_KIND_MASKS,
                             "paths": {"object": "m.png"}}]


class TestReviewArtifactReplay:
    """The hello ack replays the whole artifact snapshot on every control
    re-dial, so the Review screen's shared prior grid must be idempotent —
    ``_ArtifactGrid.add_images`` only appends, and shape_prior + alignment
    feed the SAME grid so neither setter may clear it alone."""

    def _screen(self):
        pytest.importorskip("PySide6")
        from demo_v7.gui.screens import ReviewScreen

        rendered: list[dict] = []

        class _Grid:
            def clear(self):
                rendered.clear()

            def add_images(self, paths):
                rendered.append(dict(paths))

        screen = ReviewScreen.__new__(ReviewScreen)
        screen._prior_grid = _Grid()
        screen._prior_paths = {}
        return screen, rendered

    def test_replayed_snapshot_does_not_duplicate_or_lose_stills(self) -> None:
        from demo_v7.gui.screens import ReviewScreen

        screen, rendered = self._screen()
        ReviewScreen._refresh_prior_grid(screen, {"mesh_glb": "m.png"})
        ReviewScreen._refresh_prior_grid(screen, {"match": "a.png"})
        # A reconnect replays both kinds again.
        ReviewScreen._refresh_prior_grid(screen, {"mesh_glb": "m.png"})
        ReviewScreen._refresh_prior_grid(screen, {"match": "a.png"})
        assert rendered[-1] == {"mesh_glb": "m.png", "match": "a.png"}, (
            "the shared grid lost one kind's stills or duplicated them"
        )
