"""Lifecycle/teardown regression tests (audit fixes, 2026-09-11).

Each test pins a behaviour that was previously wrong:
- FORMAL was announced (and the capture producer released) even when the
  readiness wait ended because a worker went fatal;
- ``shutdown()`` wrote STAGE_RUN_FINISHED ok=true for a wedged chunk thread,
  a service that had to be killed, or a service that reported FATAL;
- a dropped control link swallowed commands silently;
- the recorder reported a clean close after its drain timed out;
- nothing killed the run's child process groups when the GUI gave up
  waiting for a graceful shutdown.
"""

from __future__ import annotations

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
    def test_ready_commits_formal(self) -> None:
        runtime = _bare_runtime(ready=True)
        runtime._commit_formal_after_readiness(
            deadline_s=time.perf_counter() + 5.0
        )
        assert runtime._formal_go.is_set()
        assert [e["state"] for e in runtime.control.events] == [
            protocol.STATE_FORMAL
        ]

    def test_fatal_during_wait_never_announces_formal(self) -> None:
        """The regression: a worker dying mid-readiness used to still
        release the producer and tell the GUI FORMAL."""
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

    def test_stop_event_during_wait_never_announces_formal(self) -> None:
        runtime = _bare_runtime(ready=False)
        runtime.stop_event.set()
        runtime._commit_formal_after_readiness(
            deadline_s=time.perf_counter() + 5.0
        )
        assert not runtime._formal_go.is_set()
        assert runtime.control.events == []

    def test_readiness_deadline_goes_fatal_without_formal(self) -> None:
        runtime = _bare_runtime(ready=False)
        runtime._commit_formal_after_readiness(
            deadline_s=time.perf_counter() - 1.0
        )
        assert not runtime._formal_go.is_set()
        assert runtime.control.events == []
        assert runtime.fatal.snapshot() is not None


class _FakeProc:
    """An already-exited service handle (stop_process is a no-op on it)."""

    def __init__(self, code=None) -> None:
        self._code = code
        # A high, unused pid: stop_process probes the GROUP first, and an
        # invalid (<=1) id would raise EINVAL instead of "no such group".
        self.pid = 4194303
        self.returncode = code

    def poll(self):
        return self._code

    def wait(self, timeout=None):
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
    def test_clean_shutdown_reports_finished(self, tmp_path) -> None:
        from demo_v7.runtime.pipeline_status import STAGE_RUN_FINISHED

        session = _session(tmp_path)
        session.shutdown(chunk_join_timeout_s=0.01)
        stage, _detail, ok = session._status.emitted[-1]
        assert stage == STAGE_RUN_FINISHED and ok

    def test_wedged_chunk_thread_is_not_a_clean_finish(self, tmp_path) -> None:
        from demo_v7.runtime.pipeline_status import STAGE_FATAL

        session = _session(tmp_path)
        session._chunk_thread = _StuckThread()
        session.shutdown(chunk_join_timeout_s=0.01)
        stage, detail, ok = session._status.emitted[-1]
        assert stage == STAGE_FATAL and not ok
        assert "did not drain" in detail

    def test_service_fatal_state_is_not_a_clean_finish(self, tmp_path) -> None:
        from demo_v7.runtime.pipeline_status import STAGE_FATAL

        session = _session(tmp_path)
        session._service_state = protocol.STATE_FATAL
        session.shutdown(chunk_join_timeout_s=0.01)
        stage, _detail, ok = session._status.emitted[-1]
        assert stage == STAGE_FATAL and not ok

    def test_service_nonzero_exit_is_not_a_clean_finish(self, tmp_path) -> None:
        from demo_v7.runtime.pipeline_status import STAGE_FATAL

        session = _session(tmp_path)
        session._service = _FakeProc(code=2)
        session.shutdown(chunk_join_timeout_s=0.01)
        stage, detail, ok = session._status.emitted[-1]
        assert stage == STAGE_FATAL and not ok
        assert "exit code 2" in detail

    def test_force_terminate_is_idempotent_without_children(self, tmp_path) -> None:
        session = _session(tmp_path)
        assert session.force_terminate() == []
        assert session.force_terminate() == []


class TestControlCommandDelivery:
    def test_send_on_dead_socket_raises(self, tmp_path) -> None:
        """A command that never left the socket must not look delivered."""
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
                for _ in range(50):  # first send may land in the send buffer
                    client.send_command({"cmd": protocol.CMD_HELLO})
                    time.sleep(0.01)
        finally:
            try:
                client.close()
            except Exception:
                pass


class TestRecorderCloseHonesty:
    def test_stuck_writer_reports_incomplete(self, tmp_path, monkeypatch) -> None:
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
        assert "still running" in summary["error"]
