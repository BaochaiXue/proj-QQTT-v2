"""Deterministic lifecycle races: barriers select the failing interleaving."""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from demo_v7.ipc import protocol
from demo_v7.ipc.channel import ControlClient
from demo_v7.orchestration.session import OrchestratorSession


def link_session():
    # A real session prepares (and wipes) its output dirs, so keep it out of
    # the repo's shared outputs/.
    session = OrchestratorSession(
        base_path=Path(tempfile.mkdtemp(prefix="v7-concurrency-")) / "run",
        shape_prior_backend="none",
        gaussian_backend="none",
    )
    session._service = SimpleNamespace(poll=lambda: None)
    session._connect_poll_interval_s = 0.001
    return session


class Client(ControlClient):
    def __init__(self, *, hello_fails=False):
        self.hello_fails = hello_fails
        self.closed = threading.Event()
        self.hello = threading.Event()

    def close(self):
        self.closed.set()

    def send_command(self, command):
        self.hello.set()
        if self.hello_fails:
            raise ConnectionError("peer died before HELLO")


@pytest.mark.parametrize("link", ["control", "frames"])
def test_death_during_dial_is_not_lost(monkeypatch, link):
    session = link_session()
    dialed = threading.Event()
    release = threading.Event()
    redialed = threading.Event()
    clients = [Client(), Client()]
    calls = []

    def dial():
        calls.append(1)
        if len(calls) == 1:
            dialed.set()
            assert release.wait(2)
            return clients[0]
        redialed.set()
        return clients[1]

    monkeypatch.setattr(session, f"_dial_{link}", dial)
    try:
        session._schedule_reconnect(link)
        assert dialed.wait(2)
        session._schedule_reconnect(link)  # watched reader dies during dial
        release.set()
        assert redialed.wait(1), "the second reader-death notification was lost"
        assert clients[0].closed.wait(1)
    finally:
        session._shutdown_done = True
        release.set()


def test_failed_hello_closes_client_and_redials(monkeypatch):
    session = link_session()
    clients = [Client(hello_fails=True), Client()]
    redialed = threading.Event()
    calls = []

    def dial():
        calls.append(1)
        if len(calls) == 1:
            return clients[0]
        redialed.set()
        return clients[1]

    monkeypatch.setattr(session, "_dial_control", dial)
    try:
        session._schedule_reconnect("control")
        assert redialed.wait(1), "failed HELLO ended reconnect instead of redialing"
        assert clients[0].closed.wait(1)
        assert clients[1].hello.wait(1)
        assert session._control is not clients[0]
    finally:
        session._shutdown_done = True


def artifact_runtime():
    from demo_v7.service.staged_runtime import StagedRuntime

    runtime = object.__new__(StagedRuntime)
    runtime._artifacts_sent = {}
    runtime._artifacts_lock = threading.Lock()
    runtime.control = SimpleNamespace(send_event=lambda event: None)
    return runtime


def test_simultaneous_artifact_merges_keep_both_fields():
    runtime = artifact_runtime()
    copying = threading.Event()
    release = threading.Event()
    second_done = threading.Event()

    class Paths(dict):
        def items(self):
            copying.set()
            assert release.wait(2)
            return super().items()

    first = threading.Thread(
        target=runtime._emit_artifacts,
        args=("frame0", Paths(rgb="rgb.png")),
    )

    def emit_second():
        runtime._emit_artifacts("frame0", {"depth": "depth.npy"})
        second_done.set()

    second = threading.Thread(target=emit_second)
    try:
        first.start()
        assert copying.wait(2)
        second.start()
        second_done.wait(0.1)
    finally:
        release.set()
        first.join(2)
        second.join(2)
    assert runtime._artifacts_sent["frame0"] == {
        "rgb": "rgb.png",
        "depth": "depth.npy",
    }


# --- accepted commands must own real work ---------------------------------


def mesh_surface_manager(tmp_path):
    from demo_v7.service.mesh_surface_manager import MeshSurfaceGaussianManager

    manager = MeshSurfaceGaussianManager(
        case_dir=tmp_path / "case",
        out_dir=tmp_path / "gaussian",
        emit_progress=lambda *a, **k: None,
        emit_artifacts=lambda *a, **k: None,
        emit_error=lambda *a, **k: None,
    )
    manager._case_ready.set()
    return manager


def test_reservation_admits_exactly_one_regen(tmp_path):
    manager = mesh_surface_manager(tmp_path)
    assert manager.try_reserve() is True
    assert manager.try_reserve() is False, "two regens both claimed the slot"


def regen_runtime(manager):
    from demo_v7.service.staged_runtime import StagedRuntime

    runtime = object.__new__(StagedRuntime)
    runtime._state = protocol.STATE_REVIEW
    runtime._state_lock = threading.Lock()
    runtime._gaussian_manager = manager
    return runtime


def test_second_regen_in_one_tick_is_acked_false(tmp_path):
    """Both commands land before the main loop runs either deferred submit."""
    from demo_v7.service.staged_runtime import StagedRuntime

    manager = mesh_surface_manager(tmp_path)
    runtime = regen_runtime(manager)
    command = {"cmd": protocol.CMD_REGEN_GAUSSIAN, "seed": 7}
    first_ack, first_work = StagedRuntime._cmd_regen_gaussian(runtime, command)
    second_ack, second_work = StagedRuntime._cmd_regen_gaussian(runtime, command)
    assert first_ack["ok"] is True and first_work is not None
    assert second_ack["ok"] is False, "the losing regen was acked as accepted"
    assert second_work is None


def test_closed_manager_is_acked_false(tmp_path):
    from demo_v7.service.staged_runtime import StagedRuntime

    manager = mesh_surface_manager(tmp_path)
    manager.shutdown(timeout_s=0.1)
    runtime = regen_runtime(manager)
    ack, work = StagedRuntime._cmd_regen_gaussian(
        runtime, {"cmd": protocol.CMD_REGEN_GAUSSIAN}
    )
    assert ack["ok"] is False and work is None


# --- a terminal link failure must stop production -------------------------


def quiesce_session(monkeypatch, *, service_state):
    session = link_session()
    monkeypatch.setattr(
        type(session), "service_state", property(lambda _self: service_state)
    )
    sent: list[dict] = []
    stopped = threading.Event()
    session.send_command = sent.append
    session._stop_phystwin = stopped.set
    session._status = None
    return session, sent, stopped


def test_frames_link_failure_stops_formal_and_downstream(monkeypatch):
    session, sent, stopped = quiesce_session(
        monkeypatch, service_state=protocol.STATE_FORMAL
    )
    session._note_link_error("frames", "socket gone")
    assert session._capture_finished_event.is_set(), "chunk stream left waiting"
    assert stopped.is_set(), "PhysTwin kept running for a failed run"
    assert sent == [{"cmd": protocol.CMD_STOP_FORMAL}]


def test_control_link_failure_does_not_try_to_send(monkeypatch):
    session, sent, stopped = quiesce_session(
        monkeypatch, service_state=protocol.STATE_FORMAL
    )
    session._note_link_error("control", "socket gone")
    assert session._capture_finished_event.is_set()
    assert stopped.is_set()
    assert sent == [], "asked a dead control link to carry a command"


# --- FINISHED must imply the producers/writers actually stopped -----------


def finalize_runtime(worker_names, *, stuck):
    from demo_v7.runtime.mdp.plumbing import FatalErrorLatch
    from demo_v7.service.staged_runtime import StagedRuntime

    release = threading.Event()
    threads = []
    for name in worker_names:
        target = (lambda: release.wait(5)) if name in stuck else (lambda: None)
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        threads.append(thread)

    runtime = object.__new__(StagedRuntime)
    runtime.stop_event = threading.Event()
    runtime.fatal = FatalErrorLatch(
        status=SimpleNamespace(emit=lambda *a, **k: None),
        stop_event=runtime.stop_event,
    )
    runtime._acq_thread = None
    runtime._formal = SimpleNamespace(
        lossless=SimpleNamespace(close_queues=lambda: None),
        threads=threads,
        timeline_gate=SimpleNamespace(incomplete_run_error=lambda: None),
    )
    runtime.session = SimpleNamespace(
        release_camera=lambda: None,
        headless_capture_writer=None,
        depth_engine=None,
    )
    runtime.shape_prior_manager = SimpleNamespace(write_profile_json=lambda: None)
    runtime._announce_fatal_if_needed = lambda: None
    runtime._acquisition = SimpleNamespace(replay_exhausted=False)
    runtime.control = SimpleNamespace(send_event=lambda event: None)
    runtime._enter_state = lambda *a, **k: None
    return runtime, release


def test_stuck_critical_worker_blocks_a_clean_finish(monkeypatch):
    import demo_v7.service.staged_runtime as sr

    monkeypatch.setattr(sr, "_FINALIZE_DRAIN_S", 0.05)
    runtime, release = finalize_runtime(
        ["demo-v7-pair-output", "demo-v7-gaussian"], stuck={"demo-v7-pair-output"}
    )
    try:
        sr.StagedRuntime._finalize_formal(runtime)
        snapshot = runtime.fatal.snapshot()
        assert snapshot is not None, "FINISHED while a writer was still running"
        assert "demo-v7-pair-output" in snapshot.message
    finally:
        release.set()


def test_stuck_observer_worker_does_not_fail_the_run(monkeypatch):
    import demo_v7.service.staged_runtime as sr

    monkeypatch.setattr(sr, "_FINALIZE_DRAIN_S", 0.05)
    runtime, release = finalize_runtime(
        ["demo-v7-pair-output", "demo-v7-gaussian"], stuck={"demo-v7-gaussian"}
    )
    try:
        sr.StagedRuntime._finalize_formal(runtime)
        assert runtime.fatal.snapshot() is None, "display-only worker failed the run"
    finally:
        release.set()
