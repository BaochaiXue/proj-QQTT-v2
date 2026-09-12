"""Deterministic lifecycle races: barriers select the failing interleaving."""

from __future__ import annotations

import tempfile
import threading
import time
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


def test_death_during_dial_is_not_lost(monkeypatch):
    """A reader death that lands mid-dial must not be swallowed.

    Guards the reconnect-generation handshake: session.py:598 (the
    _link_generation bump in _schedule_reconnect), :662 (install only while
    `generation == self._link_generation[which]`) and :671 (releasing
    _reconnecting INSIDE the install lock).  _schedule_reconnect used to
    early-return on `which in self._reconnecting` while the worker released
    ownership only after installing, so a death announced during the dial was
    dropped and the link never came back.  The dial is link-agnostic; the only
    control-specific branch is the HELLO send, covered by the next test.
    """
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

    monkeypatch.setattr(session, "_dial_control", dial)
    try:
        session._schedule_reconnect("control")
        assert dialed.wait(2)
        session._schedule_reconnect("control")  # watched reader dies during dial
        release.set()
        assert redialed.wait(1), "the second reader-death notification was lost"
        assert clients[0].closed.wait(1)
    finally:
        session._shutdown_done = True
        release.set()


def test_failed_hello_closes_client_and_redials(monkeypatch):
    """HELLO is sent BEFORE install; a failure closes the client and redials.

    Guards session.py:650-656.  The re-hello used to run after install wrapped
    in `except Exception: pass`, so a peer that died between connect and send
    left a corpse in self._control with the reconnect flag cleared and no
    further on_dead possible.
    """
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
    """The read-copy-write of _artifacts_sent must hold _artifacts_lock.

    Guards staged_runtime.py:374.  Several workers really do call
    _emit_artifacts concurrently (mesh_surface_manager.py:205,
    gaussian_manager.py:341/473, staged_runtime.py:872/1342/1350), so an
    unlocked merge clobbered the other thread's field and the Review tab
    stayed one artifact short.
    """
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


def regen_runtime(manager):
    from demo_v7.service.staged_runtime import StagedRuntime

    runtime = object.__new__(StagedRuntime)
    runtime._state = protocol.STATE_REVIEW
    runtime._state_lock = threading.Lock()
    runtime._gaussian_manager = manager
    return runtime


@pytest.mark.parametrize("refusal", ["second-in-tick", "closed"])
def test_second_regen_in_one_tick_is_acked_false(tmp_path, refusal):
    """A regen the manager will not run must be acked ok=False, with no work.

    Guards staged_runtime.py:744 (`if not manager.try_reserve():`) against both
    terms of mesh_surface_manager.py:143-145 — `self._busy = True` claimed
    inside the lock, and the `self._closed` check.

    'second-in-tick': the control thread acks before the main loop runs the
    deferred submit, so two regens in one tick both saw `manager.busy == False`,
    both got ok=true, and the loser's later regenerate()->False was discarded —
    the GUI spun forever on a regen that never ran.  The first ack also pins the
    reservation as one-shot: only one caller may claim the slot.
    'closed': a shut-down manager reports `busy == False` too, so only the
    `_closed` term can turn this one away.
    """
    from demo_v7.service.staged_runtime import StagedRuntime

    manager = mesh_surface_manager(tmp_path)
    runtime = regen_runtime(manager)
    command = {"cmd": protocol.CMD_REGEN_GAUSSIAN, "seed": 7}
    if refusal == "second-in-tick":
        first_ack, first_work = StagedRuntime._cmd_regen_gaussian(runtime, command)
        assert first_ack["ok"] is True and first_work is not None
    else:
        manager.shutdown(timeout_s=0.1)
    ack, work = StagedRuntime._cmd_regen_gaussian(runtime, command)
    assert ack["ok"] is False, "the losing regen was acked as accepted"
    assert work is None


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


@pytest.mark.parametrize(
    "link, expected_sent",
    [
        ("frames", [{"cmd": protocol.CMD_STOP_FORMAL}]),
        ("control", []),
    ],
)
def test_frames_link_failure_stops_formal_and_downstream(
    monkeypatch, link, expected_sent
):
    """A dead link must quiesce the run, not merely report itself.

    Guards the quiesce block of _note_link_error, session.py:712-719:
    _capture_finished_event.set(), the conditional CMD_STOP_FORMAL, and
    _stop_phystwin().  Reporting the failure was not enough — the chunk stream
    waited out its shape-prior timeout for rows that could never arrive and
    PhysTwin held the GPU until the operator quit.  The 'control' row pins the
    `which != "control"` guard at session.py:713: when the control link itself
    is what died there is nothing to ask, and nothing may be handed to it.
    """
    session, sent, stopped = quiesce_session(
        monkeypatch, service_state=protocol.STATE_FORMAL
    )
    session._note_link_error(link, "socket gone")
    assert session._capture_finished_event.is_set(), "chunk stream left waiting"
    assert stopped.is_set(), "PhysTwin kept running for a failed run"
    assert sent == expected_sent, "asked a dead control link to carry a command"


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


@pytest.mark.parametrize(
    "stuck_worker, expect_fatal",
    [
        ("demo-v7-pair-output", True),
        ("demo-v7-gaussian", False),
    ],
)
def test_stuck_critical_worker_blocks_a_clean_finish(
    monkeypatch, stuck_worker, expect_fatal
):
    """FINISHED must not outrun the producers and writers.

    Guards staged_runtime.py:1687-1695 — the `critical = [n for n in
    still_running if n not in _OBSERVER_WORKERS]` filter and the
    `self.fatal.record(...)` that follows it.  The old finalize joined each
    thread for 1.0s and then simply carried on, so the run reported FINISHED
    while a producer/writer was still running.  The second row pins the
    _OBSERVER_WORKERS exclusion (staged_runtime.py:90): a display-only worker
    that overstays is noisy, not fatal, and must not fail a good run.
    """
    import demo_v7.service.staged_runtime as sr

    monkeypatch.setattr(sr, "_FINALIZE_DRAIN_S", 0.05)
    runtime, release = finalize_runtime(
        ["demo-v7-pair-output", "demo-v7-gaussian"], stuck={stuck_worker}
    )
    try:
        sr.StagedRuntime._finalize_formal(runtime)
        snapshot = runtime.fatal.snapshot()
        if expect_fatal:
            assert snapshot is not None, "FINISHED while a writer was still running"
            assert stuck_worker in snapshot.message
        else:
            assert snapshot is None, "display-only worker failed the run"
    finally:
        release.set()


# --- the gaussian manager must really free the camera GPU before FORMAL ---


def triposplat_manager(tmp_path):
    from demo_v7.service.gaussian_manager import GaussianManager

    events: list[tuple] = []
    return GaussianManager(
        case_dir=tmp_path / "case",
        out_dir=tmp_path / "gaussian",
        controller_name="hand",
        emit_progress=lambda *a, **k: events.append(("prog", a)),
        emit_artifacts=lambda *a, **k: events.append(("art", a)),
        emit_error=lambda *a, **k: events.append(("err", a)),
    ), events


def test_silent_worker_eof_settles_busy_loudly(tmp_path):
    """A hard worker death (OOM kill / native abort) skips the worker's own
    error path, so the reader loop must clear _busy and say so — otherwise
    every later regen is acked "already in flight" forever."""
    import io

    manager, events = triposplat_manager(tmp_path)
    manager._busy = True
    manager._proc = SimpleNamespace(
        stdout=io.StringIO(""), poll=lambda: -9, stdin=None
    )
    manager._reader_loop()
    assert manager.busy is False, "busy stayed set after the worker died"
    assert any(kind == "err" for kind, _ in events), "the death was silent"
