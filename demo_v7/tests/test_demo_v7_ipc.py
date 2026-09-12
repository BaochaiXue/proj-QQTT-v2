"""Unit tests for demo_v7.ipc.channel — real UDS in a tmpdir, pure CPU.

Covers the contract points: command/event round trips over control.sock,
ack-then-event ordering, re-accept after a client disconnect, latest-wins
frame delivery (with a slow reader and with nobody attached), non-blocking
publish/send with no client attached, and socket-file lifecycle (close
unlinks, a stale file is replaced at bind). No Qt, no GPU, no demo_v6_2
imports.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import pytest

from demo_v7.ipc import protocol
from demo_v7.ipc.channel import (
    ControlClient,
    ControlServer,
    FrameStreamClient,
    FrameStreamServer,
)

_WAIT_S = 10.0


def _wait_until(predicate, *, timeout_s: float = _WAIT_S) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached within timeout")


class _EventSink:
    """Thread-safe collector for control events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.events: list[dict] = []

    def __call__(self, event: dict) -> None:
        with self._lock:
            self.events.append(event)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.events)


class _FrameSink:
    """Thread-safe collector for frames, with optional per-frame delay."""

    def __init__(self, *, delay_s: float = 0.0) -> None:
        self._lock = threading.Lock()
        self._delay_s = delay_s
        self.frames: list[tuple[protocol.FrameHeader, bytes]] = []

    def __call__(self, header: protocol.FrameHeader, payload: bytes) -> None:
        if self._delay_s:
            time.sleep(self._delay_s)
        with self._lock:
            self.frames.append((header, payload))

    def snapshot(self) -> list[tuple[protocol.FrameHeader, bytes]]:
        with self._lock:
            return list(self.frames)


@pytest.fixture()
def sock_dir(tmp_path: Path) -> Path:
    return tmp_path


def _echo_ack(cmd: dict) -> dict:
    return {"event": protocol.EVT_ACK, "cmd": cmd.get("cmd"), "ok": True}


@pytest.mark.parametrize(
    "cmd, state, with_ack",
    [
        # Handler returns an ack: it must reach the GUI first, then the
        # state event sent after it (shared outbox => wire order).
        pytest.param(
            protocol.CMD_HELLO, protocol.STATE_PREVIEW, True, id="handler-acks"
        ),
        # Handler returns None: nothing at all may be enqueued for it.
        pytest.param(
            protocol.CMD_SHUTDOWN, protocol.STATE_FINISHED, False, id="handler-skips-ack"
        ),
    ],
)
def test_control_on_command_may_skip_ack(
    sock_dir: Path, cmd: str, state: str, with_ack: bool
) -> None:
    """Guards demo_v7/ipc/channel.py:154 `if ack is not None:` in
    ControlServer._serve_connection. Drop that guard (put_nowait(ack)
    unconditionally — the naive rewrite) and the handler-skips-ack row
    fails: staged_runtime.py:531 `_on_command` deliberately returns None
    whenever a command has a follow-up, and a null ack on the wire would
    reach the GUI as a bogus event. The handler-acks row pins the other
    half of the same contract: the ack is enqueued, and it precedes the
    state event that follows it.
    """
    path = sock_dir / protocol.CONTROL_SOCKET_NAME
    seen: queue.Queue = queue.Queue()

    def on_command(obj: dict) -> dict | None:
        seen.put(obj)
        return _echo_ack(obj) if with_ack else None

    server = ControlServer(path, on_command=on_command)
    sink = _EventSink()
    client = ControlClient(path, on_event=sink)
    try:
        client.send_command({"cmd": cmd})
        # Wait for the handler before send_event, so "ack first" is the
        # server's ordering and not a race in the test.
        assert seen.get(timeout=_WAIT_S) == {"cmd": cmd}
        server.send_event({"event": protocol.EVT_STATE, "state": state})
        expected: list[dict] = []
        if with_ack:
            expected.append({"event": protocol.EVT_ACK, "cmd": cmd, "ok": True})
        expected.append({"event": protocol.EVT_STATE, "state": state})
        _wait_until(lambda: len(sink.snapshot()) >= len(expected))
        assert sink.snapshot() == expected
    finally:
        client.close()
        server.close()


def test_control_reaccepts_after_client_disconnect(sock_dir: Path) -> None:
    """Guards demo_v7/ipc/channel.py:121 — the `while not self._closed.is_set():`
    accept loop that re-enters accept() once _serve_connection returns. Serve
    exactly one client (a `return` after _serve_connection) and this test alone
    fails: a GUI restart would then find a dead control.sock.
    """
    path = sock_dir / protocol.CONTROL_SOCKET_NAME
    server = ControlServer(path, on_command=_echo_ack)
    try:
        first = ControlClient(path, on_event=_EventSink())
        first.close()
        sink = _EventSink()
        second = ControlClient(path, on_event=sink)
        try:
            second.send_command({"cmd": protocol.CMD_HELLO})
            _wait_until(lambda: len(sink.snapshot()) >= 1)
            assert sink.snapshot()[0]["cmd"] == protocol.CMD_HELLO
        finally:
            second.close()
    finally:
        server.close()


def test_control_send_event_without_client_drops(sock_dir: Path) -> None:
    """Guards demo_v7/ipc/channel.py:97-100 (`outbox = self._outbox;
    if outbox is None: return`) plus the per-connection outbox created at
    channel.py:132/140-142. Hoist that queue into __init__ (one permanent
    outbox — the obvious "simplification") and this test alone fails: the
    100 pre-connect progress events are replayed to the first GUI that
    attaches, which would rewind its timeline.
    """
    path = sock_dir / protocol.CONTROL_SOCKET_NAME
    server = ControlServer(path, on_command=_echo_ack)
    try:
        # Must return immediately and silently with nobody connected.
        started = time.monotonic()
        for _ in range(100):
            server.send_event({"event": protocol.EVT_PROGRESS, "stage": "preload", "ok": True})
        assert time.monotonic() - started < 1.0
        # A later client sees only events sent after it connected.
        sink = _EventSink()
        client = ControlClient(path, on_event=sink)
        try:
            client.send_command({"cmd": protocol.CMD_HELLO})
            _wait_until(lambda: len(sink.snapshot()) >= 1)
            assert sink.snapshot() == [
                {"event": protocol.EVT_ACK, "cmd": protocol.CMD_HELLO, "ok": True}
            ]
        finally:
            client.close()
    finally:
        server.close()


def test_frames_round_trip_all_channels(sock_dir: Path) -> None:
    """Guards demo_v7/ipc/channel.py:300 `self._pending[channel] = (header, payload)`
    — the pending slot is keyed PER CHANNEL — together with the batch drain at
    channel.py:343. Collapse it to one global latest slot (a clear() before the
    assignment) and this test alone fails: staged_runtime publishes rgb + depth +
    overlay/composite/gaussian concurrently, so a global slot starves channels.
    """
    path = sock_dir / protocol.FRAMES_SOCKET_NAME
    server = FrameStreamServer(path)
    sink = _FrameSink()
    client = FrameStreamClient(path, on_frame=sink)
    try:
        payloads = {ch: ch.encode() * 100 for ch in protocol.FRAME_CHANNELS}
        for ch, payload in payloads.items():
            server.publish(ch, payload, width=640, height=480)
        _wait_until(lambda: len(sink.snapshot()) >= len(payloads))
        by_channel = {header.channel: (header, payload) for header, payload in sink.snapshot()}
        assert set(by_channel) == set(protocol.FRAME_CHANNELS)
        for ch, payload in payloads.items():
            header, received = by_channel[ch]
            assert received == payload
            assert header.seq == 1
            assert header.payload_len == len(payload)
            assert (header.width, header.height) == (640, 480)
            assert header.t_service_s > 0.0
    finally:
        client.close()
        server.close()


@pytest.mark.parametrize(
    "channel, attached, total, blob, reader_delay_s",
    [
        # Big payloads that cross the UDS buffer, read by a slow consumer.
        pytest.param(
            protocol.CH_COMPOSITE, True, 30, b"x" * 200_000, 0.02, id="slow-attached-reader"
        ),
        # Nobody attached yet: only the newest frame survives the wait.
        pytest.param(protocol.CH_RGB, False, 50, b"", 0.0, id="no-client-attached"),
        # Nobody attached, 100 KB a frame: publish still must not block.
        pytest.param(
            protocol.CH_DEPTH, False, 200, b"y" * 100_000, 0.0, id="no-client-big-payloads"
        ),
    ],
)
def test_frames_latest_wins_under_slow_reader(
    sock_dir: Path,
    channel: str,
    attached: bool,
    total: int,
    blob: bytes,
    reader_delay_s: float,
) -> None:
    """Two distinct lines. (a) demo_v7/ipc/channel.py:396-398 — the client reads
    exactly header.payload_len bytes and bails on a short read; swap in
    reader.read1(...) (one syscall's worth) and only the slow-attached-reader
    row fails, because its 200 KB payloads are the only ones that cross the UDS
    buffer. (b) the latest-wins staging in FrameStreamServer.publish: stage into
    per-channel lists, or into a blocking bounded queue, and every row fails —
    publish would either queue stale frames behind a slow GUI or block the
    service pipeline on one.
    """
    path = sock_dir / protocol.FRAMES_SOCKET_NAME
    server = FrameStreamServer(path)
    sink = _FrameSink(delay_s=reader_delay_s)
    client = FrameStreamClient(path, on_frame=sink) if attached else None
    try:
        # publish() is staged, never sent inline: it must never block the
        # service loop, attached slow reader or not.
        started = time.monotonic()
        for i in range(1, total + 1):
            server.publish(channel, blob + str(i).encode(), width=8, height=8)
        assert time.monotonic() - started < 1.0

        if not attached:
            client = FrameStreamClient(path, on_frame=sink)
            _wait_until(lambda: len(sink.snapshot()) >= 1)
            time.sleep(0.2)  # Give any (wrong) extra frames a chance to land.
            frames = sink.snapshot()
            assert len(frames) == 1  # Only the newest frame was kept.
        else:
            _wait_until(lambda: any(h.seq == total for h, _ in sink.snapshot()))
            frames = sink.snapshot()
            seqs = [h.seq for h, _ in frames]
            assert seqs == sorted(seqs)  # Wire order preserved.
            assert len(frames) < total  # Stale frames were skipped, not queued.
        assert frames[-1][0].seq == total
        assert frames[-1][1] == blob + str(total).encode()
    finally:
        if client is not None:
            client.close()
        server.close()


def test_frames_reaccepts_after_client_disconnect(sock_dir: Path) -> None:
    """Guards demo_v7/ipc/channel.py:319 (the `while not self._closed.is_set():`
    loop in FrameStreamServer._serve_loop) and channel.py:349-350
    (`except OSError: return`, so _send_frames unwinds back to accept()). Serve
    exactly one client and this test alone fails — not a duplicate of the
    control-side twin: different class, different loop.
    """
    path = sock_dir / protocol.FRAMES_SOCKET_NAME
    server = FrameStreamServer(path)
    try:
        first_sink = _FrameSink()
        first = FrameStreamClient(path, on_frame=first_sink)
        server.publish(protocol.CH_RGB, b"first", width=2, height=2)
        _wait_until(lambda: len(first_sink.snapshot()) >= 1)
        first.close()

        sink = _FrameSink()
        second = FrameStreamClient(path, on_frame=sink)
        try:
            # Keep publishing (a real service streams continuously): frames
            # published while the disconnect is still undetected may be lost.
            def _second_got_frame() -> bool:
                server.publish(protocol.CH_RGB, b"second", width=2, height=2)
                time.sleep(0.02)
                return len(sink.snapshot()) >= 1

            _wait_until(_second_got_frame)
            assert sink.snapshot()[0][1] == b"second"
        finally:
            second.close()
    finally:
        server.close()


def test_server_close_ends_client_reader(sock_dir: Path) -> None:
    """Guards demo_v7/ipc/channel.py:58-62 — `_shutdown_quietly` swallowing
    OSError, as used by ControlClient.close (channel.py:234). Drop the
    try/except and sock.shutdown raises ENOTCONN on a dead peer: exactly the
    d84b608 scenario where the camera service exits first and
    session.shutdown()/teardown() then closes its clients.
    """
    path = sock_dir / protocol.CONTROL_SOCKET_NAME
    server = ControlServer(path, on_command=_echo_ack)
    sink = _EventSink()
    client = ControlClient(path, on_event=sink)
    try:
        server.close()
        # Client close after server death must not hang or raise.
        client.close()
    finally:
        client.close()


def test_stale_socket_file_is_replaced(sock_dir: Path) -> None:
    """Socket-file lifecycle. Guards demo_v7/ipc/channel.py:48
    `socket_path.unlink(missing_ok=True)` in _bind_unix_listener: delete that
    line and this test alone fails with EADDRINUSE, because the socket dir is a
    FIXED reused path, `{base_path}/v7_sockets` (config/default.yaml:23), so a
    SIGKILLed service leaves control.sock/frames.sock behind forever. Also pins
    the clean-exit half (channel.py:118/316): close() unlinks what it bound.
    """
    path = sock_dir / protocol.CONTROL_SOCKET_NAME
    frames_path = sock_dir / protocol.FRAMES_SOCKET_NAME
    first = ControlServer(path, on_command=_echo_ack)
    frame_server = FrameStreamServer(frames_path)
    first.close()
    frame_server.close()
    assert not path.exists()
    assert not frames_path.exists()

    # Simulate a crashed service leaving the path behind.
    path.touch()
    server = ControlServer(path, on_command=_echo_ack)
    sink = _EventSink()
    client = ControlClient(path, on_event=sink)
    try:
        client.send_command({"cmd": protocol.CMD_HELLO})
        _wait_until(lambda: len(sink.snapshot()) >= 1)
    finally:
        client.close()
        server.close()
