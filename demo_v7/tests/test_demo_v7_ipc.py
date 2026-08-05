"""Unit tests for demo_v7.ipc.channel — real UDS in a tmpdir, pure CPU.

Covers the contract points: command/event round trips over control.sock,
ack-then-event ordering, re-accept after a client disconnect, latest-wins
frame delivery under a slow reader, non-blocking publish/send with no client
attached, and clean close (threads gone, socket files unlinked). No Qt, no
GPU, no demo_v6_2 imports.
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


def test_control_command_event_round_trip(sock_dir: Path) -> None:
    path = sock_dir / protocol.CONTROL_SOCKET_NAME
    received_cmds: queue.Queue = queue.Queue()

    def on_command(obj: dict) -> dict:
        received_cmds.put(obj)
        return _echo_ack(obj)

    server = ControlServer(path, on_command=on_command)
    sink = _EventSink()
    client = ControlClient(path, on_event=sink)
    try:
        client.send_command({"cmd": protocol.CMD_HELLO})
        assert received_cmds.get(timeout=_WAIT_S) == {"cmd": protocol.CMD_HELLO}
        _wait_until(lambda: len(sink.snapshot()) >= 1)
        # Ack arrives before the event sent after it.
        server.send_event({"event": protocol.EVT_STATE, "state": protocol.STATE_PREVIEW})
        _wait_until(lambda: len(sink.snapshot()) >= 2)
        events = sink.snapshot()
        assert events[0] == {"event": protocol.EVT_ACK, "cmd": protocol.CMD_HELLO, "ok": True}
        assert events[1] == {"event": protocol.EVT_STATE, "state": protocol.STATE_PREVIEW}
    finally:
        client.close()
        server.close()


def test_control_on_command_may_skip_ack(sock_dir: Path) -> None:
    path = sock_dir / protocol.CONTROL_SOCKET_NAME
    seen = queue.Queue()
    server = ControlServer(path, on_command=lambda obj: (seen.put(obj), None)[1])
    sink = _EventSink()
    client = ControlClient(path, on_event=sink)
    try:
        client.send_command({"cmd": protocol.CMD_SHUTDOWN})
        assert seen.get(timeout=_WAIT_S) == {"cmd": protocol.CMD_SHUTDOWN}
        server.send_event({"event": protocol.EVT_STATE, "state": protocol.STATE_FINISHED})
        _wait_until(lambda: len(sink.snapshot()) == 1)
        assert sink.snapshot()[0]["state"] == protocol.STATE_FINISHED
    finally:
        client.close()
        server.close()


def test_control_reaccepts_after_client_disconnect(sock_dir: Path) -> None:
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


def test_frames_latest_wins_before_connect(sock_dir: Path) -> None:
    path = sock_dir / protocol.FRAMES_SOCKET_NAME
    server = FrameStreamServer(path)
    try:
        total = 50
        for i in range(1, total + 1):
            server.publish(protocol.CH_RGB, f"frame-{i}".encode(), width=4, height=4)
        sink = _FrameSink()
        client = FrameStreamClient(path, on_frame=sink)
        try:
            _wait_until(lambda: len(sink.snapshot()) >= 1)
            time.sleep(0.2)  # Give any (wrong) extra frames a chance to land.
            frames = sink.snapshot()
            assert len(frames) == 1
            header, payload = frames[0]
            assert header.seq == total
            assert payload == f"frame-{total}".encode()
        finally:
            client.close()
    finally:
        server.close()


def test_frames_latest_wins_under_slow_reader(sock_dir: Path) -> None:
    path = sock_dir / protocol.FRAMES_SOCKET_NAME
    server = FrameStreamServer(path)
    sink = _FrameSink(delay_s=0.02)
    client = FrameStreamClient(path, on_frame=sink)
    try:
        total = 30
        blob = b"x" * 200_000  # Big enough to overrun the UDS buffers.
        for i in range(1, total + 1):
            server.publish(protocol.CH_COMPOSITE, blob + str(i).encode(), width=8, height=8)
        _wait_until(
            lambda: any(h.seq == total for h, _ in sink.snapshot())
        )
        frames = sink.snapshot()
        seqs = [h.seq for h, _ in frames]
        assert seqs == sorted(seqs)  # Wire order preserved.
        assert len(frames) < total  # Stale frames were skipped, not queued.
        assert frames[-1][0].seq == total
        assert frames[-1][1].endswith(str(total).encode())
    finally:
        client.close()
        server.close()


def test_frames_publish_never_blocks_without_client(sock_dir: Path) -> None:
    path = sock_dir / protocol.FRAMES_SOCKET_NAME
    server = FrameStreamServer(path)
    try:
        blob = b"y" * 100_000
        started = time.monotonic()
        for _ in range(200):
            server.publish(protocol.CH_DEPTH, blob, width=8, height=8)
        assert time.monotonic() - started < 1.0
    finally:
        server.close()


def test_frames_reaccepts_after_client_disconnect(sock_dir: Path) -> None:
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


def test_clean_close_unlinks_sockets_and_stops_threads(sock_dir: Path) -> None:
    control_path = sock_dir / protocol.CONTROL_SOCKET_NAME
    frames_path = sock_dir / protocol.FRAMES_SOCKET_NAME
    control_server = ControlServer(control_path, on_command=_echo_ack)
    frame_server = FrameStreamServer(frames_path)
    control_client = ControlClient(control_path, on_event=_EventSink())
    frame_client = FrameStreamClient(frames_path, on_frame=_FrameSink())

    before = threading.active_count()
    assert before >= 5  # Both servers + both clients have live threads.

    control_client.close()
    frame_client.close()
    control_server.close()
    frame_server.close()

    assert not control_path.exists()
    assert not frames_path.exists()
    _wait_until(lambda: threading.active_count() <= before - 4)


def test_server_close_ends_client_reader(sock_dir: Path) -> None:
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
    path = sock_dir / protocol.CONTROL_SOCKET_NAME
    first = ControlServer(path, on_command=_echo_ack)
    first.close()
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
