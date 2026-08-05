"""UDS channel endpoints for the demo_v7 GUI <-> camera-service link.

Implements the two sockets defined in :mod:`demo_v7.ipc.protocol`:

- ``control.sock`` (``ControlServer`` / ``ControlClient``): one JSON object
  per UTF-8 line, both directions.
- ``frames.sock`` (``FrameStreamServer`` / ``FrameStreamClient``): per frame
  one JSON header line (:class:`FrameHeader`) followed by ``payload_len``
  JPEG bytes, service -> GUI only.

Shared discipline: the service side owns each socket (binds/listens), serves
exactly ONE client at a time and re-accepts after a disconnect. Nothing here
may block a pipeline thread — control events go through a bounded outbox
drained by a writer thread, frame publishes swap a latest-wins slot and never
touch the socket. No Qt imports; callbacks run on channel-owned threads and
the consumer is responsible for hopping to its own main thread.
"""

from __future__ import annotations

import json
import queue
import socket
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from demo_v7.ipc.protocol import FrameHeader

# Accept/queue polls exist only so blocking threads notice close() promptly;
# they never delay delivery (queue.get / accept return as soon as work is up).
_ACCEPT_POLL_S = 0.2
_QUEUE_POLL_S = 0.2
_JOIN_TIMEOUT_S = 2.0
# Contract cap for the control outbox: send_event drops beyond this depth so
# a stalled GUI can never grow service memory or block the caller.
_CONTROL_OUTBOX_MAX = 1000


def _encode_json_line(obj: dict) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode("utf-8") + b"\n"


def _bind_unix_listener(socket_path: Path) -> socket.socket:
    """Bind + listen on a fresh UDS path (stale socket files are replaced)."""
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen(1)
    # Timeout only makes the accept loop poll the closed flag; accepted
    # connections are reset to fully blocking below.
    listener.settimeout(_ACCEPT_POLL_S)
    return listener


def _shutdown_quietly(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


class ControlServer:
    """Service-side ``control.sock`` endpoint (one GUI client at a time).

    Threading: a single accept+reader thread owns the listening socket and
    the current connection's receive side — each JSON line is decoded and
    handed to ``on_command`` on that thread, and the ack dict it returns
    (if any) is enqueued immediately. One writer thread per connection
    drains a bounded outbox, so ``send_event`` (callable from any thread)
    is a non-blocking ``put_nowait``: it drops the event when no client is
    connected or the outbox already holds ``_CONTROL_OUTBOX_MAX`` entries.
    Acks and events share the one outbox, which preserves the contract
    order "ack before any state event the command causes" as long as the
    handler emits those state events after returning the ack.
    """

    def __init__(
        self, socket_path: Path, *, on_command: Callable[[dict], dict | None]
    ) -> None:
        self._socket_path = Path(socket_path)
        self._on_command = on_command
        self._listener = _bind_unix_listener(self._socket_path)
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._conn: Optional[socket.socket] = None
        self._outbox: Optional[queue.Queue] = None
        self._thread = threading.Thread(
            target=self._accept_loop, name="control-server", daemon=True
        )
        self._thread.start()

    def send_event(self, event: dict) -> None:
        """Queue one event for the connected GUI; drop it if none / backlogged."""
        with self._lock:
            outbox = self._outbox
        if outbox is None:
            return
        try:
            outbox.put_nowait(event)
        except queue.Full:
            pass

    def close(self) -> None:
        self._closed.set()
        try:
            self._listener.close()
        except OSError:
            pass
        with self._lock:
            conn = self._conn
        if conn is not None:
            _shutdown_quietly(conn)
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
        self._socket_path.unlink(missing_ok=True)

    def _accept_loop(self) -> None:
        while not self._closed.is_set():
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            conn.settimeout(None)
            self._serve_connection(conn)

    def _serve_connection(self, conn: socket.socket) -> None:
        outbox: queue.Queue = queue.Queue(maxsize=_CONTROL_OUTBOX_MAX)
        stop = threading.Event()
        writer = threading.Thread(
            target=self._writer_loop,
            args=(conn, outbox, stop),
            name="control-server-writer",
            daemon=True,
        )
        with self._lock:
            self._conn = conn
            self._outbox = outbox
        writer.start()
        try:
            for raw in conn.makefile("rb"):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ack = self._on_command(obj)
                if ack is not None:
                    try:
                        outbox.put_nowait(ack)
                    except queue.Full:
                        pass
        except (OSError, ValueError):
            pass
        finally:
            with self._lock:
                self._conn = None
                self._outbox = None
            stop.set()
            _shutdown_quietly(conn)
            writer.join(timeout=_JOIN_TIMEOUT_S)
            conn.close()

    @staticmethod
    def _writer_loop(
        conn: socket.socket, outbox: queue.Queue, stop: threading.Event
    ) -> None:
        while not stop.is_set():
            try:
                item = outbox.get(timeout=_QUEUE_POLL_S)
            except queue.Empty:
                continue
            try:
                conn.sendall(_encode_json_line(item))
            except OSError:
                # Peer went away mid-send; unblock the reader too and let
                # the accept loop take the next client.
                _shutdown_quietly(conn)
                return


class ControlClient:
    """GUI-side ``control.sock`` endpoint.

    Threading: one reader thread delivers every decoded event dict to
    ``on_event`` (callbacks run ON that thread — the GUI must hop to its
    main thread itself). ``send_command`` runs on the caller's thread under
    a send lock; command payloads are a few dozen bytes, so a socket-buffer
    stall is not a practical concern. It is fire-and-forget: transport
    errors after the service vanished are swallowed, acks arrive as events
    while the link is up.
    """

    def __init__(
        self, socket_path: Path, *, on_event: Callable[[dict], None]
    ) -> None:
        self._on_event = on_event
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(str(socket_path))
        self._send_lock = threading.Lock()
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._reader_loop, name="control-client", daemon=True
        )
        self._thread.start()

    def send_command(self, cmd: dict) -> None:
        data = _encode_json_line(cmd)
        with self._send_lock:
            try:
                self._sock.sendall(data)
            except OSError:
                pass

    def close(self) -> None:
        self._closed.set()
        _shutdown_quietly(self._sock)
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
        self._sock.close()

    def _reader_loop(self) -> None:
        try:
            for raw in self._sock.makefile("rb"):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if self._closed.is_set():
                    return
                self._on_event(obj)
        except (OSError, ValueError):
            pass


class FrameStreamServer:
    """Service-side ``frames.sock`` endpoint (latest-wins per channel).

    Threading: ``publish`` (called from any pipeline thread) only swaps the
    pending slot for its channel under the condition lock and notifies the
    writer — it never touches the socket, so a slow, stalled or absent GUI
    can never block the pipeline. A single accept+writer thread owns the
    socket: it serves one client at a time, re-accepts after a disconnect,
    and on each cycle sends the newest pending frame per channel; frames
    superseded while a send was in flight are silently skipped (their seq
    numbers reveal the gaps to the client). ``seq`` increments per publish
    per channel, so it counts published — not delivered — frames.
    """

    def __init__(self, socket_path: Path) -> None:
        self._socket_path = Path(socket_path)
        self._listener = _bind_unix_listener(self._socket_path)
        self._cond = threading.Condition()
        # channel -> (header, payload); replaced wholesale on publish.
        self._pending: dict[str, tuple[FrameHeader, bytes]] = {}
        self._seq: dict[str, int] = {}
        self._conn: Optional[socket.socket] = None
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._serve_loop, name="frame-stream-server", daemon=True
        )
        self._thread.start()

    def publish(
        self, channel: str, jpeg_bytes: bytes, *, width: int, height: int
    ) -> None:
        """Stage the newest frame for ``channel``; never blocks the caller."""
        payload = bytes(jpeg_bytes)
        with self._cond:
            seq = self._seq.get(channel, 0) + 1
            self._seq[channel] = seq
            header = FrameHeader(
                channel=channel,
                seq=seq,
                payload_len=len(payload),
                width=int(width),
                height=int(height),
                t_service_s=time.perf_counter(),
            )
            self._pending[channel] = (header, payload)
            self._cond.notify_all()

    def close(self) -> None:
        self._closed.set()
        with self._cond:
            conn = self._conn
            self._cond.notify_all()
        try:
            self._listener.close()
        except OSError:
            pass
        if conn is not None:
            _shutdown_quietly(conn)
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
        self._socket_path.unlink(missing_ok=True)

    def _serve_loop(self) -> None:
        while not self._closed.is_set():
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            conn.settimeout(None)
            with self._cond:
                self._conn = conn
            try:
                self._send_frames(conn)
            finally:
                with self._cond:
                    self._conn = None
                conn.close()

    def _send_frames(self, conn: socket.socket) -> None:
        while True:
            with self._cond:
                while not self._pending and not self._closed.is_set():
                    self._cond.wait(timeout=_QUEUE_POLL_S)
                if self._closed.is_set():
                    return
                batch = [self._pending.pop(ch) for ch in list(self._pending)]
            for header, payload in batch:
                try:
                    conn.sendall(
                        _encode_json_line(header.to_json_obj()) + payload
                    )
                except OSError:
                    return


class FrameStreamClient:
    """GUI-side ``frames.sock`` endpoint.

    Threading: one reader thread parses each header line, reads exactly
    ``payload_len`` bytes and calls ``on_frame(header, jpeg_bytes)`` on
    that thread. The GUI-side latest-wins display policy lives in the
    consumer (it replaces its pending frame per channel); this class only
    guarantees frames are delivered whole and in wire order.
    """

    def __init__(
        self,
        socket_path: Path,
        *,
        on_frame: Callable[[FrameHeader, bytes], None],
    ) -> None:
        self._on_frame = on_frame
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(str(socket_path))
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._reader_loop, name="frame-stream-client", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._closed.set()
        _shutdown_quietly(self._sock)
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
        self._sock.close()

    def _reader_loop(self) -> None:
        reader = self._sock.makefile("rb")
        try:
            while not self._closed.is_set():
                line = reader.readline()
                if not line:
                    return  # EOF: server closed the stream.
                try:
                    header = FrameHeader.from_json_obj(json.loads(line))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    return  # Header corrupt -> byte stream is unrecoverable.
                payload = reader.read(header.payload_len)
                if payload is None or len(payload) != header.payload_len:
                    return  # Truncated payload (disconnect mid-frame).
                if self._closed.is_set():
                    return
                self._on_frame(header, payload)
        except (OSError, ValueError):
            pass
