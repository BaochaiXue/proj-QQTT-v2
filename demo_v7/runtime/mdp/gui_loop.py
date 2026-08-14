"""Single process-wide OpenCV HighGUI thread for every demo window.

Qt/GTK HighGUI binds its event loop to the FIRST thread that creates a
window; after that thread exits, another thread's ``namedWindow`` hangs
forever (verified empirically on this cv2 build: a second GUI thread blocked
>10s in ``namedWindow`` with zero errors). So the camera process runs ONE
persistent daemon GUI thread for its whole lifetime: clients (warm-up RGB
preview, live data-process viewer) compose display frames on their own
threads — pure numpy / raster ``putText`` work — and ``submit`` them here;
only this thread ever touches ``namedWindow`` / ``imshow`` / ``waitKey`` /
``destroyWindow``.

Semantics:
- latest-wins per window: an unshown frame is replaced silently;
- ``close_window`` destroys the window and tombstones its name — later
  submits to a closed window are ignored (a racing producer can never
  resurrect it);
- lazy start: the thread starts on the first submit, so headless or
  windows-disabled runs never initialize HighGUI at all;
- best-effort: any GUI failure disables ALL windows with one log line and
  never touches the pipeline (same policy as pipeline_status);
- submitted frames become loop-owned: clients must hand over freshly
  composed buffers and never mutate them afterwards.
"""

from __future__ import annotations

import threading
import time
from typing import Any

GUI_PUMP_MS = 33


class CvGuiLoop:
    """One daemon thread that owns every OpenCV window; never raises."""

    def __init__(
        self,
        *,
        stop_event: threading.Event,
        cv2_module: Any | None = None,
    ) -> None:
        self._stop_event = stop_event
        self._close_event = threading.Event()
        self._cv2 = cv2_module
        self._lock = threading.Lock()
        self._frames: dict[str, Any] = {}
        self._close_requests: set[str] = set()
        self._tombstoned: set[str] = set()
        self._thread: threading.Thread | None = None
        self._failed = False

    def submit(self, window_name: str, frame: Any) -> None:
        """Hand one freshly composed frame to the GUI thread (latest wins)."""
        name = str(window_name)
        with self._lock:
            if self._failed or name in self._tombstoned:
                return
            self._frames[name] = frame
        self._ensure_started()

    def close_window(self, window_name: str) -> None:
        """Destroy one window and ignore any later submits to its name."""
        name = str(window_name)
        with self._lock:
            self._tombstoned.add(name)
            self._frames.pop(name, None)
            if self._thread is None:
                return
            self._close_requests.add(name)

    def shutdown(self) -> None:
        """End the GUI thread (run finished, failed, cancelled, teardown)."""
        self._close_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def _ensure_started(self) -> None:
        with self._lock:
            if self._thread is not None or self._failed:
                return
            if self._cv2 is None:
                try:
                    import cv2  # noqa: PLC0415

                    self._cv2 = cv2
                except Exception as exc:
                    self._failed = True
                    print(f"[demo_v6_1] gui loop disabled: {exc}", flush=True)
                    return
            self._thread = threading.Thread(
                target=self._run, name="cv-gui-loop", daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        cv2 = self._cv2
        created: set[str] = set()
        try:
            while not self._stop_event.is_set() and not self._close_event.is_set():
                with self._lock:
                    pending = dict(self._frames)
                    self._frames.clear()
                    closes = set(self._close_requests)
                    self._close_requests.clear()
                for name in closes:
                    if name in created:
                        cv2.destroyWindow(name)
                        created.discard(name)
                        print(
                            f"[demo_v6_1] gui loop: window closed: {name}",
                            flush=True,
                        )
                for name, frame in pending.items():
                    if name in closes:
                        continue
                    if name not in created:
                        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
                        created.add(name)
                        print(
                            f"[demo_v6_1] gui loop: window opened: {name}",
                            flush=True,
                        )
                    cv2.imshow(name, frame)
                # waitKey pumps the GUI event loop and paces the thread —
                # but ONLY while a window exists: with zero HighGUI windows
                # waitKey returns immediately (measured: ~0.1us on this Qt
                # build), which would busy-spin a full core. Windowless
                # phases pace on the close event instead.
                if created:
                    cv2.waitKey(GUI_PUMP_MS)
                else:
                    self._close_event.wait(GUI_PUMP_MS / 1000.0)
        except Exception as exc:
            # A broken/absent display must never break the pipeline.
            with self._lock:
                self._failed = True
            print(f"[demo_v6_1] gui loop disabled: {exc}", flush=True)
        finally:
            for name in created:
                try:
                    cv2.destroyWindow(name)
                except Exception:
                    pass
            try:
                if created:
                    cv2.waitKey(1)
            except Exception:
                pass


class _NullGuiLoop:
    """Inert stand-in for contexts that never display anything."""

    def submit(self, window_name: str, frame: Any) -> None:
        return None

    def close_window(self, window_name: str) -> None:
        return None

    def shutdown(self) -> None:
        return None
