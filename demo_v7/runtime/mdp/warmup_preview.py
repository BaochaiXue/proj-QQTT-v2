"""Warm-up live RGB camera-input preview window (every downstream mode).

During the hold-still warm-up window the operator needs to see what the camera
sees (framing, both hands visible) — this small window opens with capture in
EVERY ``downstream.mode`` (it is NOT the tracking-chunk visualizer, whose
per-mode policy is unchanged) and mirrors the live RGB input straight from
``input_preview_slot`` at frame cadence, with zero disk IO.

The preview thread only composes frames (copy + text raster); the actual
window is driven by the process-wide ``CvGuiLoop`` GUI thread — Qt/GTK
HighGUI cannot survive window ownership moving between threads, so no client
ever calls ``imshow`` itself.

Lifecycle: it closes the moment warm-up ends —
- normal end: shape prior ready / formal timeline opens (the WARMUP_FINISHED
  banner site calls ``close()``), or seg-warm-up completion when shape-prior
  warm-up is disabled;
- failure / cancel / early exit: ``stop_event`` (set by fatal worker errors and
  process teardown) ends the compose loop immediately, and ``stop()`` also
  calls ``close()`` so the window never outlives the run.

Best-effort GUI: composition failures disable the preview with one log line
(display/backend failures are already absorbed inside the GUI loop) and never
touch the capture pipeline.
"""

from __future__ import annotations

import threading
import time
from typing import Any


class WarmupRgbPreview:
    """Small always-on-top-of-warmup live RGB window; never raises."""

    WINDOW_NAME = "Demo v6.2 - camera input (warm-up)"

    def __init__(
        self,
        *,
        input_preview_slot: Any,
        gui: Any,
        stop_event: threading.Event,
        enabled: bool = True,
        cv2_module: Any | None = None,
    ) -> None:
        self._slot = input_preview_slot
        self._gui = gui
        self._stop_event = stop_event
        self._close_event = threading.Event()
        self._enabled = bool(enabled)
        self._cv2 = cv2_module
        self._thread: threading.Thread | None = None
        self._started_perf_s: float | None = None

    def start(self) -> None:
        """Start the compose thread; a GUI-less environment disables it."""
        if not self._enabled or self._thread is not None:
            return
        if self._cv2 is None:
            try:
                import cv2  # noqa: PLC0415

                self._cv2 = cv2
            except Exception as exc:
                print(f"[demo_v6_1] warmup rgb preview disabled: {exc}", flush=True)
                return
        self._started_perf_s = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run, name="warmup-rgb-preview", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        """End the preview (warm-up finished, failed, cancelled, or run exit)."""
        self._close_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._gui.close_window(self.WINDOW_NAME)

    def _run(self) -> None:
        cv2 = self._cv2
        last_seq = -1
        try:
            while not self._stop_event.is_set() and not self._close_event.is_set():
                packet = self._slot.get_latest_after(last_seq)
                if packet is not None:
                    last_seq = int(packet.seq)
                    frame = packet.color_bgr.copy()
                    elapsed_s = time.perf_counter() - (self._started_perf_s or 0.0)
                    label = f"warm-up in progress - hold still ({elapsed_s:.0f}s)"
                    cv2.putText(
                        frame, label, (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame, label, (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA,
                    )
                    # The GUI loop owns the buffer from here on.
                    self._gui.submit(self.WINDOW_NAME, frame)
                time.sleep(0.033)
        except Exception as exc:
            # Composition failure must never break capture or warm-up.
            print(f"[demo_v6_1] warmup rgb preview disabled: {exc}", flush=True)
