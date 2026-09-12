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

class _NullGuiLoop:
    """Inert stand-in for contexts that never display anything."""

    def submit(self, window_name: str, frame: Any) -> None:
        return None

    def close_window(self, window_name: str) -> None:
        return None

    def shutdown(self) -> None:
        return None
