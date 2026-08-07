"""Reusable Qt widgets for the demo_v7 GUI.

Constraints these widgets enforce (see ``demo_v7/DESIGN_CONTRACTS.md``):

- Every method here is **main-thread only**. IPC callbacks arrive on socket
  reader threads; ``MainWindow`` hops them onto the Qt main thread via
  signals before any widget is touched.
- ``ImageView`` is latest-wins: a burst of ``setFrame`` calls keeps only the
  newest JPEG and repaints at most ~30 Hz (one coalescing timer per view),
  mirroring the demo_v6_2 live-viewer discipline of never queueing frames.
- ``VideoLoop`` plays mp4 files with ``cv2.VideoCapture`` on a ``QTimer`` so
  the GUI has no QtMultimedia/gstreamer dependency; it pauses automatically
  while hidden (e.g. an inactive Review tab).
- No per-pixel Python loops: decode/convert stays in cv2/Qt C++ code.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QRect, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

# One coalesced repaint per interval bounds the GUI decode cost regardless of
# how fast frames arrive on the socket (service caps are 10-20 Hz per channel).
REPAINT_INTERVAL_MS = 33

_BACKGROUND_COLOR = QColor("#16181c")
_PLACEHOLDER_COLOR = QColor("#5f6368")


def jpeg_to_qimage(jpeg_bytes: bytes) -> QImage | None:
    """Decode a JPEG buffer to a detached ``QImage`` (None if undecodable)."""
    array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return bgr_to_qimage(bgr)


def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    """Wrap a BGR ndarray as a ``QImage`` and copy it off the numpy buffer."""
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    bgr = np.ascontiguousarray(bgr)
    height, width = bgr.shape[:2]
    image = QImage(bgr.data, width, height, bgr.strides[0], QImage.Format.Format_BGR888)
    # ``QImage`` above only wraps the numpy buffer; copy so the buffer may die.
    return image.copy()


class ImageView(QWidget):
    """Aspect-preserving image surface fed by JPEG frames or static images.

    ``setFrame`` is the live path (latest-wins, coalesced repaint);
    ``setImage``/``setImagePath`` are the immediate path for stills and cancel
    any pending live frame so a frozen frame-0 candidate cannot be overwritten
    by a stale queued live frame.
    """

    def __init__(
        self,
        placeholder: str = "",
        *,
        hint_size: tuple[int, int] = (640, 360),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._placeholder = placeholder
        self._hint_size = QSize(*hint_size)
        self._pixmap: QPixmap | None = None
        self._pending_jpeg: bytes | None = None
        self._repaint_timer = QTimer(self)
        self._repaint_timer.setInterval(REPAINT_INTERVAL_MS)
        self._repaint_timer.setSingleShot(True)
        self._repaint_timer.timeout.connect(self._flush_pending)
        self.setMinimumSize(120, 90)

    def setFrame(self, jpeg_bytes: bytes) -> None:
        """Queue the newest JPEG; older undecoded frames are dropped."""
        self._pending_jpeg = jpeg_bytes
        if not self._repaint_timer.isActive():
            self._repaint_timer.start()

    def setImage(self, image: QImage | None) -> None:
        """Show a still immediately, cancelling any pending live frame."""
        self._pending_jpeg = None
        self._repaint_timer.stop()
        self._pixmap = QPixmap.fromImage(image) if image is not None else None
        self.update()

    def setImagePath(self, path: str | Path) -> bool:
        """Load a still from disk; returns False (and clears) when unreadable."""
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.setImage(None)
            return False
        self._pending_jpeg = None
        self._repaint_timer.stop()
        self._pixmap = pixmap
        self.update()
        return True

    def clear(self) -> None:
        self.setImage(None)

    def hasImage(self) -> bool:
        return self._pixmap is not None or self._pending_jpeg is not None

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return self._hint_size

    def _flush_pending(self) -> None:
        pending = self._pending_jpeg
        self._pending_jpeg = None
        if pending is None:
            return
        image = jpeg_to_qimage(pending)
        if image is None:
            return
        self._pixmap = QPixmap.fromImage(image)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BACKGROUND_COLOR)
        if self._pixmap is None:
            if self._placeholder:
                painter.setPen(_PLACEHOLDER_COLOR)
                painter.drawText(
                    self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder
                )
            painter.end()
            return
        target_size = self._pixmap.size()
        target_size.scale(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        target = QRect(0, 0, target_size.width(), target_size.height())
        target.moveCenter(self.rect().center())
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawPixmap(target, self._pixmap)
        painter.end()


class VideoLoop(QWidget):
    """Loops an mp4 via ``cv2.VideoCapture`` on a ``QTimer``.

    Playback runs only while the widget is visible (show/hide events start and
    stop the timer), so a turntable video in a background tab costs nothing.
    """

    def __init__(
        self,
        placeholder: str = "",
        *,
        hint_size: tuple[int, int] = (640, 360),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._view = ImageView(placeholder, hint_size=hint_size, parent=self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)
        self._capture: cv2.VideoCapture | None = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    def setVideoPath(self, path: str | Path) -> bool:
        """Open (or replace) the looping video; returns False when unopenable."""
        self.stop()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            return False
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1.0 or fps > 120.0:
            fps = 30.0
        self._capture = capture
        self._timer.setInterval(max(1, int(round(1000.0 / fps))))
        if self.isVisible():
            self._timer.start()
        return True

    def stop(self) -> None:
        self._timer.stop()
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        if self._capture is not None and not self._timer.isActive():
            self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().hideEvent(event)
        self._timer.stop()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.stop()
        super().closeEvent(event)

    def _advance(self) -> None:
        if self._capture is None:
            return
        ok, frame = self._capture.read()
        if not ok:
            # Loop: rewind and retry once; a twice-unreadable file stops.
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
            if not ok:
                self.stop()
                return
        self._view.setImage(bgr_to_qimage(frame))


def _format_elapsed_ms(elapsed_ms: float) -> str:
    """Human elapsed: sub-second in ms, then seconds, then m+s."""
    if elapsed_ms < 1000.0:
        return f"{elapsed_ms:.0f} ms"
    seconds = elapsed_ms / 1000.0
    if seconds < 120.0:
        return f"{seconds:.1f} s"
    return f"{int(seconds // 60)} m {int(seconds % 60)} s"


_GLYPH_PENDING = "○"
_GLYPH_OK = "✓"
_GLYPH_FAIL = "✗"
# Spinner frames for rows in the running state (ticked by the shared timer).
_SPINNER_FRAMES = ("◐", "◓", "◑", "◒")
_SPINNER_STYLE = "color: #8ab4f8; font-weight: bold;"

_GLYPH_STYLES = {
    _GLYPH_PENDING: "color: #5f6368;",
    _GLYPH_OK: "color: #81c995; font-weight: bold;",
    _GLYPH_FAIL: "color: #f28b82; font-weight: bold;",
}

_TICK_INTERVAL_MS = 250


class ProgressTimeline(QWidget):
    """Ordered stage rows with a status glyph, detail text and elapsed time.

    Stages are pre-declared with ``setStages`` so the operator sees the whole
    plan up front; unknown stage names reported later append extra rows (the
    service may forward demo_v6_2 pipeline_status stages verbatim).

    Visual-first (owner rule 2026-08-06: the GUI shows progress, logs go to
    stdout + file): ``begin`` puts a row into a live running state — spinner
    glyph + elapsed seconds ticking on a timer — and ``report`` settles it
    to ✓/✗ with the final duration.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(6)
        self._grid.setColumnStretch(2, 1)
        # stage -> (glyph, name, detail, elapsed) labels
        self._rows: dict[str, tuple[QLabel, QLabel, QLabel, QLabel]] = {}
        # stage -> perf-counter start of the running state
        self._running: dict[str, float] = {}
        self._spin_phase = 0
        self._ticker = QTimer(self)
        self._ticker.setInterval(_TICK_INTERVAL_MS)
        self._ticker.timeout.connect(self._tick)

    def setStages(self, stages: list[tuple[str, str]]) -> None:
        """Declare the expected (stage_key, human_label) rows in order."""
        for key, label in stages:
            self._ensure_row(key, label)

    def clear(self) -> None:
        """Drop every row (declared and appended) and its glyph/label widgets."""
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._rows.clear()
        self._running.clear()
        self._ticker.stop()

    def setRowLabel(self, stage: str, label: str) -> None:  # noqa: N802
        """Rename a declared row (e.g. the generate row per backend)."""
        row = self._rows.get(stage)
        if row is not None:
            row[1].setText(label)

    def begin(self, stage: str, detail: str = "") -> None:
        """Put ``stage`` into the running state (spinner + live elapsed)."""
        glyph_label, _name, detail_label, elapsed_label = self._ensure_row(
            stage, stage
        )
        if stage not in self._running:
            self._running[stage] = time.perf_counter()
        glyph_label.setText(_SPINNER_FRAMES[self._spin_phase])
        glyph_label.setStyleSheet(_SPINNER_STYLE)
        if detail:
            detail_label.setText(detail)
        elapsed_label.setText("0 s")
        if not self._ticker.isActive():
            self._ticker.start()

    def report(
        self,
        stage: str,
        detail: str = "",
        *,
        ok: bool = True,
        elapsed_ms: float | None = None,
    ) -> None:
        """Record one progress event for ``stage`` (row appended if unknown)."""
        glyph_label, _name, detail_label, elapsed_label = self._ensure_row(stage, stage)
        started_s = self._running.pop(stage, None)
        if not self._running:
            self._ticker.stop()
        glyph = _GLYPH_OK if ok else _GLYPH_FAIL
        glyph_label.setText(glyph)
        glyph_label.setStyleSheet(_GLYPH_STYLES[glyph])
        if detail:
            detail_label.setText(detail)
        if elapsed_ms is None and started_s is not None:
            elapsed_ms = (time.perf_counter() - started_s) * 1000.0
        if elapsed_ms is not None:
            elapsed_label.setText(_format_elapsed_ms(float(elapsed_ms)))

    def stopAll(self) -> None:  # noqa: N802 (Qt style)
        """Settle every running row back to pending and stop the ticker.

        Used when the chain dies (a failure event or a fatal): rows that
        were spinning did not fail themselves, but nothing will ever
        complete them — a frozen spinner would read as live progress.
        """
        for stage in list(self._running):
            glyph_label, _name, _detail, _elapsed = self._rows[stage]
            glyph_label.setText(_GLYPH_PENDING)
            glyph_label.setStyleSheet(_GLYPH_STYLES[_GLYPH_PENDING])
        self._running.clear()
        self._ticker.stop()

    def _tick(self) -> None:
        """Advance the spinner + live elapsed of every running row."""
        self._spin_phase = (self._spin_phase + 1) % len(_SPINNER_FRAMES)
        frame = _SPINNER_FRAMES[self._spin_phase]
        now_s = time.perf_counter()
        for stage, started_s in self._running.items():
            glyph_label, _name, _detail, elapsed_label = self._rows[stage]
            glyph_label.setText(frame)
            elapsed_label.setText(_format_elapsed_ms((now_s - started_s) * 1000.0))

    def _ensure_row(self, key: str, label: str) -> tuple[QLabel, QLabel, QLabel, QLabel]:
        if key in self._rows:
            return self._rows[key]
        row = len(self._rows)
        glyph = QLabel(_GLYPH_PENDING, self)
        glyph.setStyleSheet(_GLYPH_STYLES[_GLYPH_PENDING])
        name = QLabel(label, self)
        detail = QLabel("", self)
        detail.setStyleSheet("color: #9aa0a6;")
        detail.setWordWrap(True)
        elapsed = QLabel("", self)
        elapsed.setStyleSheet("color: #9aa0a6;")
        elapsed.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._grid.addWidget(glyph, row, 0)
        self._grid.addWidget(name, row, 1)
        self._grid.addWidget(detail, row, 2)
        self._grid.addWidget(elapsed, row, 3)
        self._rows[key] = (glyph, name, detail, elapsed)
        return self._rows[key]


class CaptionedImage(QWidget):
    """A disk-loaded still with a caption underneath, for artifact grids."""

    def __init__(
        self,
        caption: str,
        path: str | Path,
        *,
        max_width: int = 360,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        image_label = QLabel(self)
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            image_label.setText("(无法读取图片)")
            image_label.setStyleSheet("color: #f28b82;")
        else:
            if pixmap.width() > max_width:
                pixmap = pixmap.scaledToWidth(
                    max_width, Qt.TransformationMode.SmoothTransformation
                )
            image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        caption_label = QLabel(caption, self)
        caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        caption_label.setStyleSheet("color: #9aa0a6; font-size: 12px;")
        layout.addWidget(image_label)
        layout.addWidget(caption_label)
