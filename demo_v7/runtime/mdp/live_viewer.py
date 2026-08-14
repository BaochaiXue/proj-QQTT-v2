"""In-process realtime data-process viewer (pure observer, any downstream mode).

Shows the LATEST fully processed frame the formal pipeline produced — the
paired PCD + tracker result published by ``FormalProductStage`` — the moment
it exists, without waiting for a chunk to form:

- left panel: RGB with the PROCESSED object/hand masks tinted on top and the
  TAPNext++ tracks drawn in their frozen query-rainbow colors (visible tracks
  only; the visible/total count is in the HUD);
- right panel: the world-space object/controller point clouds plus the
  shape-prior points reprojected through the color camera;
- HUD band: seq, end-to-end pipeline latency, per-stage FPS, display drops,
  point/track counts, per-stage timing, shape-prior status, timeline state.

Non-negotiables (mirrors ``WarmupRgbPreview``'s discipline):
- pure observer — reads packet arrays only, draws on copies, never mutates
  or feeds anything back into the pipeline;
- never blocks the pipeline — frames arrive through a latest-wins
  ``LatestSlot`` (a slow viewer silently drops old DISPLAY frames; formal
  data is untouched), and the publisher side is one O(1) slot put;
- no CUDA context, no torch import, no disk IO — numpy + OpenCV drawing on
  the CPU inside the camera subprocess (GPU-1 namespace, GPU untouched);
- best-effort GUI — any display/backend failure disables the viewer with one
  log line and never touches the run;
- single GUI owner — this thread only COMPOSES frames (numpy + raster
  ``putText``); the window itself is driven by the process-wide ``CvGuiLoop``
  thread, because Qt/GTK HighGUI hangs when window ownership moves between
  threads (empirically verified: a second GUI thread blocks forever in
  ``namedWindow`` after the first exits).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import numpy as np

from qqtt.env.camera.table_calibration import TABLE_WORLD_FRAME_KIND

# Display-only constants: tints/layout never touch formal products.
OBJECT_TINT_BGR = (90, 220, 90)
HAND_A_TINT_BGR = (95, 95, 235)
HAND_B_TINT_BGR = (60, 185, 250)
SHAPE_PRIOR_POINT_BGR = (200, 160, 255)
MASK_TINT_ALPHA = 0.45
TRACK_DOT_RADIUS = 1
PCD_DOT_RADIUS = 0
HUD_HEIGHT_PX = 96
DISPLAY_SCALE = 0.75
GUI_PUMP_MS = 33

_HUD_TEXT_BGR = (235, 235, 235)
_HUD_BG_BGR = (28, 28, 28)


def _paint_dots(
    image: np.ndarray,
    points_xy: np.ndarray,
    colors_bgr: np.ndarray,
    *,
    radius: int,
) -> None:
    """Paint square dots via vectorized indexing (display buffer only)."""
    if points_xy.size == 0:
        return
    height, width = image.shape[:2]
    xs0 = points_xy[:, 0].astype(np.int64)
    ys0 = points_xy[:, 1].astype(np.int64)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            xs = xs0 + dx
            ys = ys0 + dy
            ok = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            image[ys[ok], xs[ok]] = colors_bgr[ok]


def _project_world_points_to_pixels(
    points_m: np.ndarray,
    *,
    intrinsics: Any,
    world_to_camera: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project (N,3) points to integer pixel (x, y); returns (pixels, keep).

    ``world_to_camera`` is None when the points are already in the color
    camera frame (uncalibrated runs publish camera-frame products).
    """
    pts = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=bool)
    if world_to_camera is not None:
        pts = pts @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    z = pts[:, 2]
    keep = z > 1e-6
    u = np.zeros_like(z)
    v = np.zeros_like(z)
    u[keep] = pts[keep, 0] / z[keep] * float(intrinsics.fx) + float(intrinsics.cx)
    v[keep] = pts[keep, 1] / z[keep] * float(intrinsics.fy) + float(intrinsics.cy)
    pixels = np.stack([np.round(u), np.round(v)], axis=1).astype(np.int64)
    return pixels, keep


def _tint_mask(panel: np.ndarray, mask: np.ndarray | None, tint_bgr: tuple) -> None:
    """Alpha-blend one class tint into the display panel copy."""
    if mask is None:
        return
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return
    tint = np.asarray(tint_bgr, dtype=np.float32)
    blended = (
        panel[selected].astype(np.float32) * (1.0 - MASK_TINT_ALPHA)
        + tint * MASK_TINT_ALPHA
    )
    panel[selected] = blended.astype(np.uint8)


def render_pair_frame(
    pair: Any,
    *,
    cv2: Any,
    table_c2w: np.ndarray | None,
    fps_snapshot: dict[str, float],
    display_dropped: int,
) -> np.ndarray:
    """Compose one display frame from a published pair (inputs untouched)."""
    pcd_packet = pair.pcd_result.pcd_packet
    processed = pair.pcd_result.processed_frame
    mask_packet = processed.mask_packet
    tracker = pair.tracker_packet

    # Left panel: RGB copy + processed mask tints + visible tracks.
    left = np.ascontiguousarray(mask_packet.color_bgr).copy()
    _tint_mask(left, mask_packet.object_mask, OBJECT_TINT_BGR)
    hand_a = getattr(mask_packet, "hand_a_mask", None)
    hand_b = getattr(mask_packet, "hand_b_mask", None)
    if hand_a is None and hand_b is None:
        _tint_mask(left, mask_packet.controller_mask, HAND_A_TINT_BGR)
    else:
        _tint_mask(left, hand_a, HAND_A_TINT_BGR)
        _tint_mask(left, hand_b, HAND_B_TINT_BGR)
    tracks_yx = np.asarray(tracker.all_tracks_yx, dtype=np.float64)
    track_visible = np.asarray(tracker.all_observation_visibility, dtype=bool)
    track_colors_bgr = np.asarray(tracker.query_rgb_u8)[:, ::-1]
    visible_xy = np.stack(
        [tracks_yx[track_visible, 1], tracks_yx[track_visible, 0]], axis=1
    ).astype(np.int64) if track_visible.any() else np.empty((0, 2), dtype=np.int64)
    _paint_dots(
        left,
        visible_xy,
        np.ascontiguousarray(track_colors_bgr[track_visible]),
        radius=TRACK_DOT_RADIUS,
    )

    # Right panel: world-space products reprojected through the color camera.
    right = np.zeros_like(left)
    world_frames = (TABLE_WORLD_FRAME_KIND,)
    w2c: np.ndarray | None = None
    if str(pcd_packet.coordinate_frame) in world_frames and table_c2w is not None:
        w2c = np.linalg.inv(np.asarray(table_c2w, dtype=np.float64))
    for points, colors_rgb, radius in (
        (pcd_packet.object_xyz_m, pcd_packet.object_colors_rgb_u8, PCD_DOT_RADIUS),
        (
            pcd_packet.controller_xyz_m,
            pcd_packet.controller_colors_rgb_u8,
            PCD_DOT_RADIUS + 1,
        ),
    ):
        pixels, keep = _project_world_points_to_pixels(
            points, intrinsics=pcd_packet.intrinsics, world_to_camera=w2c
        )
        colors_bgr = np.asarray(colors_rgb)[:, ::-1] if np.asarray(
            colors_rgb
        ).size else np.empty((0, 3), dtype=np.uint8)
        _paint_dots(
            right,
            pixels[keep],
            np.ascontiguousarray(colors_bgr[keep]),
            radius=radius,
        )
    prior_points = np.asarray(
        getattr(pcd_packet, "shape_prior_points_m", np.empty((0, 3)))
    )
    if prior_points.size:
        pixels, keep = _project_world_points_to_pixels(
            prior_points, intrinsics=pcd_packet.intrinsics, world_to_camera=w2c
        )
        prior_colors = np.broadcast_to(
            np.asarray(SHAPE_PRIOR_POINT_BGR, dtype=np.uint8), (pixels.shape[0], 3)
        )
        _paint_dots(right, pixels[keep], prior_colors[keep], radius=0)

    # HUD: latency, stage FPS, counts, timings, shape-prior/timeline state.
    timing = pcd_packet.timing
    latency_ms = (
        max(
            float(pcd_packet.process_done_perf_s),
            float(tracker.process_done_perf_s),
        )
        - float(pcd_packet.receive_perf_s)
    ) * 1000.0
    prior_status = str(getattr(pcd_packet, "shape_prior_status", "unknown"))
    # Inverse of the formal gate's predicate (rows gate only while the prior
    # is pending/running; failed is terminal and lifts the gate). The
    # timeout-latch edge (gate lifted while status still "running") is not
    # visible from the packet — display-only approximation.
    timeline = "warming-up" if prior_status in ("pending", "running") else "open"
    fps = {
        key: float(fps_snapshot.get(key, 0.0))
        for key in ("capture_fps", "seg_fps", "pcd_fps", "tracker_fps")
    }
    visible_count = int(track_visible.sum())
    hud_lines = [
        (
            f"seq={int(pair.seq)}  latency={latency_ms:.0f}ms  "
            f"fps cap/seg/pcd/trk="
            f"{fps['capture_fps']:.1f}/{fps['seg_fps']:.1f}/"
            f"{fps['pcd_fps']:.1f}/{fps['tracker_fps']:.1f}  "
            f"display-dropped={int(display_dropped)}"
        ),
        (
            f"obj pts={int(pcd_packet.object_point_count)}  "
            f"ctrl pts={int(pcd_packet.controller_point_count)}  "
            f"tracks visible={visible_count}/{int(tracks_yx.shape[0])}  "
            f"prior pts={int(prior_points.shape[0]) if prior_points.size else 0}"
        ),
        (
            f"mask={float(timing.mask_ms):.0f}ms  pcd={float(timing.pcd_ms):.0f}ms  "
            f"trk model={float(tracker.model_ms):.0f}ms lift="
            f"{float(tracker.lift_ms):.0f}ms  "
            f"shape-prior={prior_status}  timeline={timeline}"
        ),
    ]
    height, width = left.shape[:2]
    hud = np.full((HUD_HEIGHT_PX, width * 2, 3), _HUD_BG_BGR, dtype=np.uint8)
    for row, line in enumerate(hud_lines):
        cv2.putText(
            hud,
            line,
            (12, 26 + row * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            _HUD_TEXT_BGR,
            1,
            cv2.LINE_AA,
        )
    composite = np.vstack([np.hstack([left, right]), hud])
    if DISPLAY_SCALE != 1.0:
        composite = cv2.resize(
            composite,
            (
                int(composite.shape[1] * DISPLAY_SCALE),
                int(composite.shape[0] * DISPLAY_SCALE),
            ),
            interpolation=cv2.INTER_AREA,
        )
    return composite


class LiveDataProcessViewer:
    """Realtime latest-processed-frame window; never raises, never blocks."""

    WINDOW_NAME = "Demo v6.2 - data process (live)"

    def __init__(
        self,
        *,
        pair_slot: Any,
        stage_stats: Any,
        gui: Any,
        stop_event: threading.Event,
        enabled: bool = True,
        table_c2w: Callable[[], np.ndarray | None] | None = None,
        cv2_module: Any | None = None,
    ) -> None:
        self._slot = pair_slot
        self._stage_stats = stage_stats
        self._gui = gui
        self._stop_event = stop_event
        self._close_event = threading.Event()
        self._enabled = bool(enabled)
        self._table_c2w = table_c2w or (lambda: None)
        self._cv2 = cv2_module
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the viewer thread; a GUI-less environment disables it."""
        if not self._enabled or self._thread is not None:
            return
        if self._cv2 is None:
            try:
                import cv2  # noqa: PLC0415

                self._cv2 = cv2
            except Exception as exc:
                print(
                    f"[demo_v6_1] live data-process viewer disabled: {exc}",
                    flush=True,
                )
                return
        self._thread = threading.Thread(
            target=self._run, name="live-dataprocess-viewer", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        """End the viewer (run finished, failed, cancelled, or teardown)."""
        self._close_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._gui.close_window(self.WINDOW_NAME)

    def _run(self) -> None:
        cv2 = self._cv2
        rendered_seq = -1
        announced = False
        try:
            while not self._stop_event.is_set() and not self._close_event.is_set():
                pair = self._slot.get_latest_after(rendered_seq)
                if pair is None:
                    time.sleep(0.02)
                    continue
                frame = render_pair_frame(
                    pair,
                    cv2=cv2,
                    table_c2w=self._table_c2w(),
                    fps_snapshot=self._stage_stats.fps_snapshot(),
                    display_dropped=int(self._slot.dropped_count),
                )
                rendered_seq = int(pair.seq)
                # The GUI loop owns the composed buffer from here on.
                self._gui.submit(self.WINDOW_NAME, frame)
                if not announced:
                    announced = True
                    print(
                        "[demo_v6_1] live data-process viewer: first frame "
                        f"submitted (seq={rendered_seq})",
                        flush=True,
                    )
                # Compose at most ~30 Hz; stale pairs drop in the slot.
                time.sleep(GUI_PUMP_MS / 1000.0)
        except Exception as exc:
            # A rendering failure must never break the pipeline.
            print(
                f"[demo_v6_1] live data-process viewer disabled: {exc}", flush=True
            )
