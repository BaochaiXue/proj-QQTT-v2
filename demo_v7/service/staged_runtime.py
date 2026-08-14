"""Button-driven camera-service state machine for demo_v7.

Re-stages the demo_v6_2 pipeline behind the protocol.py states: PREVIEW
streams frames only, WARMUP runs the frame-0 derived pipeline (no tracking),
REPOSITION overlays the saved frame-0 masks on the live view, and FORMAL
runs the SAME lossless stage set as ``demo_v6_2/mdp/runtime.py`` (built in
``_formal.build_formal_pipeline``; frames produced by
``_acquisition.AcquisitionLoop``).

Threading constraints inherited from the contracts:
- the ControlServer reader thread stays non-blocking: command handlers only
  validate + mutate state; every event/side effect runs via ``_defer`` on the
  main loop, and ``_on_command`` enqueues the ack BEFORE queueing the
  follow-up, so a state event can never overtake the ack that caused it
  (acks and events share the control outbox);
- the frame stream NEVER pauses (fake-live fakes the SOURCE only — a camera
  never stops): frame-0 capture is a snapshot, the reposition overlay rides
  the live frames, and a pre-formal recording wrap notifies the GUI.
- pipeline threads must never block on the GUI: frames go through the
  latest-wins FrameStreamServer with per-channel encode caps;
- a fatal anywhere routes through the shared v6.2 ``FatalErrorLatch``; the
  main loop mirrors the first record into EVT_ERROR + STATE_FATAL and the
  service stays up until CMD_SHUTDOWN.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from demo_v7.runtime.shape_prior import warmup as shape_prior_warmup
from demo_v7.runtime.mdp import warmup as mdp_warmup
from demo_v7.runtime.mdp.cli import RunMode
from demo_v7.runtime.mdp.constants import DEFAULT_LOCAL_FFS_DEPTH_CACHE_FRAMES
from demo_v7.runtime.mdp.live_viewer import (
    HAND_A_TINT_BGR,
    HAND_B_TINT_BGR,
    OBJECT_TINT_BGR,
    render_pair_frame,
)
from demo_v7.runtime.mdp.packets import FramePacket
from demo_v7.runtime.mdp.plumbing import FatalErrorLatch
from demo_v7.runtime.mdp.preload import PerceptionPreloader
from demo_v7.runtime.mdp.session import CameraSession
from demo_v7.runtime.phystwin_strict_product import finalize_headless_capture
from demo_v7.runtime.pipeline_status import (
    STAGE_CAPTURE_START,
    STAGE_SHAPE_PRIOR,
    PipelineStatusWriter,
)
from demo_v7.runtime.utils.concurrency import elapsed_ms as _elapsed_ms
from demo_v7.runtime.utils.render import apply_wslg_open3d_env_defaults

from demo_v7.ipc import protocol
from demo_v7.ipc.channel import ControlServer, FrameStreamServer
from demo_v7.service import (
    backend_options,
    frame0_pipeline,
    gaussian_options,
    shape_prior_backends,
)
from demo_v7.service._acquisition import AcquisitionLoop
from demo_v7.service._formal import (
    FormalPipeline,
    build_formal_pipeline,
    run_early_precompile,
)

# Depth colormap saturation for the CH_DEPTH preview (display only).
DEPTH_PREVIEW_MAX_M = 4.0
# REPOSITION overlay alpha (README: 50% 透明度叠加).
REPOSITION_OVERLAY_ALPHA = 0.5
# CMD_SHUTDOWN during FORMAL drains + finalizes the run first; a drain that
# cannot finish within this bound goes fatal and the service exits anyway.
SHUTDOWN_DRAIN_DEADLINE_S = 120.0
# Shape-prior warmup progress: case-dir milestone files, checked while the
# manager status is still RUNNING (per-stage timing lands only at the end).
_SHAPE_PRIOR_MILESTONES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("upscale", ("shape", "high_resolution.png")),
    ("generate", ("shape", "object.glb")),
    ("align", ("shape", "matching", "final_mesh.glb")),
)
# Shape-prior review artifacts pulled from the case dir (v6.2 layout).
_SHAPE_PRIOR_ARTIFACTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("candidates_npz", ("shape", "candidates.npz")),
    ("mesh_glb", ("shape", "object.glb")),
    ("generate_visualization_mp4", ("shape", "visualization.mp4")),
    ("turntable_mp4", ("shape", "matching", "final_matching.mp4")),
    ("final_mesh_glb", ("shape", "matching", "final_mesh.glb")),
)
_ALIGNMENT_ARTIFACT_NAMES = (
    "mesh_matching.png",
    "render_matching.png",
    "raw_matching.png",
    "raw_matching_valid.png",
    "pnp_results.png",
    "best_match.png",
)


def _ack(cmd: str, *, ok: bool, error: str | None = None, **extra: Any) -> dict:
    """Build one protocol ack object."""
    ack: dict[str, Any] = {"event": protocol.EVT_ACK, "cmd": str(cmd), "ok": bool(ok)}
    if error is not None:
        ack["error"] = str(error)
    ack.update(extra)
    return ack


class StagedRuntime:
    """Camera-service state machine (states/commands/events in protocol.py)."""

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        socket_dir: Path,
        session: CameraSession | None = None,
        channel_max_hz: dict[str, float] | None = None,
        shape_prior_backend: str | None = None,
        shape_prior_use_upscale: bool | None = None,
        gaussian_backend: str | None = None,
        record_dir: Path | None = None,
    ) -> None:
        """Build shared services and bind both protocol sockets.

        ``channel_max_hz`` overrides the per-channel encode caps used by
        ``_publish_frame`` (cross-agent contract: camera_service forwards the
        config value); channels it omits keep the protocol defaults.
        ``shape_prior_backend`` selects the generate backend (sam3d /
        trellis2 / none; camera_service forwards the GUI choice) — it must be
        resolved here in ``__init__`` because the prewarm pool spawns the
        chosen backend's worker before the sockets even bind.
        """
        self.args = args
        # Optional fake-live case recorder (GUI 录制 option): every published
        # RGB-D packet is teed into a data_collect-format case directory via
        # a non-blocking bounded queue (see service/recorder.py).
        self.recorder = None
        if record_dir is not None:
            from demo_v7.service.recorder import (  # noqa: PLC0415
                FakeLiveCaseRecorder,
            )

            self.recorder = FakeLiveCaseRecorder(Path(record_dir))
        self.shape_prior_backend = backend_options.normalize_backend(
            shape_prior_backend
        )
        # 上采样 toggle (GUI selector; camera_service forwards). Off swaps the
        # upscale stage for the crop-only passthrough in _stage_commands.
        self.shape_prior_use_upscale = backend_options.normalize_upscale(
            shape_prior_use_upscale
        )
        # Gaussian generator (GUI selector; camera_service validates and
        # forwards — "none" when unavailable or the shape prior is off).
        self.gaussian_backend = gaussian_options.normalize_gaussian_backend(
            gaussian_backend
        )
        self.mode = RunMode.from_args(args)
        self.session = session if session is not None else CameraSession()
        self.socket_dir = Path(socket_dir)
        self.socket_dir.mkdir(parents=True, exist_ok=True)
        self.stop_event = threading.Event()
        self._shutdown = threading.Event()
        # CMD_SHUTDOWN during FORMAL: exit only after _finalize_formal, with
        # a bounded drain deadline (main loop) as the fatal escape hatch.
        self._shutdown_after_finalize = False
        self._formal_drain_deadline_s: float | None = None
        self._channel_max_hz = dict(protocol.CHANNEL_MAX_HZ)
        if channel_max_hz is not None:
            self._channel_max_hz.update(
                {str(name): float(hz) for name, hz in channel_max_hz.items()}
            )
        self._status = PipelineStatusWriter(
            Path(args.headless_capture_dir).parent, "camera"
        )
        self.fatal = FatalErrorLatch(status=self._status, stop_event=self.stop_event)
        self.shape_prior_manager = self._create_shape_prior_manager()
        self.preload = PerceptionPreloader(args=args, mode=self.mode)
        self._state = protocol.STATE_STARTING
        self._state_lock = threading.Lock()
        self._latest_lock = threading.Lock()
        self._latest_full: FramePacket | None = None
        self._frame0_candidate: FramePacket | None = None
        self._frame0_masks: mdp_warmup.InitialMaskBundle | None = None
        self._warmup_done = False
        self._first_frame_ready = threading.Event()
        self._formal_go = threading.Event()
        self._formal_stop = threading.Event()
        self._formal: FormalPipeline | None = None
        self._formal_finalized = False
        # Persistent seg-host thread: scene-independent EdgeTAM scratch
        # compile during PREVIEW, then the formal seg run on the SAME thread
        # (cudagraph state is thread-affine).
        self._early_precompile_thread: threading.Thread | None = None
        self._edgetam_precompiled = False
        self._formal_seg_start = threading.Event()
        self._fatal_announced = False
        # Gaussian-splats feature (TripoSplat; display-only, fail-soft):
        # worker spawns at PREVIEW (camera GPU; model load hides pre-
        # confirm), generation arms at chain submit, manager dies before
        # FORMAL; the live renderer is created lazily by the formal
        # gaussian worker.
        self._gaussian_manager: Any = None
        self._deferred: deque[Callable[[], None]] = deque()
        self._last_publish_s: dict[str, float] = {}
        self._acq_thread: threading.Thread | None = None
        self._warmup_thread: threading.Thread | None = None
        self._runtime_start_perf_s = time.perf_counter()
        self._acquisition = AcquisitionLoop(
            args=args,
            mode=self.mode,
            session=self.session,
            stop_event=self.stop_event,
            fatal=self.fatal,
            formal_go=self._formal_go,
            formal_stop=self._formal_stop,
            on_replay_wrapped=self._on_replay_wrapped,
            get_formal=lambda: self._formal,
            store_latest=self._store_latest,
            latest_packet=self._latest_packet,
            publish_preview=self._publish_preview_channels,
            publish_frame=self._publish_frame,
        )
        # Handlers return (ack, follow-up or None); _on_command enqueues the
        # ack BEFORE deferring the follow-up (ordering contract).
        self._handlers: dict[
            str, Callable[[dict], tuple[dict, Callable[[], None] | None]]
        ] = {
            protocol.CMD_HELLO: self._cmd_hello,
            protocol.CMD_CAPTURE_FRAME0: self._cmd_capture_frame0,
            protocol.CMD_RETAKE_FRAME0: self._cmd_retake_frame0,
            protocol.CMD_CONFIRM_FRAME0: self._cmd_confirm_frame0,
            protocol.CMD_ENTER_REVIEW: self._cmd_enter_review,
            protocol.CMD_BEGIN_REPOSITION: self._cmd_begin_reposition,
            protocol.CMD_START_FORMAL: self._cmd_start_formal,
            protocol.CMD_STOP_FORMAL: self._cmd_stop_formal,
            protocol.CMD_SHUTDOWN: self._cmd_shutdown,
            protocol.CMD_REGEN_GAUSSIAN: self._cmd_regen_gaussian,
        }
        self.control = ControlServer(
            self.socket_dir / protocol.CONTROL_SOCKET_NAME,
            on_command=self._on_command,
        )
        self.frames = FrameStreamServer(self.socket_dir / protocol.FRAMES_SOCKET_NAME)

    def _create_shape_prior_manager(
        self,
    ) -> shape_prior_warmup.ShapePriorWarmupManager:
        """Create the shape-prior warmup manager (mirrors v6.2 runtime.py).

        The backend selector only swaps WHICH client class builds the stage
        commands (shape_prior_backends); manager semantics, prewarm, and the
        request flow are the unchanged v6.2 machinery. Backend ``none`` keeps
        the manager disabled even if a stray --shape-prior-warmup was
        forwarded (camera_service also normalizes, this is the last belt).
        """
        enabled = bool(self.args.shape_prior_warmup) and (
            self.shape_prior_backend != backend_options.BACKEND_NONE
        )
        client = None
        if enabled:
            client = shape_prior_backends.create_shape_prior_client(
                self.shape_prior_backend,
                use_upscale=self.shape_prior_use_upscale,
                case_root=Path(self.args.shape_prior_case_root),
                cuda_visible_devices=str(
                    self.args.shape_prior_warmup_cuda_visible_devices
                ),
                object_prompt=str(self.args.shape_prior_object_prompt),
                controller_name=str(self.args.shape_prior_controller_name),
                object_id=self.args.shape_prior_object,
                cache_root=self.args.shape_prior_cache_root,
                sam3d_root=self.args.shape_prior_sam3d_root,
                sam3d_config=self.args.shape_prior_config,
                sam31_device=str(self.args.device),
                render_route_visualizations=not bool(
                    self.args.shape_prior_skip_route_visualizations
                ),
            )
            if bool(self.args.shape_prior_prewarm_stage_workers):
                client.prewarm()
        return shape_prior_warmup.ShapePriorWarmupManager(
            enabled=enabled,
            client=client,
            input_source=str(self.args.input_source),
            depth_backend_label=self.mode.depth_backend_label,
            depth_source=str(self.args.depth_source),
            profile_json=self.args.shape_prior_profile_json,
        )

    def _notify_frame0_geometry_from_camera(self) -> None:
        """Forward frame-0 width/height/fx to the align prewarm (v6.2 mirror)."""
        if not bool(getattr(self.args, "shape_prior_warmup", False)):
            return
        runtime = self.session.camera_runtime
        if runtime is None or runtime.k_color is None:
            return
        self.shape_prior_manager.notify_frame0_geometry(
            width=int(self.session.width),
            height=int(self.session.height),
            fx_color=float(np.asarray(runtime.k_color)[0, 0]),
        )

    # ---- state / event plumbing ---------------------------------------------

    @property
    def state(self) -> str:
        """Return the current protocol state."""
        with self._state_lock:
            return self._state

    def _announce_state(self, state: str, detail: str | None = None) -> None:
        """Send the state_changed event for an already-applied transition."""
        event: dict[str, Any] = {"event": protocol.EVT_STATE, "state": str(state)}
        if detail is not None:
            event["detail"] = str(detail)
        self.control.send_event(event)

    def _enter_state(self, state: str, detail: str | None = None) -> None:
        """Apply and announce a service-initiated transition (no ack pending)."""
        with self._state_lock:
            self._state = str(state)
        self._announce_state(state, detail)

    def _emit_progress(
        self,
        stage: str,
        detail: str,
        *,
        ok: bool = True,
        elapsed_ms: float | None = None,
    ) -> None:
        """Send one EVT_PROGRESS event."""
        event: dict[str, Any] = {
            "event": protocol.EVT_PROGRESS,
            "stage": str(stage),
            "detail": str(detail),
            "ok": bool(ok),
        }
        if elapsed_ms is not None:
            event["elapsed_ms"] = float(elapsed_ms)
        self.control.send_event(event)

    def _emit_artifacts(self, kind: str, paths: dict[str, str]) -> None:
        """Send one EVT_ARTIFACTS event."""
        self.control.send_event(
            {"event": protocol.EVT_ARTIFACTS, "kind": str(kind), "paths": dict(paths)}
        )

    def _emit_error(self, where: str, message: str) -> None:
        """Send one EVT_ERROR event (service stays up unless fatal)."""
        self.control.send_event(
            {"event": protocol.EVT_ERROR, "where": str(where), "message": str(message)}
        )

    def _defer(self, fn: Callable[[], None]) -> None:
        """Queue work for the main loop.

        Keeps the control reader free of side effects; _on_command enqueues
        the ack before calling this, so deferred events never overtake it.
        """
        self._deferred.append(fn)

    def _run_deferred(self) -> None:
        """Run queued command follow-ups on the main loop."""
        while self._deferred:
            fn = self._deferred.popleft()
            try:
                fn()
            except Exception as exc:
                if not self.stop_event.is_set():
                    self.fatal.record("deferred command", exc)

    def _announce_fatal_if_needed(self) -> None:
        """Mirror the first fatal record into EVT_ERROR + STATE_FATAL once."""
        fatal = self.fatal.snapshot()
        if fatal is None or self._fatal_announced:
            return
        self._fatal_announced = True
        self._emit_error(fatal.stage, f"{fatal.exc_type}: {fatal.message}")
        self._enter_state(protocol.STATE_FATAL, detail=fatal.stage)

    # ---- frame publishing ---------------------------------------------------

    def _publish_frame(self, channel: str, image_bgr: np.ndarray) -> None:
        """JPEG-encode and publish one frame, capped per channel.

        Caps come from ``self._channel_max_hz`` (protocol defaults, with any
        constructor overrides applied), not the module constant.
        """
        now_s = time.perf_counter()
        min_period_s = 1.0 / float(self._channel_max_hz[channel])
        if now_s - self._last_publish_s.get(channel, 0.0) < min_period_s:
            return
        ok, encoded = cv2.imencode(
            ".jpg",
            image_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(protocol.JPEG_QUALITY)],
        )
        if not ok:
            return
        self.frames.publish(
            channel,
            encoded.tobytes(),
            width=int(image_bgr.shape[1]),
            height=int(image_bgr.shape[0]),
        )
        self._last_publish_s[channel] = now_s

    def _depth_preview_bgr(self, packet: FramePacket) -> np.ndarray | None:
        """Colormap the packet's depth for CH_DEPTH; IR-left fallback on ffs."""
        if packet.depth_u16 is not None:
            depth_m = packet.depth_u16.astype(np.float32) * np.float32(
                packet.depth_scale_m_per_unit
            )
            scaled = np.clip(depth_m / np.float32(DEPTH_PREVIEW_MAX_M), 0.0, 1.0)
            preview = cv2.applyColorMap(
                (scaled * 255.0).astype(np.uint8), cv2.COLORMAP_JET
            )
            preview[packet.depth_u16 == 0] = 0
            return preview
        if packet.ir_left_u8 is not None:
            return cv2.cvtColor(packet.ir_left_u8, cv2.COLOR_GRAY2BGR)
        return None

    def _reposition_overlay_bgr(self, color_bgr: np.ndarray) -> np.ndarray | None:
        """Blend the saved frame-0 masks at 50% alpha over a live frame."""
        masks = self._frame0_masks
        if masks is None:
            return None
        overlay = color_bgr.copy()
        for mask, tint_bgr in (
            (masks.object_mask, OBJECT_TINT_BGR),
            (masks.hand_a_mask, HAND_A_TINT_BGR),
            (masks.hand_b_mask, HAND_B_TINT_BGR),
        ):
            if mask is None:
                continue
            selected = np.asarray(mask, dtype=bool)
            if selected.shape != overlay.shape[:2] or not selected.any():
                continue
            tint = np.asarray(tint_bgr, dtype=np.float32)
            overlay[selected] = (
                overlay[selected].astype(np.float32)
                * (1.0 - REPOSITION_OVERLAY_ALPHA)
                + tint * REPOSITION_OVERLAY_ALPHA
            ).astype(np.uint8)
        return overlay

    def _publish_preview_channels(self, packet: FramePacket) -> None:
        """Publish CH_RGB / CH_DEPTH (and CH_OVERLAY during REPOSITION)."""
        if self.recorder is not None:
            # Single choke point every producer loop passes through (preview,
            # formal raw stream, reposition); submit never blocks.
            self.recorder.submit(packet)
        self._publish_frame(protocol.CH_RGB, packet.color_bgr)
        depth_preview = self._depth_preview_bgr(packet)
        if depth_preview is not None:
            self._publish_frame(protocol.CH_DEPTH, depth_preview)
        if self.state == protocol.STATE_REPOSITION:
            overlay = self._reposition_overlay_bgr(packet.color_bgr)
            if overlay is not None:
                self._publish_frame(protocol.CH_OVERLAY, overlay)

    def _store_latest(self, packet: FramePacket) -> None:
        """Store the newest full packet (frame-0 candidate source)."""
        with self._latest_lock:
            self._latest_full = packet
        self._first_frame_ready.set()

    def _latest_packet(self) -> FramePacket | None:
        """Return the newest full packet."""
        with self._latest_lock:
            return self._latest_full

    # ---- command handlers (ControlServer reader thread; validate only) ------

    def _on_command(self, message: dict) -> dict | None:
        """Dispatch one command; the ack is enqueued before its follow-up.

        Handlers return ``(ack, followup)``. With no follow-up the ack is
        returned to the ControlServer (which enqueues it). Otherwise the ack
        is sent explicitly FIRST and only then is the follow-up queued for
        the main loop, so a state event caused by the command can never
        overtake its ack (channel.py skips sending on a None return).
        """
        cmd = str(message.get("cmd", ""))
        handler = self._handlers.get(cmd)
        if handler is None:
            return _ack(cmd, ok=False, error=f"unknown command: {cmd!r}")
        try:
            ack, followup = handler(message)
        except Exception as exc:
            return _ack(cmd, ok=False, error=f"{type(exc).__name__}: {exc}")
        if followup is None:
            return ack
        self.control.send_event(ack)
        self._defer(followup)
        return None

    def _cmd_hello(
        self, message: dict
    ) -> tuple[dict, Callable[[], None] | None]:
        """hello -> ack with version/state/source_kind/shape_prior_backend."""
        ack = _ack(
            protocol.CMD_HELLO,
            ok=True,
            version=protocol.PROTOCOL_VERSION,
            state=self.state,
            source_kind=str(self.args.input_source),
            shape_prior_backend=self.shape_prior_backend,
            shape_prior_upscale=self.shape_prior_use_upscale,
            gaussian_backend=self.gaussian_backend,
        )
        return ack, None

    def _cmd_capture_frame0(
        self, message: dict
    ) -> tuple[dict, Callable[[], None] | None]:
        """PREVIEW -> FRAME0_PENDING: freeze the latest aligned RGB-D frame."""

        def reject(error: str) -> tuple[dict, None]:
            return _ack(protocol.CMD_CAPTURE_FRAME0, ok=False, error=error), None

        # A camera never stops (fake-live included: only the SOURCE is fake)
        # — capture is a pure snapshot of the newest frame; the stream keeps
        # flowing and the GUI pins the candidate png for the confirm screen.
        with self._state_lock:
            if self._state != protocol.STATE_PREVIEW:
                return reject(
                    f"capture_frame0 requires PREVIEW (state={self._state})"
                )
            packet = self._latest_packet()
            if packet is None:
                return reject("no frame available yet")
            if packet.depth_u16 is None:
                return reject(
                    "frame-0 capture requires native RealSense depth "
                    f"(depth_source={packet.depth_source})"
                )
            self._frame0_candidate = packet
            self._state = protocol.STATE_FRAME0_PENDING
        ack = _ack(protocol.CMD_CAPTURE_FRAME0, ok=True, seq=int(packet.seq))
        return ack, lambda: self._announce_frame0_candidate(packet)

    def _announce_frame0_candidate(self, packet: FramePacket) -> None:
        """FRAME0_PENDING state event, then pin the exact candidate png.

        The saved png + EVT_ARTIFACTS let the GUI show the frame the service
        actually froze (a live camera keeps streaming, so the newest CH_RGB
        frame is NOT the candidate). Display-only: failures degrade to an
        EVT_ERROR, the frozen candidate itself stays valid.
        """
        self._announce_state(protocol.STATE_FRAME0_PENDING)
        try:
            out_dir = self._review_dir() / "frame0"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "frame0_candidate.png"
            if not cv2.imwrite(
                str(path), np.ascontiguousarray(packet.color_bgr, dtype=np.uint8)
            ):
                raise RuntimeError(f"failed to write {path}")
            self._emit_artifacts(
                protocol.ARTIFACT_KIND_FRAME0, {"candidate": str(path)}
            )
        except Exception as exc:
            self._emit_error("frame0_candidate", f"{type(exc).__name__}: {exc}")

    def _cmd_retake_frame0(
        self, message: dict
    ) -> tuple[dict, Callable[[], None] | None]:
        """FRAME0_PENDING -> PREVIEW: discard the candidate (stream never stopped)."""
        with self._state_lock:
            if self._state != protocol.STATE_FRAME0_PENDING:
                ack = _ack(
                    protocol.CMD_RETAKE_FRAME0,
                    ok=False,
                    error=(
                        f"retake_frame0 requires FRAME0_PENDING (state={self._state})"
                    ),
                )
                return ack, None
            self._frame0_candidate = None
            self._state = protocol.STATE_PREVIEW
        ack = _ack(protocol.CMD_RETAKE_FRAME0, ok=True)
        return ack, lambda: self._announce_state(protocol.STATE_PREVIEW)

    def _cmd_confirm_frame0(
        self, message: dict
    ) -> tuple[dict, Callable[[], None] | None]:
        """FRAME0_PENDING -> WARMUP: run the frame-0 pipeline on a worker."""
        with self._state_lock:
            if self._state != protocol.STATE_FRAME0_PENDING:
                ack = _ack(
                    protocol.CMD_CONFIRM_FRAME0,
                    ok=False,
                    error=(
                        f"confirm_frame0 requires FRAME0_PENDING (state={self._state})"
                    ),
                )
                return ack, None
            candidate = self._frame0_candidate
            if candidate is None:
                ack = _ack(
                    protocol.CMD_CONFIRM_FRAME0,
                    ok=False,
                    error="no frame-0 candidate",
                )
                return ack, None
            self._state = protocol.STATE_WARMUP
        ack = _ack(protocol.CMD_CONFIRM_FRAME0, ok=True)
        return ack, lambda: self._begin_warmup(candidate)

    def _cmd_enter_review(
        self, message: dict
    ) -> tuple[dict, Callable[[], None] | None]:
        """WARMUP(done) -> REVIEW; a no-op because the service auto-enters."""
        state = self.state
        if state == protocol.STATE_REVIEW:
            return _ack(protocol.CMD_ENTER_REVIEW, ok=True), None
        ack = _ack(
            protocol.CMD_ENTER_REVIEW,
            ok=False,
            error=f"warmup not finished (state={state})",
        )
        return ack, None

    def _cmd_begin_reposition(
        self, message: dict
    ) -> tuple[dict, Callable[[], None] | None]:
        """REVIEW -> REPOSITION: overlay the saved masks on the LIVE stream.

        The stream (real or fake) never stops — the overlay rides the live
        frames so the operator can judge whether object/hands match the
        frame-0 positions in real time.
        """
        with self._state_lock:
            if self._state != protocol.STATE_REVIEW:
                ack = _ack(
                    protocol.CMD_BEGIN_REPOSITION,
                    ok=False,
                    error=f"begin_reposition requires REVIEW (state={self._state})",
                )
                return ack, None
            if self._frame0_masks is None:
                ack = _ack(
                    protocol.CMD_BEGIN_REPOSITION,
                    ok=False,
                    error="frame-0 masks are not available",
                )
                return ack, None
            self._state = protocol.STATE_REPOSITION
        ack = _ack(protocol.CMD_BEGIN_REPOSITION, ok=True)
        return ack, lambda: self._announce_state(protocol.STATE_REPOSITION)

    def _cmd_start_formal(
        self, message: dict
    ) -> tuple[dict, Callable[[], None] | None]:
        """REPOSITION -> FORMAL: construct + start the v6.2 lossless stages."""
        with self._state_lock:
            if self._state != protocol.STATE_REPOSITION:
                ack = _ack(
                    protocol.CMD_START_FORMAL,
                    ok=False,
                    error=f"start_formal requires REPOSITION (state={self._state})",
                )
                return ack, None
            if not self.mode.lossless_enabled:
                ack = _ack(
                    protocol.CMD_START_FORMAL,
                    ok=False,
                    error="formal stage requires the lossless masked-PCD mode",
                )
                return ack, None
            if self._frame0_masks is None:
                ack = _ack(
                    protocol.CMD_START_FORMAL,
                    ok=False,
                    error="frame-0 masks are not available",
                )
                return ack, None
            self._state = protocol.STATE_FORMAL
        return _ack(protocol.CMD_START_FORMAL, ok=True), self._launch_formal

    def _cmd_regen_gaussian(
        self, message: dict
    ) -> tuple[dict, Callable[[], None] | None]:
        """REVIEW 拣选/换seed: re-roll the TripoSplat generation."""
        manager = self._gaussian_manager
        if self.state != protocol.STATE_REVIEW:
            ack = _ack(
                protocol.CMD_REGEN_GAUSSIAN,
                ok=False,
                error=f"regen_gaussian requires REVIEW (state={self.state})",
            )
            return ack, None
        if manager is None:
            ack = _ack(
                protocol.CMD_REGEN_GAUSSIAN,
                ok=False,
                error="gaussian feature is not active in this run",
            )
            return ack, None
        if manager.busy:
            ack = _ack(
                protocol.CMD_REGEN_GAUSSIAN,
                ok=False,
                error="a gaussian generation is already in flight",
            )
            return ack, None
        seed = message.get("seed")
        seed = int(seed) if seed is not None else None
        ack = _ack(protocol.CMD_REGEN_GAUSSIAN, ok=True)
        return ack, lambda: manager.regenerate(seed)

    def _cmd_stop_formal(
        self, message: dict
    ) -> tuple[dict, Callable[[], None] | None]:
        """FORMAL -> FINISHED: stop the producer; FINISHED lands after drain."""
        if self.state != protocol.STATE_FORMAL:
            ack = _ack(
                protocol.CMD_STOP_FORMAL,
                ok=False,
                error=f"stop_formal requires FORMAL (state={self.state})",
            )
            return ack, None
        self._formal_stop.set()
        return _ack(protocol.CMD_STOP_FORMAL, ok=True), None

    def _cmd_shutdown(
        self, message: dict
    ) -> tuple[dict, Callable[[], None] | None]:
        """any -> exit(0); an in-flight FORMAL drains + finalizes first."""
        return _ack(protocol.CMD_SHUTDOWN, ok=True), self._begin_shutdown

    def _begin_shutdown(self) -> None:
        """Deferred CMD_SHUTDOWN: never skip an active formal run's finalize.

        During FORMAL this mirrors stop_formal — stop the producer, let the
        main loop run ``_finalize_formal`` after the drain (frames.jsonl
        gate check + finalize_headless_capture), THEN latch ``_shutdown`` —
        under a bounded drain deadline that goes fatal on expiry. Outside
        FORMAL (or after finalize/fatal) exit is immediate; ``_teardown``
        stays the last-resort cleanup path.
        """
        if (
            self._formal is not None
            and not self._formal_finalized
            and self.fatal.snapshot() is None
        ):
            self._shutdown_after_finalize = True
            self._formal_drain_deadline_s = (
                time.perf_counter() + SHUTDOWN_DRAIN_DEADLINE_S
            )
            self._formal_stop.set()
            return
        self._shutdown.set()

    # ---- WARMUP (frame-0 derived pipeline, no tracking) ----------------------

    def _begin_warmup(self, candidate: FramePacket) -> None:
        """Announce WARMUP and start the frame-0 pipeline worker."""
        self._announce_state(protocol.STATE_WARMUP)
        self._warmup_thread = threading.Thread(
            target=self._warmup_worker,
            args=(candidate,),
            name="demo-v7-warmup",
            daemon=True,
        )
        self._warmup_thread.start()

    def _review_dir(self) -> Path:
        """Run output dir for frame-0 review artifacts (parent of capture dir)."""
        return Path(self.args.headless_capture_dir).parent

    def _shape_prior_case_dir(self) -> Path | None:
        """Return the shape-prior case dir (v6.2 layout), if configured."""
        client = self.shape_prior_manager.client
        if client is None:
            return None
        return Path(client.case_root) / str(client.case_name)

    def _warmup_worker(self, candidate: FramePacket) -> None:
        """compute_sam31_masks -> review artifacts -> shape prior -> REVIEW."""
        manager = self.shape_prior_manager
        perception_profile: dict[str, Any] = {}
        try:
            sam31_start_s = time.perf_counter()
            sam31_preload = self.preload.join_sam31()
            if sam31_preload:
                # The decisive number for a slow sam31 row: how long the
                # 3.5G checkpoint preload made THIS confirm wait (compute
                # itself is ~0.5s warm). Rides the run log.
                print(
                    "[v7-timing] sam31 preload "
                    + " ".join(
                        f"{k}={v:.0f}ms" for k, v in sam31_preload.items()
                    ),
                    flush=True,
                )
            object_mask, hand_a_mask, hand_b_mask = (
                frame0_pipeline.compute_sam31_masks(
                    candidate.color_bgr,
                    device=str(self.args.device),
                    args=self.args,
                    mode=self.mode,
                    reuse_sam31_runtime=manager.requires_sam31_reuse,
                    profile_out=perception_profile,
                )
            )
            self._emit_progress(
                "sam31_masks",
                "frame-0 SAM3.1 masks ready",
                elapsed_ms=_elapsed_ms(sam31_start_s),
            )
            bundle = frame0_pipeline.Frame0Bundle(
                color_bgr=candidate.color_bgr,
                depth_u16=candidate.depth_u16,
                intrinsics=candidate.intrinsics,
                depth_scale_m_per_unit=float(candidate.depth_scale_m_per_unit),
                object_mask=object_mask,
                hand_a_mask=hand_a_mask,
                hand_b_mask=hand_b_mask,
                source_timestamp_s=candidate.source_timestamp_s,
                source_frame_index=candidate.source_frame_index,
                source_step=candidate.source_step,
            )
            artifacts = frame0_pipeline.save_review_artifacts(
                self._review_dir(), bundle
            )
            for kind, paths in artifacts.items():
                self._emit_artifacts(kind, paths)
            self._frame0_masks = mdp_warmup.InitialMaskBundle(
                controller_mask=np.ascontiguousarray(
                    np.asarray(hand_a_mask, dtype=bool)
                    | np.asarray(hand_b_mask, dtype=bool)
                ),
                object_mask=object_mask,
                hand_a_mask=hand_a_mask,
                hand_b_mask=hand_b_mask,
            )
        except Exception as exc:
            # Recoverable: nothing irreversible submitted yet (e.g. the
            # two-separable-hands gate) — surface the error and let the
            # operator retake frame 0 from PREVIEW.
            if self.stop_event.is_set():
                return
            print(
                f"[demo-v7] frame-0 warmup failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            self._emit_error("warmup", f"{type(exc).__name__}: {exc}")
            self._emit_progress("sam31_masks", str(exc), ok=False)
            self._frame0_candidate = None
            self._enter_state(
                protocol.STATE_PREVIEW, detail="warmup failed; retake frame 0"
            )
            return
        if not manager.enabled:
            # Backend "none" (or an explicit --no-shape-prior-warmup): skip
            # generation/align/sample entirely but keep the review data the
            # GUI can honestly show — the OBSERVED frame-0 object points are
            # the whole tracking structure in this mode. Best-effort: this
            # branch has consumed nothing one-shot, so failures stay
            # display-only and never block the REVIEW transition.
            try:
                processed = frame0_pipeline.build_frame0_processed(
                    bundle, args=self.args, session=self.session
                )
                self._emit_observed_points(processed)
            except Exception as exc:
                print(
                    f"[v7] observed-points review data skipped: {exc}", flush=True
                )
                self._emit_error(
                    "observed_points", f"{type(exc).__name__}: {exc}"
                )
            self._emit_progress(
                "shape_prior_submit",
                "shape prior disabled(backend=none);跳过生成/对齐/补点",
            )
            self._emit_progress("shape_prior", "skipped(无 shape prior)")
            # shape_prior_ready is the GUI's 查看结果 gate; emitting it here
            # keeps the button flow identical across backends.
            self._emit_progress("shape_prior_ready", "shape prior 已跳过")
            self._warmup_done = True
            self._enter_state(protocol.STATE_REVIEW)
            return
        try:
            processed = frame0_pipeline.build_frame0_processed(
                bundle, args=self.args, session=self.session
            )
            self._emit_observed_points(processed)
            submit_s = time.perf_counter()
            frame0_pipeline.submit_shape_prior(
                manager,
                processed,
                args=self.args,
                session=self.session,
                warmup_start_perf_s=self._runtime_start_perf_s,
                perception_profile=perception_profile,
            )
            self._status.emit(
                STAGE_SHAPE_PRIOR, "frame-0 submitted; generating shape prior"
            )
            self._emit_progress(
                "shape_prior_submit", "frame-0 shape-prior request submitted"
            )
            # Gaussian generation rides IN PARALLEL with the chain (worker
            # already spawned at PREVIEW): arm the first generate — it fires
            # as soon as THIS run's segment writes masked_image.png (the
            # same image the mesh generator conditions on; mtime-gated
            # against stale files). Alignment parks until
            # notify_case_ready() below. Display-only, fail-soft.
            self._start_gaussian_manager()  # no-op if PREVIEW spawned it
            if self._gaussian_manager is not None:
                self._gaussian_manager.notify_submitted()
            self._poll_shape_prior_until_terminal()
            self._emit_progress(
                "shape_prior_ready",
                "shape prior ready",
                elapsed_ms=_elapsed_ms(submit_s),
            )
            self._emit_shape_prior_artifacts()
            # No STAGE_WARMUP_READY here: the untouched v6.2 ShapePriorPublisher
            # is the single warmup_ready emitter (formal timeline open); the
            # GUI already got the shape_prior_ready EVT_PROGRESS above.
            self._warmup_done = True
            self._enter_state(protocol.STATE_REVIEW)
            # Chain READY: alignment inputs (best_match.pkl, final_mesh.glb)
            # are settled on disk — release the parked gaussian alignment.
            if self._gaussian_manager is not None:
                self._gaussian_manager.notify_case_ready()
        except Exception as exc:
            # Irrecoverable: the one-shot ShapePriorWarmupManager cannot accept
            # a second frame-0 request in-process. The failed progress event
            # settles the GUI timeline (✗ + stopped spinners) before the
            # fatal path takes over the status band.
            if not self.stop_event.is_set():
                self._emit_progress(
                    "shape_prior", f"{type(exc).__name__}: {exc}", ok=False
                )
                self.fatal.record("frame-0 warmup", exc)

    def _emit_observed_points(self, processed: Any) -> None:
        """补点 review data: save + announce the frame-0 OBSERVED object points.

        Lets the GUI contrast the observed cloud with the shape-prior
        surface/interior fill (candidates.npz ships via the shape-prior
        artifact list) — or show it alone under backend "none".
        Display-only; a write failure never blocks the warmup.
        """
        try:
            observed_path = (
                self._review_dir() / "frame0" / "frame0_object_points.npz"
            )
            np.savez_compressed(
                observed_path,
                object_xyz_m=np.asarray(
                    processed.pcd_packet.object_xyz_m, dtype=np.float32
                ),
                object_colors_rgb_u8=np.asarray(
                    processed.pcd_packet.object_colors_rgb_u8, dtype=np.uint8
                ),
            )
            self._emit_artifacts(
                protocol.ARTIFACT_KIND_FRAME0,
                {"object_points_npz": str(observed_path)},
            )
        except Exception as exc:
            print(f"[v7] frame0 object-points artifact skipped: {exc}", flush=True)

    def _start_gaussian_manager(self) -> None:
        """Spawn the TripoSplat manager (fail-soft; GUI selector gates it).

        GPU policy (contention): the worker inherits the service env — the
        CAMERA GPU — so generation never competes with the mesh backend on
        the shape-prior GPU (SAM3D peaks at ~18G there; sharing OOMed).
        DEMO_V7_GAUSSIAN_CUDA_VISIBLE_DEVICES overrides the placement;
        DEMO_V7_GAUSSIAN_SPLATS=0 stays as the emergency kill switch.
        """
        if self.gaussian_backend == gaussian_options.GAUSSIAN_NONE:
            return
        if os.environ.get("DEMO_V7_GAUSSIAN_SPLATS", "1") == "0":
            return
        if self._gaussian_manager is not None:
            return
        try:
            case_dir = self._shape_prior_case_dir()
            if case_dir is None:
                return
            emit_progress = lambda stage, detail="", **kw: self._emit_progress(  # noqa: E731
                stage, detail, **kw
            )
            if self.gaussian_backend == gaussian_options.GAUSSIAN_MESH_SURFACE:
                from demo_v7.service.mesh_surface_manager import (
                    MeshSurfaceGaussianManager,
                )

                manager = MeshSurfaceGaussianManager(
                    case_dir=case_dir,
                    out_dir=self._review_dir() / "gaussian",
                    emit_progress=emit_progress,
                    emit_artifacts=self._emit_artifacts,
                    emit_error=self._emit_error,
                )
            else:
                from demo_v7.service.gaussian_manager import GaussianManager

                manager = GaussianManager(
                    case_dir=case_dir,
                    out_dir=self._review_dir() / "gaussian",
                    controller_name=str(self.args.shape_prior_controller_name),
                    cuda_visible_devices=os.environ.get(
                        "DEMO_V7_GAUSSIAN_CUDA_VISIBLE_DEVICES"
                    ),
                    emit_progress=emit_progress,
                    emit_artifacts=self._emit_artifacts,
                    emit_error=self._emit_error,
                )
            manager.start()
            self._gaussian_manager = manager
        except Exception as exc:
            self._emit_error("gaussian", f"{type(exc).__name__}: {exc}")

    def _shutdown_gaussian_manager(self) -> None:
        """Free the camera GPU before FORMAL's perception stack (idempotent)."""
        manager = self._gaussian_manager
        if manager is None:
            return
        try:
            manager.shutdown()
        except Exception:
            pass

    def _gaussian_worker(self) -> None:
        """FORMAL: deform the aligned splats by tracked object motion and
        publish CH_GAUSSIAN (pure observer of live_viz_slot; own cursor)."""
        formal = self._formal
        manager = self._gaussian_manager
        if formal is None or manager is None or not manager.has_world_ply():
            return
        try:
            from demo_v7.runtime.mdp.constants import TABLE_WORLD_FRAME_KIND
            from demo_v7.service.gaussian_live import (
                GaussianLiveRenderer,
                MeshAnchoredGaussianRenderer,
            )

            anchors_path = getattr(manager, "anchors_path", None)
            if anchors_path is not None:
                # mesh_surface backend: splats ride the mesh vertices via
                # their barycentric anchors (hard binding stays in motion).
                live = MeshAnchoredGaussianRenderer(
                    str(manager.world_ply_path), str(anchors_path)
                )
            else:
                live = GaussianLiveRenderer(str(manager.world_ply_path))
        except Exception as exc:
            print(f"[gaussian-live] init failed: {exc}", flush=True)
            return
        table_c2w = self.session.table_c2w
        if table_c2w is None:
            return
        viewmat = np.linalg.inv(np.asarray(table_c2w, dtype=np.float64))
        self._gaussian_formal_catchup(live)
        try:
            background_whiten = float(
                os.environ.get("DEMO_V7_GAUSSIAN_BG_WHITEN", "0.65")
            )
        except ValueError:
            background_whiten = 0.65
        slot = formal.live_viz_slot
        rendered_seq = -1
        min_interval_s = 1.0 / self._channel_max_hz[protocol.CH_GAUSSIAN]
        last_render_s = 0.0
        stats_interval_s = 5.0
        last_stats_s = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                if formal.lossless.processing_done.is_set() or live.failed:
                    return
                pair = slot.get_latest_after(rendered_seq)
                if pair is None:
                    time.sleep(0.02)
                    continue
                rendered_seq = int(pair.seq)
                tracker = pair.tracker_packet
                if str(tracker.coordinate_frame) != str(TABLE_WORLD_FRAME_KIND):
                    return  # uncalibrated run: world-frame gaussians undefined
                live.step(
                    tracker.marker_xyz_m,
                    tracker.query_indices,
                    tracker.query_is_object,
                )
                now_s = time.perf_counter()
                if now_s - last_render_s < min_interval_s:
                    continue
                last_render_s = now_s
                mask_packet = pair.pcd_result.processed_frame.mask_packet
                intr = mask_packet.intrinsics  # CameraIntrinsics dataclass -> K
                frame = live.render_over(
                    mask_packet.color_bgr,
                    viewmat=viewmat,
                    intrinsics=np.array(
                        [
                            [float(intr.fx), 0.0, float(intr.cx)],
                            [0.0, float(intr.fy), float(intr.cy)],
                            [0.0, 0.0, 1.0],
                        ]
                    ),
                    background_whiten=background_whiten,
                )
                if frame is not None:
                    self._publish_frame(protocol.CH_GAUSSIAN, frame)
                if now_s - last_stats_s >= stats_interval_s:
                    last_stats_s = now_s
                    self._write_gaussian_live_stats(live)
        finally:
            # A FORMAL shorter than the stats interval (2-chunk E2E runs)
            # or a mid-run deform failure must still leave an honest final
            # snapshot — the json carries failed/frames_stepped so a stale
            # healthy-looking file cannot mask a dead channel.
            self._write_gaussian_live_stats(live)

    def _gaussian_formal_catchup(self, live) -> None:
        """Close the capture-frame-0 -> FORMAL seq-0 gap before the loop.

        The world splats are registered to the CAPTURE frame-0 pose while
        the tracker's bones are seeded on FORMAL seq 0 (minutes later);
        without this the gaussian starts — and forever stays — offset by
        whatever the object did in between. Two corrections, both from
        prepared_phystwin/000000.npz (written with the first strict pair):
        rigid ICP of the splats onto the seq-0 object cloud, then seq-0
        rest positions for the bones so the first live packet becomes a
        one-shot catch-up pose instead of a new rest pose. Every failure path
        degrades to the old behavior — never fatal.
        """
        try:
            from demo_v7.service.gaussian_align import rigid_world_catchup
            from demo_v7.service.gaussian_live import (
                load_formal_frame0_rest_positions,
            )

            writer = self.session.headless_capture_writer
            if writer is None:
                print(
                    "[gaussian-live] no headless capture writer; skipping "
                    "formal seq-0 catch-up",
                    flush=True,
                )
                return
            npz_path = Path(writer.prepared_phystwin_dir) / "000000.npz"
            # This worker spawns BEFORE _formal_go releases the producer
            # (frame-0 readiness alone is budgeted up to 120s), so any
            # fixed deadline from spawn would silently skip the catch-up
            # on a legitimately slow start. Wait for the first strict pair
            # instead — it has no bounded wall budget of its own — and
            # only then bound the npz poll: first_pair_published fires
            # BEFORE the npz write lands, so poll the file itself (atomic
            # write -> exists == complete).
            first_pair = self._formal.lossless.first_pair_published
            while not first_pair.wait(timeout=0.5):
                if self.stop_event.is_set():
                    return
            deadline_s = time.perf_counter() + 30.0
            while not npz_path.is_file():
                if self.stop_event.is_set() or time.perf_counter() > deadline_s:
                    print(
                        "[gaussian-live] formal seq-0 prepared frame not "
                        "available; skipping catch-up",
                        flush=True,
                    )
                    return
                time.sleep(0.2)
            rest, object_cloud = load_formal_frame0_rest_positions(npz_path)
            transform, info = rigid_world_catchup(
                live._tensors["means"].detach().cpu().numpy(),
                live._tensors["opacities"].detach().cpu().numpy(),
                object_cloud,
            )
            if transform is not None:
                live.apply_rigid_transform(transform)
            print(
                f"[gaussian-live] formal seq-0 catch-up: rest_bones={len(rest)} "
                f"rigid={info}",
                flush=True,
            )
            live.seed_rest_positions(rest)
        except Exception as exc:
            print(
                f"[gaussian-live] formal seq-0 catch-up skipped "
                f"({type(exc).__name__}: {exc})",
                flush=True,
            )

    def _write_gaussian_live_stats(self, live) -> None:
        """Periodic follow telemetry: one stdout line + an atomic json.

        The json is the E2E driver's assertion surface (service stdout is
        not capturable there); the stdout line is for humans tailing a run.
        """
        try:
            stats = live.follow_stats()
            if stats is None:
                return
            print(
                "[gaussian-live] follow: "
                + " ".join(f"{key}={value}" for key, value in stats.items()),
                flush=True,
            )
            manager = self._gaussian_manager
            if manager is None:
                return
            stats_path = manager.out_dir / "gaussian_live_stats.json"
            tmp_path = stats_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(stats, indent=1))
            os.replace(tmp_path, stats_path)
        except Exception as exc:
            print(f"[gaussian-live] stats skipped: {exc}", flush=True)

    def _poll_shape_prior_until_terminal(self) -> None:
        """Poll the manager profile until READY; raise on FAILED/stop.

        Logs a one-line milestone timing summary at READY (owner ask:
        warmup timing in the run log to steer optimization) — seconds
        since submit for each stage-milestone file plus the ready total.
        """
        manager = self.shape_prior_manager
        case_dir = self._shape_prior_case_dir()
        poll_start_s = time.perf_counter()
        seen: dict[str, float] = {}
        gaussian_release_sent = False
        while not self.stop_event.is_set():
            profile = manager.profile()
            status = str(
                profile.get("shape_prior_status", shape_prior_warmup.STATUS_DISABLED)
            )
            if case_dir is not None:
                for stage, parts in _SHAPE_PRIOR_MILESTONES:
                    if stage in seen:
                        continue
                    if case_dir.joinpath(*parts).is_file():
                        seen[stage] = time.perf_counter() - poll_start_s
                        self._emit_progress("shape_prior", f"{stage} finished")
                # Gaussian alignment needs only the ALIGN stage's settled
                # outputs (best_match.pkl + final_mesh.glb) — release it at
                # the align profile's COMPLETED (written after every export),
                # not at chain READY: the splats land ~7s earlier. The
                # sample stage's concurrent final_mesh cleanup is an atomic
                # os.replace (readers get old-or-new, both valid).
                if not gaussian_release_sent and self._gaussian_manager is not None:
                    align_profile = case_dir / "shape" / "timing" / "align.json"
                    if align_profile.is_file():
                        try:
                            align_status = json.loads(
                                align_profile.read_text()
                            ).get("status")
                        except (OSError, json.JSONDecodeError):
                            align_status = None
                        if align_status == "completed":
                            gaussian_release_sent = True
                            self._gaussian_manager.notify_case_ready()
            if status in (
                shape_prior_warmup.STATUS_READY,
                shape_prior_warmup.STATUS_DISABLED,
            ):
                milestones = " ".join(
                    f"{stage}={elapsed:.1f}s" for stage, elapsed in seen.items()
                )
                print(
                    f"[v7-timing] chain since submit: {milestones} "
                    f"ready={time.perf_counter() - poll_start_s:.1f}s",
                    flush=True,
                )
                return
            if status == shape_prior_warmup.STATUS_FAILED:
                raise RuntimeError(
                    f"shape prior failed: {profile.get('shape_prior_error')}"
                )
            time.sleep(0.1)
        raise RuntimeError("service stopped during shape-prior warmup")

    def _emit_shape_prior_artifacts(self) -> None:
        """EVT_ARTIFACTS for shape-prior + alignment files in the case dir."""
        case_dir: Path | None = None
        result = self.shape_prior_manager.ready_result()
        if result is not None:
            metadata_dir = result.metadata.get("shape_prior_case_dir")
            if metadata_dir:
                case_dir = Path(str(metadata_dir))
        if case_dir is None:
            case_dir = self._shape_prior_case_dir()
        if case_dir is None:
            return
        prior_paths: dict[str, str] = {}
        for name, parts in _SHAPE_PRIOR_ARTIFACTS:
            path = case_dir.joinpath(*parts)
            if path.is_file():
                prior_paths[name] = str(path)
        if prior_paths:
            self._emit_artifacts(protocol.ARTIFACT_KIND_SHAPE_PRIOR, prior_paths)
        alignment_paths: dict[str, str] = {}
        matching_dir = case_dir / "shape" / "matching"
        for filename in _ALIGNMENT_ARTIFACT_NAMES:
            path = matching_dir / filename
            if path.is_file():
                alignment_paths[filename.rsplit(".", 1)[0]] = str(path)
        if alignment_paths:
            self._emit_artifacts(protocol.ARTIFACT_KIND_ALIGNMENT, alignment_paths)

    # ---- FORMAL ---------------------------------------------------------------

    def _spawn_worker(self, name: str, target: Callable[[], None]) -> threading.Thread:
        """Start one daemon worker routed through the shared fatal latch."""

        def run_worker() -> None:
            """Run worker."""
            try:
                target()
            except BaseException as exc:
                # BaseException mirrors v6.2: preload legs re-raise SystemExit
                # at their joins and only fatal.record sets stop_event.
                if not self.stop_event.is_set():
                    self.fatal.record(f"{name} worker", exc)

        thread = threading.Thread(target=run_worker, name=f"demo-v7-{name}", daemon=True)
        thread.start()
        return thread

    def _on_replay_wrapped(self) -> None:
        """Pre-formal fake replay wrap: notify the GUI, keep streaming.

        Called from the acquisition thread — send_event is thread-safe and
        non-blocking. wrapped=True distinguishes the informational pre-formal
        wrap (recording restarted from the top; a camera never stops) from
        the FORMAL-ending exhaustion (wrapped=False).
        """
        self.control.send_event(
            {
                "event": protocol.EVT_REPLAY_EXHAUSTED,
                "wrapped": True,
                "state": self.state,
            }
        )

    def _spawn_early_precompile(self) -> None:
        """Start the persistent seg-host thread the moment PREVIEW opens.

        torch.compile 'reduce-overhead' cudagraph state is thread-affine
        (empirically: precompiling on one thread and running the formal
        forwards on another dies with a bare AssertionError inside inductor's
        cudagraph trees), and in v6.2 the scratch precompile and every real
        forward share the seg worker thread. So v7 dedicates ONE host thread:
        it runs the scene-independent scratch compile during PREVIEW, then
        parks until start_formal hands it the formal SegmentationStage, whose
        run() executes on this same thread — identical thread semantics to
        v6.2. Precompile failure is best-effort (formal pays it inline).
        """
        if self._early_precompile_thread is not None:
            return
        if self.args.track_mode == "none":
            return

        def seg_host() -> None:
            try:
                edgetam = self.preload.join_edgetam()
                run_early_precompile(
                    edgetam,
                    args=self.args,
                    mode=self.mode,
                    session=self.session,
                )
                self._edgetam_precompiled = True
            except Exception as exc:
                print(f"[v7] early edgetam precompile skipped: {exc}", flush=True)
            while not self.stop_event.is_set() and not self._shutdown.is_set():
                if self._formal_seg_start.wait(timeout=0.1):
                    break
            formal = self._formal
            if (
                formal is None
                or self.stop_event.is_set()
                or self._shutdown.is_set()
            ):
                return
            try:
                formal.seg.run()
            except Exception as exc:  # same latch route as _spawn_worker
                if not self.stop_event.is_set():
                    self.fatal.record("segmentation worker", exc)

        self._early_precompile_thread = threading.Thread(
            target=seg_host, name="seg-host", daemon=True
        )
        self._early_precompile_thread.start()

    def _launch_formal(self) -> None:
        """Deferred start_formal: build stages, start workers, release capture."""
        masks = self._frame0_masks
        if masks is None:
            raise RuntimeError("formal stage requires saved frame-0 masks")
        # The TripoSplat worker's job is done (world ply on disk); its GPU
        # goes to the PhysTwin children now.
        self._shutdown_gaussian_manager()
        formal = build_formal_pipeline(
            args=self.args,
            mode=self.mode,
            session=self.session,
            preload=self.preload,
            shape_prior_manager=self.shape_prior_manager,
            status=self._status,
            saved_masks=masks,
            warmup_runtime_start_perf_s=self._runtime_start_perf_s,
            stop_event=self.stop_event,
            fatal=self.fatal,
            skip_precompile=self._edgetam_precompiled,
        )
        self._formal = formal
        workers: list[tuple[str, Callable[[], None]]] = [
            ("processed-frame", formal.product.processed_frame_worker),
            ("tracker", formal.tracker.run_lossless),
            ("pair-output", formal.product.pair_output_worker),
            ("composite", self._composite_worker),
            ("gaussian", self._gaussian_worker),
        ]
        formal.threads = [self._spawn_worker(name, target) for name, target in workers]
        # The seg stage runs on the persistent seg-host thread (cudagraph
        # state is thread-affine; the host already holds the scratch compile).
        seg_host = self._early_precompile_thread
        if seg_host is None:
            # track_mode none never reaches formal; a missing host here means
            # PREVIEW was never entered — treat as fatal wiring.
            self.fatal.record(
                "formal readiness",
                RuntimeError("seg host thread was never started"),
            )
            return
        formal.threads.append(seg_host)
        self._formal_seg_start.set()
        # v6.2 readiness-barrier equivalent: the lossless producer starts (and
        # fake replay resumes) only once the seg worker reports it can consume
        # the formal frame 0 immediately — otherwise every frame captured
        # while the EdgeTAM session seeds would be dropped and chunk 0 would
        # span a discontinuous slice of recording time.
        deadline_s = time.perf_counter() + 120.0
        while not self.stop_event.is_set() and self.fatal.snapshot() is None:
            if self.preload.wait_frame0_consumers_ready(0.2):
                break
            if time.perf_counter() > deadline_s:
                self.fatal.record(
                    "formal readiness",
                    RuntimeError("formal seg did not become frame-0 ready"),
                )
                return
        self._formal_go.set()
        self._announce_state(protocol.STATE_FORMAL)

    def _composite_worker(self) -> None:
        """Render v6.2 pair composites to CH_COMPOSITE + EVT_FORMAL_STATS."""
        formal = self._formal
        if formal is None:
            return
        slot = formal.live_viz_slot
        rendered_seq = -1
        last_stats_s = 0.0
        while not self.stop_event.is_set():
            if formal.lossless.processing_done.is_set():
                return
            pair = slot.get_latest_after(rendered_seq)
            if pair is None:
                time.sleep(0.02)
                continue
            rendered_seq = int(pair.seq)
            frame = render_pair_frame(
                pair,
                cv2=cv2,
                table_c2w=self.session.table_c2w,
                fps_snapshot=formal.stage_stats.fps_snapshot(),
                display_dropped=int(slot.dropped_count),
            )
            self._publish_frame(protocol.CH_COMPOSITE, frame)
            now_s = time.perf_counter()
            if now_s - last_stats_s >= 1.0:
                last_stats_s = now_s
                pcd_packet = pair.pcd_result.pcd_packet
                latency_ms = (
                    max(
                        float(pcd_packet.process_done_perf_s),
                        float(pair.tracker_packet.process_done_perf_s),
                    )
                    - float(pcd_packet.receive_perf_s)
                ) * 1000.0
                self.control.send_event(
                    {
                        "event": protocol.EVT_FORMAL_STATS,
                        "seq": rendered_seq,
                        "fps": formal.stage_stats.fps_snapshot(),
                        "latency_ms": latency_ms,
                    }
                )
            # Compose at most ~30 Hz; stale pairs drop in the slot.
            time.sleep(0.02)

    # ---- lifecycle ------------------------------------------------------------

    def _start_service(self) -> None:
        """STARTING work: source, calibration, preloads, acquisition thread."""
        self._status.emit(STAGE_CAPTURE_START, f"input={self.args.input_source}")
        apply_wslg_open3d_env_defaults()
        if self.args.depth_source == "ffs":
            # Lazy: the FFS/TensorRT/numba import chain must not tax the
            # default native-realsense startup path (v6.2 runtime mirror).
            from demo_v7.runtime.utils.ffs_align import (  # noqa: PLC0415
                FfsDepthEngine,
                warm_up_numba_ffs_align,
            )

            self.session.depth_engine = FfsDepthEngine(
                ffs_repo=Path(self.args.ffs_repo),
                model_dir=Path(self.args.ffs_trt_model_dir),
                trt_root=(
                    None
                    if self.args.ffs_trt_root is None
                    else Path(self.args.ffs_trt_root)
                ),
                cache_frames=DEFAULT_LOCAL_FFS_DEPTH_CACHE_FRAMES,
            )
            warm_up_numba_ffs_align()
        self.preload.start()
        self._emit_progress("preload", "perception preloads started")
        self.session.prepare_source(self.args, self.mode)
        if self.recorder is not None:
            runtime = self.session.camera_runtime
            if runtime is not None and getattr(runtime, "serial", None):
                self.recorder.serial = str(runtime.serial)
        self._notify_frame0_geometry_from_camera()
        self.session.initialize_table_calibration(self.args)
        self._acq_thread = self._spawn_worker("acquisition", self._acquisition.run)

    def run(self) -> int:
        """Run the service until shutdown; returns the process exit code."""
        try:
            self._start_service()
        except Exception as exc:
            self.fatal.record("service startup", exc)
        try:
            self._main_loop()
        finally:
            self._teardown()
        return 2 if self.fatal.snapshot() is not None else 0

    def _main_loop(self) -> None:
        """Main loop: deferred commands, state edges, formal drain, telemetry."""
        last_telemetry_s = time.perf_counter()
        while not self._shutdown.is_set():
            self._run_deferred()
            self._announce_fatal_if_needed()
            if (
                self.state == protocol.STATE_STARTING
                and self._first_frame_ready.is_set()
                and self.fatal.snapshot() is None
            ):
                self._enter_state(protocol.STATE_PREVIEW)
                self._spawn_early_precompile()
                # NOTE deliberately NO TripoSplat spawn here: its ~8s load
                # (disk + camera GPU) competes with the SAM3.1 3.5G
                # checkpoint preload — the very thing frame-0 confirm waits
                # on. It spawns after the chain submit instead; the
                # gaussian park window (~30s) absorbs the later start.
            formal = self._formal
            if formal is not None and not self._formal_finalized:
                if formal.lossless.processing_done.is_set():
                    self._formal_finalized = True
                    self._finalize_formal()
                    if self._shutdown_after_finalize:
                        self._shutdown.set()
                elif (
                    self._formal_drain_deadline_s is not None
                    and time.perf_counter() > self._formal_drain_deadline_s
                ):
                    # Shutdown drain escape hatch: the run cannot finalize,
                    # so record fatal (exit code 2) and stop waiting.
                    self.fatal.record(
                        "shutdown drain",
                        RuntimeError(
                            "formal pipeline did not drain within "
                            f"{SHUTDOWN_DRAIN_DEADLINE_S:.0f}s of shutdown"
                        ),
                    )
                    self._announce_fatal_if_needed()
                    self._shutdown.set()
                else:
                    now_s = time.perf_counter()
                    if now_s - last_telemetry_s >= 5.0:
                        last_telemetry_s = now_s
                        print(
                            "[queue-telemetry] "
                            + json.dumps(formal.lossless.telemetry(), sort_keys=True),
                            flush=True,
                        )
            self._shutdown.wait(0.05)

    def _finalize_formal(self) -> None:
        """Formal drain complete: teardown like v6.2 stop() + finalize product."""
        formal = self._formal
        assert formal is not None
        self.stop_event.set()
        formal.lossless.close_queues()
        threads = list(formal.threads)
        if self._acq_thread is not None:
            threads.append(self._acq_thread)
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        self.session.release_camera()
        self.shape_prior_manager.write_profile_json()
        incomplete_error = formal.timeline_gate.incomplete_run_error()
        writer = self.session.headless_capture_writer
        if writer is not None and incomplete_error is not None:
            # Same policy as v6.2 stop(): a run that ends while formal rows
            # are still gated on the shape prior is a fatal error.
            self.fatal.record("formal chunk timeline", RuntimeError(incomplete_error))
        if self.fatal.snapshot() is None and writer is not None:
            output_dir = Path(self.args.phystwin_strict_output_dir)
            print(f"[phystwin-strict] finalizing output_dir={output_dir}", flush=True)
            manifest = finalize_headless_capture(
                writer.output_dir, output_dir=output_dir
            )
            print(
                "[phystwin-strict] "
                f"frames={manifest.get('frame_count')} "
                f"queries={manifest.get('query_count')} "
                f"manifest={output_dir / 'manifest.json'}",
                flush=True,
            )
        self.session.headless_capture_writer = None
        self.session.depth_engine = None
        self._announce_fatal_if_needed()
        if self.fatal.snapshot() is None:
            if self._acquisition.replay_exhausted:
                self.control.send_event(
                    {"event": protocol.EVT_REPLAY_EXHAUSTED, "wrapped": False}
                )
            self._enter_state(protocol.STATE_FINISHED)

    def _teardown(self) -> None:
        """Stop workers, release the camera, close both sockets."""
        self.stop_event.set()
        # The gaussian worker is a child process (model in VRAM): a fatal
        # mid-warmup exit must not orphan it. Idempotent after _launch_formal.
        self._shutdown_gaussian_manager()
        formal = self._formal
        if formal is not None:
            try:
                formal.lossless.close_queues()
            except Exception:
                pass
            for thread in formal.threads:
                if thread.is_alive():
                    thread.join(timeout=1.0)
        for thread in (self._acq_thread, self._warmup_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=1.0)
        self.session.release_camera()
        if self.recorder is not None:
            try:
                self.recorder.close()
            except Exception as exc:
                print(f"[camera-service] recorder close failed: {exc}", flush=True)
        try:
            self.shape_prior_manager.write_profile_json()
        except Exception:
            pass
        # Give the control writer a moment to flush the final ack/events.
        time.sleep(0.2)
        self.control.close()
        self.frames.close()

    def close(self) -> None:
        """Release sockets without running the state machine (self-check)."""
        self.stop_event.set()
        self._shutdown.set()
        self.control.close()
        self.frames.close()


__all__ = ["StagedRuntime"]
