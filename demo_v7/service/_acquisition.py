"""Frame acquisition for the demo_v7 camera service (private helper).

One loop owns the frame source from PREVIEW through FORMAL. The per-frame
read/align/copy calls are the same ones ``demo_v7.runtime.mdp.capture.CaptureStage``
drives (RealSense ``wait_for_frames``/``align.process`` or the v6.2
``RecordedRgbdFrameSource`` replayer with its tick pacing and
``source_index_for_recording_elapsed_s`` frame selection); wrap-around and
the button-driven formal handover are v7-side additions only.

Constraints:
- never blocks on the GUI: every publish goes through callbacks that end in
  the latest-wins FrameStreamServer;
- fake-live NEVER pauses (a camera never stops; only the SOURCE is fake);
  pre-formal the recording wraps around with a GUI notification.
- the formal producer mirrors v6.2's lossless capture exactly: frame-0
  submit, seg + first-pair gates (previews keep flowing), then the fixed
  ``lossless_input_fps`` cadence with contiguous seqs from 0 — including the
  headless writer's input_frames.jsonl / input_rgb timeline rows.
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from demo_v7.runtime.mdp.capture_source import LiveLatestFrameSampler
from demo_v7.runtime.mdp.cli import RunMode
from demo_v7.runtime.mdp.packets import FramePacket, PipelineTiming
from demo_v7.runtime.mdp.plumbing import FatalErrorLatch
from demo_v7.runtime.utils.concurrency import elapsed_ms as _elapsed_ms

from demo_v7.ipc import protocol

if TYPE_CHECKING:
    from demo_v7.runtime.mdp.session import CameraSession
    from demo_v7.service._formal import FormalPipeline


class _FakeReplayClock:
    """v7-side wrapper over the v6.2 tick-pacing replay clock.

    Mirrors ``CaptureStage._capture_recording_worker``: recording-elapsed
    seconds advance one fixed period per tick and a tick is due only once its
    wall target passes; frame selection stays with the v6.2 replayer's
    ``source_index_for_recording_elapsed_s``. There is deliberately NO pause:
    fake-live fakes the SOURCE, never the continuous-input property of a
    camera — the stream runs from source open to shutdown (owner
    requirement), wrapping around pre-formal when the recording ends.
    """

    def __init__(self, period_s: float) -> None:
        """Initialize the clock with the emitted sample period."""
        self.period_s = float(period_s)
        self.origin_s = time.perf_counter()
        self.tick = 1

    def restart(self) -> None:
        """Restart recording time from zero (pre-formal preview wrap-around)."""
        self.origin_s = time.perf_counter()
        self.tick = 1

    def next_due(self, now_s: float) -> float | None:
        """Return the recording-elapsed seconds of the next due tick, or None."""
        elapsed_s = float(self.tick) * self.period_s
        if self.origin_s + elapsed_s > float(now_s):
            return None
        self.tick += 1
        return elapsed_s


class AcquisitionLoop:
    """Preview streamer + formal lossless producer, driven by StagedRuntime.

    All coupling to the state machine goes through injected callables so this
    loop never imports the runtime: ``on_replay_wrapped`` reports wraps,
    ``get_formal`` returns the stage set once start_formal built it, and the
    publish callbacks own JPEG encoding and per-channel rate caps.
    """

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        mode: RunMode,
        session: CameraSession,
        stop_event: threading.Event,
        fatal: FatalErrorLatch,
        formal_go: threading.Event,
        formal_stop: threading.Event,
        on_replay_wrapped: Callable[[], None],
        get_formal: Callable[[], FormalPipeline | None],
        store_latest: Callable[[FramePacket], None],
        latest_packet: Callable[[], FramePacket | None],
        publish_preview: Callable[[FramePacket], None],
        publish_frame: Callable[[str, np.ndarray], None],
    ) -> None:
        """Initialize AcquisitionLoop."""
        self.args = args
        self.mode = mode
        self.session = session
        self.stop_event = stop_event
        self.fatal = fatal
        self._formal_go = formal_go
        self._formal_stop = formal_stop
        self._on_replay_wrapped = on_replay_wrapped
        self._get_formal = get_formal
        self._store_latest = store_latest
        self._latest_packet = latest_packet
        self._publish_preview = publish_preview
        self._publish_frame = publish_frame
        # True when the fake source ran out during FORMAL (EVT_REPLAY_EXHAUSTED).
        self.replay_exhausted = False

    def run(self) -> None:
        """Own the frame source from PREVIEW through FORMAL."""
        if self.mode.fake_live_input:
            self._acquire_fake()
        else:
            self._acquire_live()

    def _write_input_timeline_frame(self, packet: FramePacket) -> None:
        """Mirror v6.2 ``CaptureStage._publish_input_preview_packet``'s writer leg.

        input_frames.jsonl / input_rgb rows land whenever the headless writer
        exists and the run is fake-live (always) or the operator asked for the
        live input timeline (``--write-input-rgb-timeline``).
        """
        writer = self.session.headless_capture_writer
        if writer is None:
            return
        if not (
            self.mode.fake_live_input or bool(self.args.write_input_rgb_timeline)
        ):
            return
        writer.write_input_frame(packet)

    # ---- live RealSense -----------------------------------------------------

    def _read_live_packet(self, seq: int) -> FramePacket | None:
        """One live read/align/copy (v6.2 CaptureStage.run per-frame body)."""
        runtime = self.session.camera_runtime
        assert runtime is not None
        pipeline = runtime.pipeline
        if pipeline is None:
            raise RuntimeError("live capture requires an initialized camera pipeline")
        wait_start_s = time.perf_counter()
        frames = pipeline.wait_for_frames()
        receive_perf_s = time.perf_counter()
        align_start_s = receive_perf_s
        if self.args.depth_source == "ffs":
            align_done_s = receive_perf_s
            color_frame = frames.get_color_frame()
            ir_left_frame = frames.get_infrared_frame(1)
            ir_right_frame = frames.get_infrared_frame(2)
            if not color_frame or not ir_left_frame or not ir_right_frame:
                return None
            depth_frame = None
        elif self.args.depth_source == "none":
            align_done_s = receive_perf_s
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            depth_frame = None
            ir_left_frame = None
            ir_right_frame = None
        else:
            align = runtime.align
            assert align is not None
            aligned = align.process(frames)
            align_done_s = time.perf_counter()
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                return None
            ir_left_frame = None
            ir_right_frame = None
        copy_start_s = time.perf_counter()
        color_bgr = np.ascontiguousarray(np.asanyarray(color_frame.get_data()).copy())
        if self.args.depth_source == "ffs":
            assert ir_left_frame is not None and ir_right_frame is not None
            depth_u16 = None
            ir_left_u8 = np.ascontiguousarray(
                np.asanyarray(ir_left_frame.get_data()).copy()
            )
            ir_right_u8 = np.ascontiguousarray(
                np.asanyarray(ir_right_frame.get_data()).copy()
            )
        elif self.args.depth_source == "none":
            depth_u16 = None
            ir_left_u8 = None
            ir_right_u8 = None
        else:
            assert depth_frame is not None
            depth_u16 = np.ascontiguousarray(
                np.asanyarray(depth_frame.get_data()).copy()
            )
            ir_left_u8 = None
            ir_right_u8 = None
        copy_done_s = time.perf_counter()
        return FramePacket(
            seq=int(seq),
            color_bgr=color_bgr,
            depth_source=str(self.args.depth_source),
            intrinsics=runtime.intrinsics,
            depth_scale_m_per_unit=runtime.depth_scale_m_per_unit,
            receive_perf_s=receive_perf_s,
            timing=PipelineTiming(
                wait_ms=_elapsed_ms(wait_start_s, receive_perf_s),
                align_ms=_elapsed_ms(align_start_s, align_done_s),
                frame_copy_ms=_elapsed_ms(copy_start_s, copy_done_s),
            ),
            depth_u16=depth_u16,
            ir_left_u8=ir_left_u8,
            ir_right_u8=ir_right_u8,
            k_ir_left=runtime.k_ir_left,
            t_ir_left_to_color=runtime.t_ir_left_to_color,
            k_color=runtime.k_color,
            ir_baseline_m=runtime.ir_baseline_m,
        )

    def _acquire_live(self) -> None:
        """Live preview loop; hands over to the formal producer on go."""
        raw_seq = 0
        while not self.stop_event.is_set() and not self._formal_go.is_set():
            try:
                packet = self._read_live_packet(raw_seq)
            except Exception as exc:
                if not self.stop_event.is_set():
                    self.fatal.record("RealSense capture", exc)
                return
            if packet is None:
                continue
            raw_seq += 1
            self._store_latest(packet)
            self._publish_preview(packet)
        if self._formal_go.is_set() and not self.stop_event.is_set():
            self._formal_capture_live()

    def _formal_capture_live(self) -> None:
        """Live formal producer (v6.2 lossless live branch + latest sampler)."""
        formal = self._get_formal()
        assert formal is not None
        lossless = formal.lossless
        sampler = LiveLatestFrameSampler(self.mode.lossless_input_fps)
        output_seq = 0
        raw_seq = 0

        def publish_output(packet: FramePacket, record_s: float) -> None:
            """Publish one formal output packet with contiguous sequencing."""
            nonlocal output_seq
            out = replace(packet, seq=output_seq)
            # v6.2 parity: every live output frame lands one input-timeline
            # row (gated on --write-input-rgb-timeline) before the capture
            # slot / lossless submit, like _publish_capture_packet.
            self._write_input_timeline_frame(out)
            formal.capture_slot.put(out)
            if lossless.submit_frame(out, stop_event=self.stop_event):
                formal.stage_stats.record("capture", float(record_s))
            output_seq += 1

        def pump_preview() -> None:
            """Drain the camera during a gate wait; publish CH_RGB only."""
            runtime = self.session.camera_runtime
            if runtime is None or runtime.pipeline is None:
                return
            try:
                gate_frames = runtime.pipeline.wait_for_frames()
            except Exception:
                return
            color_frame = gate_frames.get_color_frame()
            if not color_frame:
                return
            self._publish_frame(
                protocol.CH_RGB,
                np.ascontiguousarray(np.asanyarray(color_frame.get_data())),
            )

        while not self.stop_event.is_set() and not self._formal_stop.is_set():
            try:
                packet = self._read_live_packet(raw_seq)
            except Exception as exc:
                if not self.stop_event.is_set():
                    self.fatal.record("RealSense capture", exc)
                break
            if packet is None:
                continue
            raw_seq += 1
            self._publish_preview(packet)
            published_sample_before_current = False
            if output_seq > 0:
                due_sample = sampler.pop_due(now_s=packet.receive_perf_s)
                if due_sample is not None:
                    due_packet, sample_s = due_sample
                    publish_output(due_packet, sample_s)
                    published_sample_before_current = True
            copy_done_s = time.perf_counter()
            if output_seq == 0:
                publish_output(packet, copy_done_s)
                while (
                    not self.stop_event.is_set()
                    and not self._formal_stop.is_set()
                ):
                    if formal.first_frame_segmented.wait(timeout=0.005):
                        break
                    pump_preview()
                while (
                    not self.stop_event.is_set()
                    and not self._formal_stop.is_set()
                ):
                    if lossless.first_pair_published.wait(timeout=0.005):
                        break
                    pump_preview()
                if self.stop_event.is_set() or self._formal_stop.is_set():
                    break
                # v6.2 live capture never assigns startup_hold_s; it stays
                # at CaptureHold's 0.0 (only the recording worker measures it).
                sampler.start(first_publish_s=time.perf_counter())
                continue
            sampler.put_latest(packet)
            if not published_sample_before_current:
                due_sample = sampler.pop_due(now_s=copy_done_s)
                if due_sample is not None:
                    due_packet, sample_s = due_sample
                    publish_output(due_packet, sample_s)
        lossless.finish_capture()

    # ---- fake-live replay ---------------------------------------------------

    def _acquire_fake(self) -> None:
        """Fake-live preview loop over the v6.2 replayer (never pauses)."""
        source = self.session.recording_source
        assert source is not None
        period_s = (
            1.0 / self.mode.lossless_input_fps
            if self.mode.lossless_enabled
            else 1.0 / float(source.effective_fps)
        )
        clock = _FakeReplayClock(period_s)
        preview_seq = 0
        last_index = -1

        def show_index(index: int) -> None:
            """Read one full replay frame, store + publish it."""
            nonlocal preview_seq, last_index
            packet = source.read_packet(seq=preview_seq, frame_index=int(index))
            preview_seq += 1
            last_index = int(index)
            self._store_latest(packet)
            self._publish_preview(packet)

        try:
            show_index(0)
        except Exception as exc:
            if not self.stop_event.is_set():
                self.fatal.record("fake-live replay", exc)
            return
        while not self.stop_event.is_set() and not self._formal_go.is_set():
            elapsed_s = clock.next_due(time.perf_counter())
            if elapsed_s is None:
                self.stop_event.wait(0.005)
                continue
            index = source.source_index_for_recording_elapsed_s(elapsed_s)
            if index <= last_index:
                if last_index >= source.frame_count - 1:
                    # A camera never stops: pre-formal the recording wraps
                    # around and the GUI is told (播放完毕 toast) via the
                    # runtime's wrap callback.
                    clock.restart()
                    last_index = -1
                    self._on_replay_wrapped()
                continue
            try:
                show_index(index)
            except Exception as exc:
                if not self.stop_event.is_set():
                    self.fatal.record("fake-live replay", exc)
                return
        if self._formal_go.is_set() and not self.stop_event.is_set():
            self._formal_capture_fake(source, clock, last_index, preview_seq)

    def _formal_capture_fake(
        self,
        source: Any,
        clock: _FakeReplayClock,
        last_index: int,
        preview_seq: int,
    ) -> None:
        """Fake formal producer (v6.2 recording worker, resumed mid-case)."""
        formal = self._get_formal()
        assert formal is not None
        lossless = formal.lossless
        replay_exhausted = False

        def submit_index(index: int, seq: int) -> FramePacket:
            """Read one replay frame and feed the lossless pipeline."""
            nonlocal preview_seq
            packet = source.read_packet(seq=int(seq), frame_index=int(index))
            self._publish_preview(packet)
            # v6.2 parity: every formal output frame lands one input-timeline
            # row, sequenced on the shared preview counter so filenames stay
            # unique against the gate-tick rows (capture.py fake branch keeps
            # ONE monotonic preview seq across previews and outputs).
            self._write_input_timeline_frame(replace(packet, seq=preview_seq))
            preview_seq += 1
            formal.capture_slot.put(packet)
            if lossless.submit_frame(packet, stop_event=self.stop_event):
                formal.stage_stats.record("capture", packet.receive_perf_s)
            return packet

        def tick_previews() -> None:
            """Keep due preview frames flowing during the frame-0 gate waits.

            Mirrors v6.2's publish_due_fake_live_previews: recording time keeps
            running through the gate, and frames shown here are skipped from
            the formal timeline by the shared ``last_index`` cursor.
            """
            nonlocal last_index, preview_seq
            elapsed_s = clock.next_due(time.perf_counter())
            if elapsed_s is None:
                return
            index = source.source_index_for_recording_elapsed_s(elapsed_s)
            if index <= last_index or index >= source.frame_count:
                return
            try:
                packet = source.read_preview_packet(
                    seq=preview_seq, frame_index=int(index)
                )
            except Exception:
                return
            preview_seq += 1
            last_index = int(index)
            self._publish_preview(packet)
            self._write_input_timeline_frame(packet)

        start_index = max(0, int(last_index))
        try:
            first_packet = submit_index(start_index, 0)
        except Exception as exc:
            if not self.stop_event.is_set():
                self.fatal.record("fake-live replay", exc)
            lossless.finish_capture()
            return
        # v6.2 capture.py startup hold: from the frame-0 submit packet's
        # receive stamp (camera_start_s, :139) to right after the first
        # lossless pair publishes (gate_done_s, :240-243).
        camera_start_s = float(first_packet.receive_perf_s)
        while not self.stop_event.is_set() and not self._formal_stop.is_set():
            if formal.first_frame_segmented.wait(timeout=0.01):
                break
            tick_previews()
        while not self.stop_event.is_set() and not self._formal_stop.is_set():
            if lossless.first_pair_published.wait(timeout=0.01):
                break
            tick_previews()
        formal.capture_hold.startup_hold_s = max(
            0.0, time.perf_counter() - camera_start_s
        )
        runtime_seq = 1
        while not self.stop_event.is_set() and not self._formal_stop.is_set():
            elapsed_s = clock.next_due(time.perf_counter())
            if elapsed_s is None:
                self.stop_event.wait(0.005)
                continue
            index = source.source_index_for_recording_elapsed_s(elapsed_s)
            if index <= last_index:
                if last_index >= source.frame_count - 1:
                    replay_exhausted = True
                    break
                continue
            last_index = int(index)
            try:
                submit_index(index, runtime_seq)
            except Exception as exc:
                if not self.stop_event.is_set():
                    self.fatal.record("fake-live replay", exc)
                break
            runtime_seq += 1
            if last_index >= source.frame_count - 1:
                replay_exhausted = True
                break
        self.replay_exhausted = replay_exhausted
        lossless.finish_capture()


__all__ = ["AcquisitionLoop"]
