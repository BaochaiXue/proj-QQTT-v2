"""MainDataProcessingDemo capture-worker mixin."""
from __future__ import annotations

from demo_v6_2.mdp_constants import *  # noqa: F401,F403
from demo_v6_2.mdp_cli import _is_replay_input_source
from demo_v6_2.mdp_packets import FramePacket, LiveLatestFrameSampler, PipelineTiming


class _CaptureMixin:
    """MainDataProcessingDemo capture-worker mixin."""

    def _publish_input_preview_packet(self, packet: FramePacket, *, record_s: float | None = None) -> None:
        """Publish input preview packet."""
        self.input_preview_slot.put(packet)
        should_write_timeline = _is_replay_input_source(str(self.args.input_source)) or bool(
            getattr(self.args, "write_input_rgb_timeline", False)
        )
        if self.headless_capture_writer is not None and should_write_timeline:
            self.headless_capture_writer.write_input_frame(packet)

    def _publish_capture_packet(
        self,
        packet: FramePacket,
        *,
        record_s: float | None = None,
        write_input_timeline: bool = True,
    ) -> None:
        """Publish capture packet."""
        if bool(write_input_timeline):
            self._publish_input_preview_packet(packet, record_s=record_s)
        self.capture_slot.put(packet)
        if self._lossless_enabled():
            if self.lossless_frame_queue.put_wait(packet, stop_event=self.stop_event) <= 0:
                return
            self._lossless_offered_frames += 1
        self.capture_stats.record(packet.receive_perf_s if record_s is None else float(record_s))

    def _capture_recording_worker(self) -> None:
        """Return the capture recording worker."""
        assert self.recording_source is not None
        source = self.recording_source
        fake_live_clock = str(self.args.input_source) == INPUT_SOURCE_FAKE_LIVE
        if self._lossless_enabled():
            frame_period_s = 1.0 / self._lossless_input_fps()
        else:
            frame_period_s = 1.0 / float(source.effective_fps)
        try:
            first_packet = source.read_packet(seq=0)
        except Exception as exc:
            if not self.stop_event.is_set():
                self._record_fatal_worker_error("recording replay", exc)
            return
        camera_start_s = float(first_packet.receive_perf_s)
        preview_seq = 0
        preview_tick = 1
        last_preview_source_index = -1

        def preview_from_packet(packet: FramePacket, *, seq: int) -> FramePacket:
            """Return the preview from packet."""
            return replace(
                packet,
                seq=int(seq),
                depth_u16=None,
                ir_left_u8=None,
                ir_right_u8=None,
                k_ir_left=None,
                t_ir_left_to_color=None,
                ir_baseline_m=0.0,
            )

        def read_preview_packet(
            *,
            seq: int,
            source_index: int,
            wait_ms: float = 0.0,
        ) -> FramePacket:
            """Read preview packet."""
            reader = getattr(source, "read_preview_packet", None)
            if callable(reader):
                return reader(seq=int(seq), frame_index=int(source_index), wait_ms=float(wait_ms))
            packet = source.read_packet(seq=int(seq), frame_index=int(source_index), wait_ms=float(wait_ms))
            return preview_from_packet(packet, seq=int(seq))

        def publish_preview_packet(packet: FramePacket) -> None:
            """Publish preview packet."""
            nonlocal preview_seq, last_preview_source_index
            self._publish_input_preview_packet(packet, record_s=packet.receive_perf_s)
            preview_seq += 1
            if packet.source_frame_index is not None:
                last_preview_source_index = max(last_preview_source_index, int(packet.source_frame_index))

        def publish_preview_source_index(*, source_index: int, wait_ms: float = 0.0) -> None:
            """Publish preview source index."""
            nonlocal preview_seq, last_preview_source_index
            if int(source_index) <= int(last_preview_source_index):
                return
            packet = read_preview_packet(seq=preview_seq, source_index=int(source_index), wait_ms=float(wait_ms))
            publish_preview_packet(packet)

        def publish_due_fake_live_previews() -> bool:
            """Publish due fake live previews."""
            nonlocal preview_tick
            if not fake_live_clock:
                return True
            now_s = time.perf_counter()
            while not self.stop_event.is_set():
                source_elapsed_s = float(preview_tick) * frame_period_s
                target_s = camera_start_s + source_elapsed_s
                if target_s > now_s:
                    break
                source_index = source.source_index_for_recording_elapsed_s(source_elapsed_s)
                preview_tick += 1
                if source_index <= last_preview_source_index:
                    if last_preview_source_index >= source.frame_count - 1:
                        break
                    continue
                try:
                    publish_preview_source_index(source_index=source_index)
                except Exception as exc:
                    if not self.stop_event.is_set():
                        self._record_fatal_worker_error("recording replay preview", exc)
                    return False
                if source_index >= source.frame_count - 1:
                    break
            return True

        if fake_live_clock:
            publish_preview_packet(preview_from_packet(first_packet, seq=preview_seq))
            self._publish_capture_packet(
                first_packet,
                record_s=first_packet.receive_perf_s,
                write_input_timeline=False,
            )
        else:
            self._publish_capture_packet(first_packet, record_s=first_packet.receive_perf_s)
        if source.frame_count <= 1:
            if self._lossless_enabled():
                self._lossless_capture_done.set()
                self.lossless_frame_queue.close()
            else:
                self.stop_event.set()
            return
        if self.args.track_mode != "none":
            while not self.stop_event.is_set():
                if self._recording_first_frame_segmented.wait(timeout=0.01):
                    break
                if not publish_due_fake_live_previews():
                    return
            if self.stop_event.is_set():
                return
        if not self._wait_for_lossless_replay_startup_pair(on_wait_tick=publish_due_fake_live_previews):
            return
        gate_done_s = time.perf_counter()
        if not publish_due_fake_live_previews():
            return
        self._startup_hold_s = max(0.0, float(gate_done_s - camera_start_s))
        if self.headless_capture_writer is not None:
            self.headless_capture_writer.update_metadata({"startup_hold_s": float(self._startup_hold_s)})
        replay_start_s = gate_done_s
        runtime_seq = 1
        if fake_live_clock:
            output_tick = max(1, int(preview_tick))
            last_source_index = max(0, int(last_preview_source_index))
            while not self.stop_event.is_set():
                source_elapsed_s = float(output_tick) * frame_period_s
                source_index = source.source_index_for_recording_elapsed_s(source_elapsed_s)
                output_tick += 1
                if source_index <= last_source_index:
                    if last_source_index >= source.frame_count - 1:
                        break
                    continue
                wait_start_s = time.perf_counter()
                target_s = camera_start_s + source_elapsed_s
                wait_s = target_s - wait_start_s
                if wait_s > 0.0 and self.stop_event.wait(wait_s):
                    break
                wait_done_s = time.perf_counter()
                try:
                    packet = source.read_packet(
                        seq=runtime_seq,
                        frame_index=source_index,
                        wait_ms=_elapsed_ms(wait_start_s, wait_done_s),
                    )
                except Exception as exc:
                    if not self.stop_event.is_set():
                        self._record_fatal_worker_error("recording replay", exc)
                    break
                publish_preview_packet(preview_from_packet(packet, seq=preview_seq))
                self._publish_capture_packet(
                    packet,
                    record_s=packet.receive_perf_s,
                    write_input_timeline=False,
                )
                runtime_seq += 1
                last_source_index = source_index
                if last_source_index >= source.frame_count - 1:
                    break
        else:
            for source_index in range(1, source.frame_count):
                if self.stop_event.is_set():
                    break
                wait_start_s = time.perf_counter()
                target_s = replay_start_s + (float(runtime_seq) * frame_period_s)
                wait_s = target_s - wait_start_s
                if wait_s > 0.0 and self.stop_event.wait(wait_s):
                    break
                wait_done_s = time.perf_counter()
                try:
                    packet = source.read_packet(
                        seq=runtime_seq,
                        frame_index=source_index,
                        wait_ms=_elapsed_ms(wait_start_s, wait_done_s),
                    )
                except Exception as exc:
                    if not self.stop_event.is_set():
                        self._record_fatal_worker_error("recording replay", exc)
                    break
                self._publish_capture_packet(packet, record_s=packet.receive_perf_s)
                runtime_seq += 1
        if self._lossless_enabled():
            self._lossless_capture_done.set()
            self.lossless_frame_queue.close()
        else:
            self.stop_event.set()

    def _capture_worker(self) -> None:
        """Return the capture worker."""
        assert self.runtime is not None
        if _is_replay_input_source(str(self.args.input_source)):
            self._capture_recording_worker()
            return
        raw_seq = 0
        output_seq = 0
        live_sampler = (
            LiveLatestFrameSampler(self._lossless_input_fps())
            if self._lossless_enabled()
            else None
        )
        pipeline = self.runtime.pipeline
        align = self.runtime.align

        def publish_output_packet(packet: FramePacket, *, record_s: float) -> None:
            """Publish one live output packet with contiguous demo sequencing."""
            nonlocal output_seq
            output_packet = replace(packet, seq=output_seq)
            self._publish_capture_packet(output_packet, record_s=float(record_s))
            output_seq += 1

        while not self.stop_event.is_set():
            wait_start_s = time.perf_counter()
            try:
                frames = pipeline.wait_for_frames()
            except Exception as exc:
                if not self.stop_event.is_set():
                    self._record_fatal_worker_error("RealSense capture", exc)
                break
            receive_perf_s = time.perf_counter()
            published_sample_before_current = False
            if live_sampler is not None:
                due_sample = live_sampler.pop_due(now_s=receive_perf_s)
                if due_sample is not None:
                    due_packet, sample_s = due_sample
                    publish_output_packet(due_packet, record_s=sample_s)
                    published_sample_before_current = True
            align_start_s = receive_perf_s
            if self.args.depth_source == "ffs":
                align_done_s = receive_perf_s
                color_frame = frames.get_color_frame()
                ir_left_frame = frames.get_infrared_frame(1)
                ir_right_frame = frames.get_infrared_frame(2)
                if not color_frame or not ir_left_frame or not ir_right_frame:
                    continue
                depth_frame = None
            elif self.args.depth_source == "none":
                align_done_s = receive_perf_s
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                depth_frame = None
                ir_left_frame = None
                ir_right_frame = None
            else:
                assert align is not None
                aligned = align.process(frames)
                align_done_s = time.perf_counter()
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                ir_left_frame = None
                ir_right_frame = None
            copy_start_s = time.perf_counter()
            color_bgr = np.ascontiguousarray(np.asanyarray(color_frame.get_data()).copy())
            if self.args.depth_source == "ffs":
                assert ir_left_frame is not None and ir_right_frame is not None
                depth_u16 = None
                ir_left_u8 = np.ascontiguousarray(np.asanyarray(ir_left_frame.get_data()).copy())
                ir_right_u8 = np.ascontiguousarray(np.asanyarray(ir_right_frame.get_data()).copy())
            elif self.args.depth_source == "none":
                depth_u16 = None
                ir_left_u8 = None
                ir_right_u8 = None
            else:
                assert depth_frame is not None
                depth_u16 = np.ascontiguousarray(np.asanyarray(depth_frame.get_data()).copy())
                ir_left_u8 = None
                ir_right_u8 = None
            copy_done_s = time.perf_counter()
            packet = FramePacket(
                seq=raw_seq,
                color_bgr=color_bgr,
                depth_source=str(self.args.depth_source),
                intrinsics=self.runtime.intrinsics,
                depth_scale_m_per_unit=self.runtime.depth_scale_m_per_unit,
                receive_perf_s=receive_perf_s,
                timing=PipelineTiming(
                    wait_ms=_elapsed_ms(wait_start_s, receive_perf_s),
                    align_ms=_elapsed_ms(align_start_s, align_done_s),
                    frame_copy_ms=_elapsed_ms(copy_start_s, copy_done_s),
                ),
                depth_u16=depth_u16,
                ir_left_u8=ir_left_u8,
                ir_right_u8=ir_right_u8,
                k_ir_left=self.runtime.k_ir_left,
                t_ir_left_to_color=self.runtime.t_ir_left_to_color,
                k_color=self.runtime.k_color,
                ir_baseline_m=self.runtime.ir_baseline_m,
            )
            raw_seq += 1
            if live_sampler is None:
                publish_output_packet(packet, record_s=copy_done_s)
                continue
            if output_seq == 0:
                publish_output_packet(packet, record_s=copy_done_s)
                if self.args.track_mode != "none":
                    while not self.stop_event.is_set():
                        if self._recording_first_frame_segmented.wait(timeout=0.01):
                            break
                    if self.stop_event.is_set():
                        break
                live_sampler.start(first_publish_s=time.perf_counter())
                continue
            live_sampler.put_latest(packet)
            if not published_sample_before_current:
                due_sample = live_sampler.pop_due(now_s=copy_done_s)
                if due_sample is not None:
                    due_packet, sample_s = due_sample
                    publish_output_packet(due_packet, record_s=sample_s)
        if self._lossless_enabled():
            self._lossless_capture_done.set()
            self.lossless_frame_queue.close()


__all__ = ["_CaptureMixin"]
