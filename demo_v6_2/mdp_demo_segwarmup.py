"""MainDataProcessingDemo segmentation/warmup/shape-prior mixin.

Pipeline questions Q7-Q14 (see PIPELINE.md): warm-up anchors on the FIRST single
frame (``_wait_for_first_frame`` reads it with a sentinel seq of -1 and seeds
EdgeTAM once); later frames are consumed by the tracker but not chunked until the
formal gate lifts. The dominant warm-up cost is the SAM3D shape-prior chain
(submitted here, generated in ``shape_prior_warmup``); warm-up produces the frozen
frame-0 masks + seeded EdgeTAM session + shape-prior artifacts. The formal timeline
starts at output frame 1, one frame after the warm-up frame-0 anchor, once
shape-prior is READY. Warm-up failures route to ``_record_fatal_worker_error`` and
tear the process down (surfaced on the live status band, Q23).
"""
from __future__ import annotations

from demo_v6_2.mdp_constants import *  # noqa: F401,F403
from demo_v6_2.mdp_cli import _is_replay_input_source, active_object_id_labels, active_object_ids, controller_pcd_mask_erode_pixels, controller_tracking_enabled, depth_backend_label, headless_capture_enabled, headless_capture_saved_pcd_source, object_pcd_mask_erode_pixels, object_tracking_enabled, pcd_filter_enabled, runtime_metadata_identity, tracker_enabled, tracker_marker_gate, tracker_marker_retirement_policy, tracker_query_source, tracker_retire_filtered_markers
from demo_v6_2.mdp_packets import MaskPacket, _formal_chunk_rows_gated
from demo_v6_2.mdp_segmentation import _load_hf_streaming_runtime, _time_model_forward, _time_runtime_ms, extract_object_masks_from_hf_output
from demo_v6_2.pipeline_status import STAGE_SHAPE_PRIOR, STAGE_WARMUP_READY


class _SegWarmupMixin:
    """MainDataProcessingDemo segmentation/warmup/shape-prior mixin."""

    def _init_hf_model(self) -> tuple[Any, Any, Any, Any, Any]:
        """Return the init HF model."""
        init_start_s = time.perf_counter()
        runtime_load_start_s = time.perf_counter()
        hf_stream = _load_hf_streaming_runtime()
        torch_module = hf_stream.torch
        if str(self.args.device).startswith("cuda") and not torch_module.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        dtype = hf_stream._dtype_from_name(self.args.dtype)
        runtime_load_end_s = time.perf_counter()
        model_load_start_s = time.perf_counter()
        model = hf_stream.EdgeTamVideoModel.from_pretrained(DEFAULT_MODEL_ID).to(
            self.args.device,
            dtype=dtype,
        )
        model.eval()
        model_load_end_s = time.perf_counter()
        compile_start_s = time.perf_counter()
        model, compile_metadata = hf_stream._apply_compile_mode(model, DEFAULT_COMPILE_MODE)
        compile_end_s = time.perf_counter()
        processor_load_start_s = time.perf_counter()
        processor = hf_stream.Sam2VideoProcessor.from_pretrained(DEFAULT_MODEL_ID)
        processor_load_end_s = time.perf_counter()
        self._warmup_perception_profile["edgetam_runtime_init"] = {
            "runtime_import_ms": _elapsed_ms(
                runtime_load_start_s,
                runtime_load_end_s,
            ),
            "model_load_ms": _elapsed_ms(
                model_load_start_s,
                model_load_end_s,
            ),
            "compile_ms": _elapsed_ms(compile_start_s, compile_end_s),
            "processor_load_ms": _elapsed_ms(
                processor_load_start_s,
                processor_load_end_s,
            ),
            "total_ms": _elapsed_ms(init_start_s, processor_load_end_s),
        }
        metadata = {
            **runtime_metadata_identity(self.args),
            "edge_model": DEFAULT_MODEL_ID,
            "demo_preset": "none",
            "compile_mode": DEFAULT_COMPILE_MODE,
            "applied_targets": compile_metadata.get("applied_targets", []),
            "dtype": self.args.dtype,
            "inference_device": self.args.device,
            "inference_state_device": self.args.device,
            "video_storage_device": self.args.device,
            "frame_by_frame_streaming": True,
            "edgetam_live_session_keep_frames": int(DEFAULT_EDGETAM_LIVE_SESSION_KEEP_FRAMES),
            "offline_video_input_used": _is_replay_input_source(str(self.args.input_source)),
            "input_source": self.args.input_source,
            "demo_visual_mode": str(self.args.demo_visual_mode),
            "recording_case": (
                _repo_relative_path_text(self.args.recording_case) if _is_replay_input_source(str(self.args.input_source)) else None
            ),
            "replay_fps": (
                self.recording_source.effective_fps
                if _is_replay_input_source(str(self.args.input_source)) and self.recording_source is not None
                else None
            ),
            "recording_fps": (
                self.recording_source.recording_fps
                if _is_replay_input_source(str(self.args.input_source)) and self.recording_source is not None
                else None
            ),
            "fake_live_frame_selection_policy": (
                FAKE_LIVE_FRAME_SELECTION_POLICY if str(self.args.input_source) == INPUT_SOURCE_FAKE_LIVE else None
            ),
            "track_mode": self.args.track_mode,
            "edgetam_tracking_identities": list(active_object_id_labels(self.args).values()),
            "depth_source": self.args.depth_source,
            "depth_source_internal": str(self.args.depth_source),
            "depth_units": "meters",
            "depth_coordinate_frame": COORDINATE_FRAME,
            "depth_alignment_target": "color",
            "local_ffs_depth_cache_frames": (
                DEFAULT_LOCAL_FFS_DEPTH_CACHE_FRAMES if self.args.depth_source == "ffs" else None
            ),
            "pcd_mode": self.args.pcd_mode,
            "pcd_coordinate_frame": self._pcd_coordinate_frame(),
            "camera_coordinate_frame": COORDINATE_FRAME,
            "table_calibration_path": _repo_relative_path_text(self.table_calibration_path),
            "table_world_frame_kind": TABLE_WORLD_FRAME_KIND if self._table_world_enabled() else None,
            "table_z_m": TABLE_Z_M if self._table_world_enabled() else None,
            "table_z_above_direction": TABLE_Z_ABOVE_DIRECTION,
            "camera_to_world_c2w": (
                None
                if self.table_c2w is None
                else np.asarray(self.table_c2w, dtype=np.float32).reshape(4, 4).tolist()
            ),
            "pcd_max_points": int(60000),
            "pcd_stride": int(1),
            "pcd_mask_erode_pixels": int(DEFAULT_PCD_MASK_ERODE_PIXELS),
            "object_pcd_mask_erode_pixels": object_pcd_mask_erode_pixels(self.args),
            "controller_pcd_mask_erode_pixels": controller_pcd_mask_erode_pixels(self.args),
            "pcd_filter_enabled": pcd_filter_enabled(self.args),
            "pcd_filter_mode": self.args.pcd_filter_mode if pcd_filter_enabled(self.args) else PCD_FILTER_NONE,
            "pcd_filter_preset": getattr(self.args, "pcd_filter_preset", None),
            "world_z_diagnostic_thresholds_m": [
                float(value) for value in DEFAULT_TABLE_Z_DIAGNOSTIC_THRESHOLDS_M
            ],
            "table_z_filter_enabled": bool(self.args.enable_table_z_filter),
            "table_z_filter_threshold_m": float(DEFAULT_TABLE_Z_FILTER_THRESHOLD_M),
            "table_z_filter_classes": str(TABLE_Z_FILTER_CLASS_BOTH),
            "headless_capture_enabled": headless_capture_enabled(self.args),
            "headless_prepared_only": bool(getattr(self.args, "headless_prepared_only", False)),
            "headless_capture_dir": (
                _repo_relative_path_text(self.args.headless_capture_dir) if headless_capture_enabled(self.args) else None
            ),
            "saved_pcd_source": (
                headless_capture_saved_pcd_source(self.args) if headless_capture_enabled(self.args) else None
            ),
            "object_filter": self.args.object_filter,
            "controller_filter": self.args.controller_filter,
            "object_filter_cap": int(self.args.object_filter_cap),
            "controller_filter_cap": int(self.args.controller_filter_cap),
            "object_filter_keep_components": int(self.args.object_filter_keep_components),
            "controller_filter_keep_components": int(self.args.controller_filter_keep_components),
            "object_filter_min_retain_ratio": float(DEFAULT_OBJECT_FILTER_MIN_RETAIN_RATIO),
            "controller_filter_min_retain_ratio": float(DEFAULT_CONTROLLER_FILTER_MIN_RETAIN_RATIO),
            "object_filter_min_raw_retain_ratio": float(DEFAULT_OBJECT_FILTER_MIN_RAW_RETAIN_RATIO),
            "controller_filter_min_raw_retain_ratio": float(DEFAULT_CONTROLLER_FILTER_MIN_RAW_RETAIN_RATIO),
            "filter_every_n": int(self.args.filter_every_n),
            "filter_max_age_frames": int(self.args.filter_max_age_frames),
            "filter_budget_ms": float(12.0),
            "filter_min_cap": int(5000),
            "lossless_controller_filter_min_cap": (
                int(self.controller_filter_budget.min_cap) if self._lossless_enabled() else None
            ),
            "tracker_backend": str(self.args.tracker_backend),
            "tracking_product_backend": str(
                normalize_tracking_product_backend(getattr(self.args, "tracking_product_backend", DEFAULT_TRACKING_PRODUCT_BACKEND))
            ),
            "phystwin_strict_output_dir": (
                None
                if getattr(self.args, "phystwin_strict_output_dir", None) is None
                else _repo_relative_path_text(self.args.phystwin_strict_output_dir)
            ),
            "compatibility_target": COMPATIBILITY_TARGET_PHYSTWIN,
            "mask_backend": "edgetam",
            "depth_backend": depth_backend_label(self.args),
            "execution_mode": PHYSTWIN_STRICT_EXECUTION_MODE,
            "tracker_device": str(self.args.tracker_device),
            "tracker_query_count": int(DEFAULT_TRACKER_QUERY_COUNT),
            "tracker_query_source": tracker_query_source(self.args) if tracker_enabled(self.args) else None,
            "tracker_marker_gate": tracker_marker_gate(self.args) if tracker_enabled(self.args) else None,
            "tracker_retire_filtered_markers": (
                tracker_retire_filtered_markers(self.args) if tracker_enabled(self.args) else None
            ),
            "tracker_marker_retirement_policy": (
                tracker_marker_retirement_policy(self.args) if tracker_enabled(self.args) else None
            ),
            "tracker_display_scope": str(DEFAULT_TRACKER_DISPLAY_SCOPE),
            "tracker_overlay_max_points": int(self.args.tracker_overlay_max_points),
            "tracker_marker_point_size": float(DEFAULT_TRACKER_MARKER_POINT_SIZE),
            "tracker_strict_same_seq_render": bool(tracker_enabled(self.args) and self.args.pcd_mode == "masked"),
            "tracker_visualization_mode": (
                "phystwin_rainbow_identity_3d_lift" if tracker_enabled(self.args) else "none"
            ),
            "tracker_sync_policy": (
                "strict_same_seq_lossless_5fps" if tracker_enabled(self.args) and self.args.pcd_mode == "masked" else "none"
            ),
            "lossless_input_fps": (
                float(self._lossless_input_fps()) if tracker_enabled(self.args) and self.args.pcd_mode == "masked" else None
            ),
            "lossless_max_backlog_frames": (
                int(self.lossless_max_backlog_frames) if tracker_enabled(self.args) and self.args.pcd_mode == "masked" else None
            ),
            "query_display_policy": "visible_3d_lifted_all" if tracker_enabled(self.args) else "none",
            "query_color_mode": "phystwin_rainbow_identity" if tracker_enabled(self.args) else "none",
            "tracker_lift_mask_erode_pixels": min(
                object_pcd_mask_erode_pixels(self.args),
                controller_pcd_mask_erode_pixels(self.args),
            ),
            "tapnet_repo_dir": str(DEFAULT_TAPNET_REPO_DIR),
            "tapnextpp_checkpoint": str(DEFAULT_TAPNEXTPP_CHECKPOINT),
            "tapnextpp_image_size": str("256,256"),
            "tapnextpp_autocast_dtype": str("fp16"),
            "tapnextpp_compile": bool(False),
            "tapnextpp_fast_postprocess": bool(True),
        }
        print(
            "[edgetam] "
            f"model={DEFAULT_MODEL_ID} device={self.args.device} dtype={self.args.dtype} "
            f"track_mode={self.args.track_mode} compile_mode={DEFAULT_COMPILE_MODE} "
            f"applied={compile_metadata.get('applied_targets', [])}",
            flush=True,
        )
        print(f"[edgetam-metadata] {json.dumps(metadata, sort_keys=True)}", flush=True)
        return hf_stream, torch_module, dtype, model, processor

    def _seg_worker(self) -> None:
        """Return the seg worker."""
        try:
            warmup = main_warmup.prepare_segmentation_warmup(self)
            first_frame = warmup.first_frame
            if first_frame is None:
                return
            initial_masks = warmup.initial_masks
            if initial_masks is None:
                raise RuntimeError("segmentation warmup did not produce frame-0 masks")
            session_start_s = time.perf_counter()
            session = warmup.hf_stream.EdgeTamVideoInferenceSession(
                video=None,
                video_height=int(first_frame.color_bgr.shape[0]),
                video_width=int(first_frame.color_bgr.shape[1]),
                inference_device=self.args.device,
                inference_state_device=self.args.device,
                video_storage_device=self.args.device,
                dtype=warmup.dtype,
            )
            session_end_s = time.perf_counter()
            self._warmup_perception_profile["edgetam_session_init_ms"] = (
                _elapsed_ms(session_start_s, session_end_s)
            )
            with warmup.torch_module.inference_mode():
                first_packet = self._run_segmentation_frame(
                    hf_stream=warmup.hf_stream,
                    torch_module=warmup.torch_module,
                    dtype=warmup.dtype,
                    model=warmup.model,
                    processor=warmup.processor,
                    session=session,
                    frame=first_frame,
                    initial_masks=initial_masks,
                    add_prompt=True,
                )
                self._publish_mask_packet(first_packet)
                self.seg_stats.record(first_packet.process_done_perf_s)
                if self._lossless_enabled() or _is_replay_input_source(str(self.args.input_source)):
                    self._recording_first_frame_segmented.set()
                last_seq = first_frame.seq
                while not self.stop_event.is_set():
                    if self._lossless_enabled():
                        frame = self.lossless_frame_queue.get(stop_event=self.stop_event)
                        if frame is None:
                            break
                    else:
                        frame = self.capture_slot.get_latest_after(last_seq)
                        if frame is None:
                            time.sleep(0.001)
                            continue
                    last_seq = frame.seq
                    try:
                        packet = self._run_segmentation_frame(
                            hf_stream=warmup.hf_stream,
                            torch_module=warmup.torch_module,
                            dtype=warmup.dtype,
                            model=warmup.model,
                            processor=warmup.processor,
                            session=session,
                            frame=frame,
                            initial_masks=initial_masks,
                            add_prompt=False,
                        )
                    except Exception as exc:
                        self._record_fatal_worker_error("EdgeTAM segmentation", exc)
                        break
                    self._publish_mask_packet(packet)
                    self.seg_stats.record(packet.process_done_perf_s)
                if self._lossless_enabled():
                    self.lossless_pcd_mask_queue.close()
                    self.lossless_tracker_mask_queue.close()
        except Exception as exc:
            if not self.stop_event.is_set():
                self._record_fatal_worker_error("segmentation worker", exc)
            if self._lossless_enabled():
                self.lossless_pcd_mask_queue.close()
                self.lossless_tracker_mask_queue.close()

    def _shape_prior_frame0_request_from_pcd_result(
        self,
        result: PcdBuildResult,
    ) -> shape_prior_warmup.ShapePriorFrame0Request | None:
        """Return the shape prior frame0 request from PCD result."""
        if not bool(getattr(self.args, "shape_prior_warmup", False)):
            return None
        if self.table_c2w is None:
            return None
        if result.depth_m is None:
            return None
        mask_packet = result.mask_packet
        if int(result.packet.seq) != int(mask_packet.seq):
            return None
        k_color = mask_packet.k_color
        if k_color is None and self.runtime is not None:
            k_color = np.asarray(self.runtime.k_color, dtype=np.float32)
        if k_color is None:
            return None
        object_observation_mask = self._shape_prior_observation_mask_from_pcd_result(result)
        return shape_prior_warmup.ShapePriorFrame0Request(
            seq=int(mask_packet.seq),
            source_timestamp_s=mask_packet.source_timestamp_s,
            input_source=str(self.args.input_source),
            depth_backend=depth_backend_label(self.args),
            depth_source_internal=str(self.args.depth_source),
            rgb_u8=mask_packet.color_bgr[:, :, ::-1],
            object_mask=mask_packet.object_mask,
            object_observation_mask=object_observation_mask,
            controller_mask=mask_packet.controller_mask,
            depth_color_m=result.depth_m,
            k_color=k_color,
            camera_to_world_c2w=self.table_c2w,
            table_z_m=TABLE_Z_M,
            warmup_runtime_start_perf_s=self._warmup_runtime_start_perf_s,
            frame_receive_perf_s=float(mask_packet.receive_perf_s),
            frame_mask_ready_perf_s=float(mask_packet.process_done_perf_s),
            frame_pcd_ready_perf_s=float(result.packet.process_done_perf_s),
            frame0_pipeline_timing_ms={
                key: float(value)
                for key, value in asdict(result.packet.timing).items()
            },
            frame0_perception_profile=dict(self._warmup_perception_profile),
        )

    def _shape_prior_observation_mask_from_pcd_result(self, result: PcdBuildResult) -> np.ndarray:
        """Return the shape prior observation mask from PCD result."""
        raw = np.asarray(result.mask_packet.object_mask, dtype=bool)
        candidate = result.object_observation_mask
        if candidate is None:
            candidate = result.object_pcd_mask
        if candidate is None:
            return np.ascontiguousarray(raw, dtype=bool)

        mask = np.asarray(candidate, dtype=bool)
        if mask.shape == raw.shape:
            return np.ascontiguousarray(mask, dtype=bool)

        stride = max(1, int(result.pcd_stride))
        if stride > 1 and mask.shape == raw[::stride, ::stride].shape:
            expanded = np.zeros_like(raw, dtype=bool)
            expanded[::stride, ::stride] = mask
            return np.ascontiguousarray(expanded, dtype=bool)

        return np.ascontiguousarray(mask, dtype=bool)

    def _packet_with_shape_prior_state(self, packet: MaskedPcdPacket) -> MaskedPcdPacket:
        """Return the packet with shape prior state."""
        profile = self._shape_prior_profile()
        result = self.shape_prior_manager.ready_result()
        if result is not None and result.ready:
            return replace(
                packet,
                shape_prior_points_m=np.ascontiguousarray(result.points_m, dtype=np.float32).reshape(-1, 3),
                shape_prior_colors_rgb_u8=np.ascontiguousarray(result.colors_rgb_u8, dtype=np.uint8).reshape(-1, 3),
                shape_prior_status=shape_prior_warmup.STATUS_READY,
                shape_prior_profile=profile,
            )
        return replace(
            packet,
            shape_prior_points_m=np.empty((0, 3), dtype=np.float32),
            shape_prior_colors_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            shape_prior_status=str(
                profile.get("shape_prior_status", shape_prior_warmup.STATUS_DISABLED)
            ),
            shape_prior_profile=profile,
        )

    def _maybe_start_shape_prior_from_pcd_result(
        self,
        result: PcdBuildResult,
        *,
        from_strict_pair: bool = False,
    ) -> bool:
        """Maybe start or update start shape prior from PCD result."""
        frame0_request = self._shape_prior_frame0_request_from_pcd_result(result)
        if frame0_request is None:
            return False
        submitted = self.shape_prior_manager.maybe_submit(frame0_request)
        if submitted:
            self._write_shape_prior_profile_json()
            self._status.emit(STAGE_SHAPE_PRIOR, "frame-0 submitted; generating shape prior")
        return bool(submitted)

    def _maybe_write_shape_prior_headless_result(self) -> None:
        """Maybe start or update write shape prior headless result."""
        result = self.shape_prior_manager.ready_result()
        if self.headless_capture_writer is not None and result is not None and result.ready and not self._shape_prior_written:
            self.headless_capture_writer.write_shape_prior_result(result)
            self._shape_prior_written = True
            self.shape_prior_manager.mark_gate_open()
            profile = self._shape_prior_profile_payload()
            self._write_shape_prior_profile_json(profile)
            self._status.emit(STAGE_WARMUP_READY, "shape prior ready; formal timeline open")
            return
        profile = self._shape_prior_profile_payload()
        if self.headless_capture_writer is not None:
            self.headless_capture_writer.update_metadata(profile)
        self._write_shape_prior_profile_json(profile)

    def _run_deferred_shape_prior_after_teardown(self) -> None:
        """Run deferred shape prior after teardown."""
        return

    def _wait_for_first_frame(self) -> FramePacket | None:
        """Wait for for first frame."""
        if self._lossless_enabled():
            return self.lossless_frame_queue.get(stop_event=self.stop_event)
        while not self.stop_event.is_set():
            frame = self.capture_slot.get_latest_after(-1)
            if frame is not None:
                return frame
            time.sleep(0.005)
        return None

    def _autocast_context(self, torch_module: Any) -> Any:
        """Return the autocast context."""
        if not str(self.args.device).startswith("cuda") or self.args.dtype == "float32":
            return nullcontext()
        dtype = torch_module.bfloat16 if self.args.dtype == "bfloat16" else torch_module.float16
        return torch_module.autocast("cuda", dtype=dtype)

    def _prune_edgetam_live_session(self, session: Any, *, current_frame_idx: int) -> None:
        """Prune edgetam live session."""
        keep_frames = int(DEFAULT_EDGETAM_LIVE_SESSION_KEEP_FRAMES)
        if keep_frames <= 0:
            return
        min_frame_idx = int(current_frame_idx) - keep_frames + 1

        processed_frames = getattr(session, "processed_frames", None)
        if isinstance(processed_frames, dict):
            for frame_idx in list(processed_frames):
                if int(frame_idx) < min_frame_idx:
                    processed_frames.pop(frame_idx, None)

        output_dict_per_obj = getattr(session, "output_dict_per_obj", None)
        if isinstance(output_dict_per_obj, dict):
            for output_dict in output_dict_per_obj.values():
                if not isinstance(output_dict, dict):
                    continue
                non_cond_outputs = output_dict.get("non_cond_frame_outputs")
                if isinstance(non_cond_outputs, dict):
                    for frame_idx in list(non_cond_outputs):
                        if int(frame_idx) < min_frame_idx:
                            non_cond_outputs.pop(frame_idx, None)

        frames_tracked_per_obj = getattr(session, "frames_tracked_per_obj", None)
        if isinstance(frames_tracked_per_obj, dict):
            for tracked_frames in frames_tracked_per_obj.values():
                if not isinstance(tracked_frames, dict):
                    continue
                for frame_idx in list(tracked_frames):
                    if int(frame_idx) < min_frame_idx:
                        tracked_frames.pop(frame_idx, None)

    def _run_segmentation_frame(
        self,
        *,
        hf_stream: Any,
        torch_module: Any,
        dtype: Any,
        model: Any,
        processor: Any,
        session: Any,
        frame: FramePacket,
        initial_masks: InitialMaskBundle,
        add_prompt: bool,
    ) -> MaskPacket:
        """Run segmentation frame."""
        image = main_warmup.bgr_to_pil_rgb(frame.color_bgr)
        inputs, preprocess_ms, preprocess_pre_sync_ms, preprocess_post_sync_ms = _time_runtime_ms(
            lambda: processor(images=image, device=self.args.device, return_tensors="pt"),
        )
        pixel_values = inputs.pixel_values[0].to(device=self.args.device, dtype=dtype)
        prompt_ms = 0.0
        with self._autocast_context(torch_module):
            if add_prompt:
                prompt_obj_ids: list[int] = []
                prompt_masks: list[np.ndarray] = []
                if controller_tracking_enabled(self.args):
                    prompt_obj_ids.append(HAND_A_ID)
                    prompt_masks.append(
                        np.asarray(initial_masks.hand_a_mask, dtype=bool)
                    )
                if object_tracking_enabled(self.args):
                    prompt_obj_ids.append(OBJECT_ID)
                    prompt_masks.append(
                        np.asarray(initial_masks.object_mask, dtype=bool)
                    )
                if controller_tracking_enabled(self.args):
                    prompt_obj_ids.append(HAND_B_ID)
                    prompt_masks.append(
                        np.asarray(initial_masks.hand_b_mask, dtype=bool)
                    )
                _unused, prompt_ms, prompt_pre_sync_ms, prompt_post_sync_ms = _time_runtime_ms(
                    lambda: processor.add_inputs_to_inference_session(
                        inference_session=session,
                        frame_idx=int(frame.seq),
                        obj_ids=prompt_obj_ids,
                        input_masks=prompt_masks,
                    ),
                )
            else:
                prompt_pre_sync_ms = 0.0
                prompt_post_sync_ms = 0.0
            output, wall_model_ms, cuda_event_model_ms, model_pre_sync_ms, model_post_sync_ms = _time_model_forward(
                lambda: model(inference_session=session, frame=pixel_values, frame_idx=int(frame.seq)),
            )
            post_masks, postprocess_ms, postprocess_pre_sync_ms, postprocess_post_sync_ms = _time_runtime_ms(
                lambda: processor.post_process_masks(
                    [output.pred_masks],
                    original_sizes=inputs.original_sizes,
                    binarize=False,
                )[0],
            )
        masks_by_id = extract_object_masks_from_hf_output(
            output,
            post_masks,
            mask_logit_threshold=float(self.args.edgetam_mask_logit_threshold),
        )
        missing = [obj_id for obj_id in active_object_ids(self.args) if obj_id not in masks_by_id]
        if missing:
            raise RuntimeError(f"HF output missing tracked object ids: {missing}")
        reference_mask = next(iter(masks_by_id.values()))
        object_mask = masks_by_id.get(OBJECT_ID)
        if object_mask is None:
            object_mask = np.zeros_like(reference_mask, dtype=bool)
        hand_a_mask = masks_by_id.get(HAND_A_ID)
        if hand_a_mask is None:
            hand_a_mask = np.zeros_like(reference_mask, dtype=bool)
        hand_b_mask = masks_by_id.get(HAND_B_ID)
        if hand_b_mask is None:
            hand_b_mask = np.zeros_like(reference_mask, dtype=bool)
        controller_mask = np.logical_or(hand_a_mask, hand_b_mask)
        self._prune_edgetam_live_session(session, current_frame_idx=int(output.frame_idx))
        process_done_s = time.perf_counter()
        timing = replace(
            frame.timing,
            preprocess_ms=preprocess_ms,
            prompt_ms=prompt_ms,
            model_ms=wall_model_ms,
            wall_model_ms=wall_model_ms,
            cuda_event_model_ms=cuda_event_model_ms,
            pre_sync_wait_ms=float(preprocess_pre_sync_ms + prompt_pre_sync_ms + model_pre_sync_ms + postprocess_pre_sync_ms),
            post_sync_wait_ms=float(preprocess_post_sync_ms + prompt_post_sync_ms + model_post_sync_ms + postprocess_post_sync_ms),
            postprocess_ms=postprocess_ms,
            mask_ms=float(preprocess_ms + prompt_ms + wall_model_ms + postprocess_ms),
        )
        return MaskPacket(
            seq=frame.seq,
            color_bgr=frame.color_bgr,
            depth_source=frame.depth_source,
            intrinsics=frame.intrinsics,
            depth_scale_m_per_unit=frame.depth_scale_m_per_unit,
            receive_perf_s=frame.receive_perf_s,
            process_done_perf_s=process_done_s,
            dropped_capture_frames=self.capture_slot.dropped_count,
            timing=timing,
            controller_mask=np.ascontiguousarray(controller_mask, dtype=bool),
            object_mask=np.ascontiguousarray(object_mask, dtype=bool),
            hand_a_mask=np.ascontiguousarray(hand_a_mask, dtype=bool),
            hand_b_mask=np.ascontiguousarray(hand_b_mask, dtype=bool),
            depth_u16=frame.depth_u16,
            ir_left_u8=frame.ir_left_u8,
            ir_right_u8=frame.ir_right_u8,
            k_ir_left=frame.k_ir_left,
            t_ir_left_to_color=frame.t_ir_left_to_color,
            k_color=frame.k_color,
            ir_baseline_m=frame.ir_baseline_m,
            source_timestamp_s=frame.source_timestamp_s,
            source_frame_index=frame.source_frame_index,
            source_step=frame.source_step,
        )

    def _headless_product_rows_gated(self) -> bool:
        """True while post-warmup frames must stay out of the chunk timeline.

        The gate carries its own deadline: --shape-prior-timeout-ms bounds how
        long formal rows may be withheld. On expiry rows resume so the chunk
        bridge's shape-prior wait/failure path reports loudly, instead of the
        row stream stalling silently on a hung prior.
        """
        writer = self.headless_capture_writer
        if writer is None or self._formal_timeline_gate_expired:
            return False
        profile = self._shape_prior_profile()
        gated = _formal_chunk_rows_gated(
            warmup_anchor_written=self._warmup_anchor_row_written,
            shape_prior_status=str(
                profile.get("shape_prior_status", shape_prior_warmup.STATUS_DISABLED)
            ),
        )
        if not gated:
            return False
        now_s = time.perf_counter()
        if self._formal_timeline_gate_started_s is None:
            self._formal_timeline_gate_started_s = now_s
        timeout_ms = int(
            getattr(
                self.args,
                "shape_prior_timeout_ms",
                shape_prior_warmup.DEFAULT_SHAPE_PRIOR_TIMEOUT_MS,
            )
        )
        if timeout_ms > 0 and (now_s - self._formal_timeline_gate_started_s) * 1000.0 >= float(timeout_ms):
            self._formal_timeline_gate_expired = True
            print(
                "[WARN] shape prior still not ready after --shape-prior-timeout-ms="
                f"{timeout_ms}; resuming formal chunk rows so the chunk bridge can "
                "surface the shape-prior wait/failure loudly.",
                flush=True,
            )
            return False
        return True


__all__ = ["_SegWarmupMixin"]
