"""Recorded RGB-D replay source and RealSense capture startup.

Pipeline questions Q1-Q6 (see PIPELINE.md): the camera starts here via
``_start_realsense_pipeline`` (live) or ``RecordedRgbdFrameSource`` (fake-live
replay); ``_start_realsense_pipeline`` applies a single ``--camera-fps`` (default
30) to every enabled RGB/depth stream, while the output/chunk cadence is the
separate ``--replay-fps`` (default 5). Each frame becomes a ``FramePacket`` whose
``seq`` is its frame id and whose ``source_timestamp_s`` / ``source_frame_index``
carry the true capture provenance.
"""
from __future__ import annotations

from demo_v6_2.mdp_constants import *  # noqa: F401,F403
from demo_v6_2.mdp_packets import FramePacket, PipelineTiming, RealtimeCameraRuntime, RecordedRgbdFrameRef, _NoopPipeline

class RecordedRgbdFrameSource:
    def __init__(
        self,
        case_path: str | Path,
        *,
        replay_fps: float = 0.0,
        camera_index: int = 0,
        depth_source: str = "realsense",
    ) -> None:
        """Initialize RecordedRgbdFrameSource."""
        self.case_path = _resolve_path(case_path)
        self.camera_index = int(camera_index)
        self.depth_source = str(depth_source)
        if self.depth_source not in DEPTH_SOURCES:
            raise ValueError(
                f"fake-live replay depth_source must be one of {DEPTH_SOURCES}"
            )
        self.requires_depth = self.depth_source == "realsense"
        self.requires_ir = self.depth_source == "ffs"
        self.metadata_path = self.case_path / "metadata.json"
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"recording metadata not found: {self.metadata_path}")
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"recording metadata is not valid JSON: {self.metadata_path}") from exc
        self.metadata: dict[str, Any] = metadata
        streams_present = {str(item) for item in metadata.get("streams_present", [])}
        if "color" not in streams_present:
            raise ValueError(
                "fake-live replay requires streams_present to include color"
            )
        if self.requires_depth and "depth" not in streams_present:
            raise ValueError(
                "RealSense fake-live replay requires streams_present to include depth"
            )
        if self.requires_ir and not {"ir_left", "ir_right"}.issubset(streams_present):
            raise ValueError("FFS fake-live replay requires streams_present to include ir_left and ir_right")
        recording_by_camera = metadata.get("recording")
        if not isinstance(recording_by_camera, dict):
            raise ValueError("recording metadata must contain a recording object")
        camera_key = str(self.camera_index)
        camera_recording = recording_by_camera.get(camera_key)
        if not isinstance(camera_recording, dict) or not camera_recording:
            raise ValueError(f"recording metadata has no frames for camera {self.camera_index}")
        self.k_color = self._camera_matrix(metadata, "K_color", fallback_key="intrinsics")
        self.intrinsics = CameraIntrinsics(
            fx=float(self.k_color[0, 0]),
            fy=float(self.k_color[1, 1]),
            cx=float(self.k_color[0, 2]),
            cy=float(self.k_color[1, 2]),
        )
        self.depth_scale_m_per_unit = self._camera_float(metadata, "depth_scale_m_per_unit")
        self.serial = self._camera_string(metadata, "serial_numbers", default=f"recording-cam{self.camera_index}")
        self.width, self.height = self._resolve_dimensions(metadata)
        self.recording_fps = self._resolve_recording_fps(metadata)
        self.effective_fps = self._resolve_replay_fps(float(replay_fps))
        self.k_ir_left: np.ndarray | None = None
        self.t_ir_left_to_color: np.ndarray | None = None
        self.ir_baseline_m = 0.0
        self.has_ir_stereo = False
        if {"ir_left", "ir_right"}.issubset(streams_present):
            try:
                self.k_ir_left = self._camera_matrix(metadata, "K_ir_left")
                self.t_ir_left_to_color = self._camera_transform(metadata, "T_ir_left_to_color")
                self.ir_baseline_m = self._camera_baseline(metadata)
                self.has_ir_stereo = True
            except ValueError:
                if self.requires_ir:
                    raise
        if self.requires_ir and not self.has_ir_stereo:
            raise ValueError("FFS fake-live replay requires IR stereo calibration in metadata")
        self.frames = self._build_frame_refs(camera_recording)
        self._recording_elapsed_s = self._build_recording_elapsed_s(self.frames)

    @property
    def frame_count(self) -> int:
        """Return the frame count."""
        return len(self.frames)

    @property
    def steps(self) -> list[int]:
        """Return the steps."""
        return [frame.step for frame in self.frames]

    def make_runtime(self) -> RealtimeCameraRuntime:
        """Create a replay runtime wrapper around recorded RGB-D frames."""
        return RealtimeCameraRuntime(
            pipeline=_NoopPipeline(),
            align=None,
            serial=self.serial,
            intrinsics=self.intrinsics,
            depth_scale_m_per_unit=self.depth_scale_m_per_unit,
            k_color=self.k_color,
            k_ir_left=self.k_ir_left,
            t_ir_left_to_color=self.t_ir_left_to_color,
            ir_baseline_m=float(self.ir_baseline_m),
        )

    def read_packet(
        self,
        *,
        seq: int,
        frame_index: int | None = None,
        wait_ms: float = 0.0,
        receive_perf_s: float | None = None,
        frame_copy_ms: float | None = None,
    ) -> FramePacket:
        """Read packet."""
        packet_seq = int(seq)
        source_index = packet_seq if frame_index is None else int(frame_index)
        if source_index < 0 or source_index >= len(self.frames):
            raise IndexError(
                f"fake-live replay frame_index {source_index} out of range for "
                f"{len(self.frames)} frames"
            )
        ref = self.frames[source_index]
        copy_start_s = time.perf_counter()
        color_bgr = self._load_color_bgr(ref.color_path)
        depth_u16 = self._load_depth_u16(ref.depth_path) if ref.depth_path is not None else None
        ir_left_u8 = self._load_gray_u8(ref.ir_left_path) if ref.ir_left_path is not None else None
        ir_right_u8 = self._load_gray_u8(ref.ir_right_path) if ref.ir_right_path is not None else None
        copy_done_s = time.perf_counter()
        if depth_u16 is not None and color_bgr.shape[:2] != depth_u16.shape:
            raise ValueError(
                f"recording color/depth shape mismatch for step {ref.step}: "
                f"{tuple(color_bgr.shape[:2])} vs {tuple(depth_u16.shape)}"
            )
        if (ir_left_u8 is None) != (ir_right_u8 is None):
            raise ValueError(f"recording IR pair is incomplete for step {ref.step}")
        if ir_left_u8 is not None and ir_left_u8.shape != ir_right_u8.shape:
            raise ValueError(
                f"recording IR left/right shape mismatch for step {ref.step}: "
                f"{tuple(ir_left_u8.shape)} vs {tuple(ir_right_u8.shape)}"
            )
        if tuple(color_bgr.shape[:2]) != (self.height, self.width):
            raise ValueError(
                f"recording frame shape {tuple(color_bgr.shape[:2])} does not match metadata "
                f"height/width {(self.height, self.width)} for step {ref.step}"
            )
        receive_s = copy_done_s if receive_perf_s is None else float(receive_perf_s)
        copy_ms = _elapsed_ms(copy_start_s, copy_done_s) if frame_copy_ms is None else float(frame_copy_ms)
        return FramePacket(
            seq=packet_seq,
            color_bgr=color_bgr,
            depth_source=self.depth_source,
            intrinsics=self.intrinsics,
            depth_scale_m_per_unit=self.depth_scale_m_per_unit,
            receive_perf_s=receive_s,
            timing=PipelineTiming(wait_ms=float(wait_ms), align_ms=0.0, frame_copy_ms=copy_ms),
            depth_u16=depth_u16,
            ir_left_u8=ir_left_u8,
            ir_right_u8=ir_right_u8,
            k_ir_left=self.k_ir_left if ir_left_u8 is not None else None,
            t_ir_left_to_color=self.t_ir_left_to_color if ir_left_u8 is not None else None,
            k_color=self.k_color,
            ir_baseline_m=float(self.ir_baseline_m) if ir_left_u8 is not None else 0.0,
            source_timestamp_s=float(ref.timestamp_s),
            source_frame_index=int(source_index),
            source_step=int(ref.step),
        )

    def read_preview_packet(
        self,
        *,
        seq: int,
        frame_index: int | None = None,
        wait_ms: float = 0.0,
        receive_perf_s: float | None = None,
    ) -> FramePacket:
        """Read preview packet."""
        packet_seq = int(seq)
        source_index = packet_seq if frame_index is None else int(frame_index)
        if source_index < 0 or source_index >= len(self.frames):
            raise IndexError(
                f"recording preview frame_index {source_index} out of range for {len(self.frames)} frames"
            )
        ref = self.frames[source_index]
        copy_start_s = time.perf_counter()
        color_bgr = self._load_color_bgr(ref.color_path)
        copy_done_s = time.perf_counter()
        if tuple(color_bgr.shape[:2]) != (self.height, self.width):
            raise ValueError(
                f"recording preview frame shape {tuple(color_bgr.shape[:2])} does not match metadata "
                f"height/width {(self.height, self.width)} for step {ref.step}"
            )
        receive_s = copy_done_s if receive_perf_s is None else float(receive_perf_s)
        return FramePacket(
            seq=packet_seq,
            color_bgr=color_bgr,
            depth_source=self.depth_source,
            intrinsics=self.intrinsics,
            depth_scale_m_per_unit=self.depth_scale_m_per_unit,
            receive_perf_s=receive_s,
            timing=PipelineTiming(
                wait_ms=float(wait_ms),
                align_ms=0.0,
                frame_copy_ms=_elapsed_ms(copy_start_s, copy_done_s),
            ),
            k_color=self.k_color,
            source_timestamp_s=float(ref.timestamp_s),
            source_frame_index=int(source_index),
            source_step=int(ref.step),
        )

    def _camera_matrix(self, metadata: dict[str, Any], key: str, *, fallback_key: str | None = None) -> np.ndarray:
        """Return the camera matrix."""
        values = metadata.get(key)
        if values is None and fallback_key is not None:
            values = metadata.get(fallback_key)
        if not isinstance(values, list) or self.camera_index >= len(values) or values[self.camera_index] is None:
            raise ValueError(f"recording metadata missing {key} for camera {self.camera_index}")
        matrix = np.asarray(values[self.camera_index], dtype=np.float32)
        if matrix.shape != (3, 3):
            raise ValueError(f"recording metadata {key}[{self.camera_index}] must be 3x3")
        if float(matrix[0, 0]) <= 0.0 or float(matrix[1, 1]) <= 0.0:
            raise ValueError(f"recording metadata {key}[{self.camera_index}] has non-positive focal length")
        return np.ascontiguousarray(matrix, dtype=np.float32)

    def _camera_transform(self, metadata: dict[str, Any], key: str) -> np.ndarray:
        """Return the camera transform."""
        values = metadata.get(key)
        if not isinstance(values, list) or self.camera_index >= len(values) or values[self.camera_index] is None:
            raise ValueError(f"recording metadata missing {key} for camera {self.camera_index}")
        matrix = np.asarray(values[self.camera_index], dtype=np.float32)
        if matrix.shape != (4, 4):
            raise ValueError(f"recording metadata {key}[{self.camera_index}] must be 4x4")
        return np.ascontiguousarray(matrix, dtype=np.float32)

    def _camera_baseline(self, metadata: dict[str, Any]) -> float:
        """Return the camera baseline."""
        values = metadata.get("ir_baseline_m")
        if isinstance(values, list) and self.camera_index < len(values) and values[self.camera_index] is not None:
            value = float(values[self.camera_index])
            if value <= 0.0:
                raise ValueError(f"recording metadata ir_baseline_m[{self.camera_index}] must be positive")
            return value
        transform = self._camera_transform(metadata, "T_ir_left_to_right")
        baseline = float(np.linalg.norm(transform[:3, 3]))
        if baseline <= 0.0:
            raise ValueError(f"recording metadata T_ir_left_to_right[{self.camera_index}] has non-positive baseline")
        return baseline

    def _camera_float(self, metadata: dict[str, Any], key: str) -> float:
        """Return the camera float."""
        values = metadata.get(key)
        if not isinstance(values, list) or self.camera_index >= len(values) or values[self.camera_index] is None:
            raise ValueError(f"recording metadata missing {key} for camera {self.camera_index}")
        value = float(values[self.camera_index])
        if value <= 0.0:
            raise ValueError(f"recording metadata {key}[{self.camera_index}] must be positive")
        return value

    def _camera_string(self, metadata: dict[str, Any], key: str, *, default: str) -> str:
        """Return the camera string."""
        values = metadata.get(key)
        if isinstance(values, list) and self.camera_index < len(values) and values[self.camera_index] is not None:
            return str(values[self.camera_index])
        return default

    def _resolve_dimensions(self, metadata: dict[str, Any]) -> tuple[int, int]:
        """Resolve dimensions."""
        wh = metadata.get("WH")
        if not isinstance(wh, list) or len(wh) != 2:
            raise ValueError("recording metadata missing WH")
        width = int(wh[0])
        height = int(wh[1])
        if width <= 0 or height <= 0:
            raise ValueError("recording metadata WH must be positive")
        return width, height

    def _resolve_recording_fps(self, metadata: dict[str, Any]) -> float:
        """Resolve the recording FPS from case metadata."""
        try:
            fps = float(metadata.get("fps", 0.0))
        except (TypeError, ValueError):
            fps = 0.0
        return fps if fps > 0.0 else 30.0

    def _resolve_replay_fps(self, replay_fps: float) -> float:
        """Resolve the effective replay FPS for fake-live playback."""
        return float(replay_fps) if float(replay_fps) > 0.0 else float(self.recording_fps)

    def _build_recording_elapsed_s(self, frames: list[RecordedRgbdFrameRef]) -> np.ndarray:
        """Build recording elapsed s."""
        timestamps = np.asarray([float(frame.timestamp_s) for frame in frames], dtype=np.float64)
        if len(timestamps) and np.isfinite(timestamps).all() and np.all(np.diff(timestamps) >= 0.0):
            return np.ascontiguousarray(timestamps - timestamps[0], dtype=np.float64)
        frame_indices = np.arange(len(frames), dtype=np.float64)
        return np.ascontiguousarray(frame_indices / float(self.recording_fps), dtype=np.float64)

    def source_index_for_recording_elapsed_s(self, elapsed_s: float) -> int:
        """Return the source frame index nearest a recording elapsed time."""
        if len(self.frames) <= 1:
            return 0
        elapsed = max(0.0, float(elapsed_s))
        index = int(np.searchsorted(self._recording_elapsed_s, elapsed + 1e-9, side="right") - 1)
        return max(0, min(index, len(self.frames) - 1))

    def _build_frame_refs(self, camera_recording: dict[str, Any]) -> list[RecordedRgbdFrameRef]:
        """Build frame refs."""
        refs: list[RecordedRgbdFrameRef] = []
        color_dir = self.case_path / "color" / str(self.camera_index)
        depth_dir = self.case_path / "depth" / str(self.camera_index)
        ir_left_dir = self.case_path / "ir_left" / str(self.camera_index)
        ir_right_dir = self.case_path / "ir_right" / str(self.camera_index)
        for step_text, timestamp in sorted(camera_recording.items(), key=lambda item: int(item[0])):
            step = int(step_text)
            color_path = color_dir / f"{step}.png"
            depth_path = depth_dir / f"{step}.npy"
            if not color_path.is_file():
                raise FileNotFoundError(f"recording color frame missing: {color_path}")
            if self.requires_depth and not depth_path.is_file():
                raise FileNotFoundError(f"recording depth frame missing: {depth_path}")
            ir_left_path = ir_left_dir / f"{step}.png"
            ir_right_path = ir_right_dir / f"{step}.png"
            if self.requires_ir:
                if not ir_left_path.is_file():
                    raise FileNotFoundError(f"recording IR left frame missing: {ir_left_path}")
                if not ir_right_path.is_file():
                    raise FileNotFoundError(f"recording IR right frame missing: {ir_right_path}")
            optional_ir_pair = self.has_ir_stereo and ir_left_path.is_file() and ir_right_path.is_file()
            refs.append(
                RecordedRgbdFrameRef(
                    step=step,
                    timestamp_s=float(timestamp),
                    color_path=color_path,
                    depth_path=depth_path if self.requires_depth else None,
                    ir_left_path=ir_left_path if self.requires_ir or optional_ir_pair else None,
                    ir_right_path=ir_right_path if self.requires_ir or optional_ir_pair else None,
                )
            )
        if not refs:
            raise ValueError(f"recording has no complete fake-live frames for camera {self.camera_index}")
        return refs

    def _load_color_bgr(self, path: Path) -> np.ndarray:
        """Load color BGR."""
        try:
            from PIL import Image

            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
        except Exception as exc:
            raise ValueError(f"failed to load recording color frame {path}: {exc}") from exc
        return np.ascontiguousarray(rgb[:, :, ::-1], dtype=np.uint8)

    def _load_depth_u16(self, path: Path) -> np.ndarray:
        """Load depth u16."""
        try:
            depth = np.load(path)
        except Exception as exc:
            raise ValueError(f"failed to load recording depth frame {path}: {exc}") from exc
        depth_u16 = np.asarray(depth)
        if depth_u16.ndim != 2:
            raise ValueError(f"recording depth frame must be 2D: {path}")
        if depth_u16.dtype != np.uint16:
            depth_u16 = depth_u16.astype(np.uint16, copy=False)
        return np.ascontiguousarray(depth_u16)

    def _load_gray_u8(self, path: Path) -> np.ndarray:
        """Load gray u8."""
        try:
            from PIL import Image

            with Image.open(path) as image:
                gray = np.asarray(image.convert("L"))
        except Exception as exc:
            raise ValueError(f"failed to load recording IR frame {path}: {exc}") from exc
        return np.ascontiguousarray(gray, dtype=np.uint8)


def _start_realsense_pipeline(args: argparse.Namespace) -> RealtimeCameraRuntime:
    """Start realsense pipeline."""
    rs = load_realsense_module()
    width, height = parse_profile(DEFAULT_PROFILE)
    serial = resolve_serial(rs, args.serial)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, int(args.fps))
    if args.depth_source == "ffs":
        config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, int(args.fps))
        config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, int(args.fps))
    if args.depth_source == "realsense":
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, int(args.fps))
    profile = pipeline.start(config)
    try:
        apply_emitter(profile, args.emitter, rs)
        # Fixed RGB exposure/gain only stick after auto-exposure is disabled on the sensor.
        exposure = getattr(args, "color_exposure", None)
        gain = getattr(args, "color_gain", None)
        if exposure is not None or gain is not None:
            color_sensor = profile.get_device().first_color_sensor()
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                if not color_sensor.supports(rs.option.exposure):
                    raise RuntimeError("RealSense RGB sensor does not support exposure control")
                color_sensor.set_option(rs.option.exposure, float(exposure))
            if gain is not None:
                if not color_sensor.supports(rs.option.gain):
                    raise RuntimeError("RealSense RGB sensor does not support gain control")
                color_sensor.set_option(rs.option.gain, float(gain))
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = camera_intrinsics_from_rs(color_stream.get_intrinsics())
        k_color = rs_intrinsics_to_matrix(color_stream.get_intrinsics())
        if args.depth_source == "ffs":
            ir_left_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
            ir_right_profile = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
            ir_left_to_right = ir_left_profile.get_extrinsics_to(ir_right_profile)
            ir_left_to_color = ir_left_profile.get_extrinsics_to(color_stream)
            if args.depth_source == "realsense":
                align = rs.align(rs.stream.color)
                return RealtimeCameraRuntime(
                    pipeline=pipeline,
                    align=align,
                    serial=serial,
                    intrinsics=intrinsics,
                    depth_scale_m_per_unit=depth_scale,
                    k_color=k_color,
                    k_ir_left=rs_intrinsics_to_matrix(ir_left_profile.get_intrinsics()),
                    t_ir_left_to_color=rs_extrinsics_to_matrix(ir_left_to_color),
                    ir_baseline_m=rs_translation_norm(ir_left_to_right),
                )
            return RealtimeCameraRuntime(
                pipeline=pipeline,
                align=None,
                serial=serial,
                intrinsics=intrinsics,
                depth_scale_m_per_unit=depth_scale,
                k_color=k_color,
                k_ir_left=rs_intrinsics_to_matrix(ir_left_profile.get_intrinsics()),
                t_ir_left_to_color=rs_extrinsics_to_matrix(ir_left_to_color),
                ir_baseline_m=rs_translation_norm(ir_left_to_right),
            )
        if args.depth_source == "none":
            return RealtimeCameraRuntime(
                pipeline=pipeline,
                align=None,
                serial=serial,
                intrinsics=intrinsics,
                depth_scale_m_per_unit=depth_scale,
                k_color=k_color,
            )
        align = rs.align(rs.stream.color)
    except Exception:
        pipeline.stop()
        raise
    return RealtimeCameraRuntime(
        pipeline=pipeline,
        align=align,
        serial=serial,
        intrinsics=intrinsics,
        depth_scale_m_per_unit=depth_scale,
        k_color=k_color,
    )


__all__ = [
    "RecordedRgbdFrameSource",
    "_start_realsense_pipeline",
]
