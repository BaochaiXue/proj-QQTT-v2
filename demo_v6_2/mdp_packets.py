"""Shared dataclasses & packet types flowing between pipeline stages."""
from __future__ import annotations

from demo_v6_2.mdp_constants import *  # noqa: F401,F403

@dataclass(frozen=True)
class PipelineTiming:
    wait_ms: float = 0.0
    align_ms: float = 0.0
    frame_copy_ms: float = 0.0
    ffs_ms: float = 0.0
    ffs_align_ms: float = 0.0
    remote_rtt_ms: float = 0.0
    remote_server_total_ms: float = 0.0
    remote_request_kb: float = 0.0
    remote_response_kb: float = 0.0
    depth_convert_ms: float = 0.0
    preprocess_ms: float = 0.0
    prompt_ms: float = 0.0
    model_ms: float = 0.0
    wall_model_ms: float = 0.0
    cuda_event_model_ms: float = 0.0
    pre_sync_wait_ms: float = 0.0
    post_sync_wait_ms: float = 0.0
    postprocess_ms: float = 0.0
    mask_ms: float = 0.0
    pcd_mask_intersection_ms: float = 0.0
    pcd_select_ms: float = 0.0
    pcd_point_cap_ms: float = 0.0
    pcd_backproject_ms: float = 0.0
    pcd_color_gather_ms: float = 0.0
    pcd_ms: float = 0.0
    pcd_filter_ms: float = 0.0
    object_filter_ms: float = 0.0
    controller_filter_ms: float = 0.0
    receive_to_render_ms: float = 0.0


@dataclass(frozen=True)
class RealtimeCameraRuntime:
    pipeline: object
    align: object | None
    serial: str
    intrinsics: CameraIntrinsics
    depth_scale_m_per_unit: float
    k_color: np.ndarray
    k_ir_left: np.ndarray | None = None
    t_ir_left_to_color: np.ndarray | None = None
    ir_baseline_m: float = 0.0


@dataclass(frozen=True)
class FramePacket:
    seq: int
    color_bgr: np.ndarray
    depth_source: str
    intrinsics: CameraIntrinsics
    depth_scale_m_per_unit: float
    receive_perf_s: float
    timing: PipelineTiming
    depth_u16: np.ndarray | None = None
    ir_left_u8: np.ndarray | None = None
    ir_right_u8: np.ndarray | None = None
    k_ir_left: np.ndarray | None = None
    t_ir_left_to_color: np.ndarray | None = None
    k_color: np.ndarray | None = None
    ir_baseline_m: float = 0.0
    source_timestamp_s: float | None = None
    source_frame_index: int | None = None
    source_step: int | None = None


class LiveLatestFrameSampler:
    """Sample the latest live camera frame on a fixed output cadence."""

    def __init__(self, sample_fps: float) -> None:
        """Initialize LiveLatestFrameSampler."""
        fps = float(sample_fps)
        if fps <= 0.0:
            raise ValueError("live latest sampler FPS must be positive")
        self.period_s = 1.0 / fps
        self._next_sample_s: float | None = None
        self._pending_packet: FramePacket | None = None

    def start(self, *, first_publish_s: float) -> None:
        """Start fixed-cadence sampling after the first published frame."""
        self._next_sample_s = float(first_publish_s) + self.period_s
        self._pending_packet = None

    def put_latest(self, packet: FramePacket) -> None:
        """Store the newest live input frame."""
        if self._next_sample_s is None:
            raise RuntimeError("live latest sampler must be started before use")
        self._pending_packet = packet

    def pop_due(self, *, now_s: float) -> tuple[FramePacket, float] | None:
        """Return the pending packet if its fixed output tick is due."""
        if self._next_sample_s is None:
            return None
        if self._pending_packet is None or float(now_s) < self._next_sample_s:
            return None
        packet = self._pending_packet
        sample_s = self._next_sample_s
        self._pending_packet = None
        while self._next_sample_s <= float(now_s):
            self._next_sample_s += self.period_s
        return packet, sample_s


@dataclass(frozen=True)
class FatalWorkerError:
    stage: str
    exc_type: str
    message: str

    def log_message(self) -> str:
        """Format the worker failure for logs and HUD output."""
        return f"{self.stage} failed: {self.exc_type}: {self.message}"


@dataclass(frozen=True)
class RecordedRgbdFrameRef:
    step: int
    timestamp_s: float
    color_path: Path
    depth_path: Path | None = None
    ir_left_path: Path | None = None
    ir_right_path: Path | None = None


class _NoopPipeline:
    def stop(self) -> None:
        """Stop _NoopPipeline."""
        return


@dataclass(frozen=True)
class MaskPacket:
    seq: int
    color_bgr: np.ndarray
    depth_source: str
    intrinsics: CameraIntrinsics
    depth_scale_m_per_unit: float
    receive_perf_s: float
    process_done_perf_s: float
    dropped_capture_frames: int
    timing: PipelineTiming
    controller_mask: np.ndarray
    object_mask: np.ndarray
    hand_a_mask: np.ndarray | None = None
    hand_b_mask: np.ndarray | None = None
    depth_u16: np.ndarray | None = None
    ir_left_u8: np.ndarray | None = None
    ir_right_u8: np.ndarray | None = None
    k_ir_left: np.ndarray | None = None
    t_ir_left_to_color: np.ndarray | None = None
    k_color: np.ndarray | None = None
    ir_baseline_m: float = 0.0
    source_timestamp_s: float | None = None
    source_frame_index: int | None = None
    source_step: int | None = None


@dataclass(frozen=True)
class MaskedPcdPacket:
    seq: int
    controller_xyz_m: np.ndarray
    controller_colors_rgb_u8: np.ndarray
    object_xyz_m: np.ndarray
    object_colors_rgb_u8: np.ndarray
    intrinsics: CameraIntrinsics
    receive_perf_s: float
    process_done_perf_s: float
    dropped_capture_frames: int
    dropped_seg_frames: int
    timing: PipelineTiming
    filter_telemetry: PcdFilterTelemetry = field(default_factory=lambda: PcdFilterTelemetry())
    coordinate_frame: str = COORDINATE_FRAME
    source_timestamp_s: float | None = None
    source_frame_index: int | None = None
    source_step: int | None = None
    shape_prior_points_m: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    shape_prior_colors_rgb_u8: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.uint8))
    shape_prior_status: str = shape_prior_warmup.STATUS_DISABLED
    shape_prior_profile: dict[str, Any] = field(default_factory=dict)

    @property
    def controller_point_count(self) -> int:
        """Return the controller point count."""
        return int(self.controller_xyz_m.shape[0])

    @property
    def object_point_count(self) -> int:
        """Return the object point count."""
        return int(self.object_xyz_m.shape[0])

    @property
    def point_count(self) -> int:
        """Return the point count."""
        return self.controller_point_count + self.object_point_count

    @property
    def shape_prior_point_count(self) -> int:
        """Return the shape prior point count."""
        return int(np.asarray(self.shape_prior_points_m, dtype=np.float32).reshape(-1, 3).shape[0])


@dataclass(frozen=True)
class MarkerResidualAudit:
    pixels_yx: np.ndarray
    valid: np.ndarray
    violation: np.ndarray
    checked_count: int
    violation_count: int
    gate: str = TRACKER_MARKER_GATE_PCD_FILTER_RESIDUAL_TABLE_Z


def _fit_bool_array(values: np.ndarray, length: int, *, fill: bool = False) -> np.ndarray:
    """Fit a boolean vector to the requested length."""
    output = np.full((max(0, int(length)),), bool(fill), dtype=bool)
    arr = np.asarray(values, dtype=bool).reshape(-1)
    count = min(len(arr), len(output))
    if count:
        output[:count] = arr[:count]
    return output


def _remaining_query_class_counts(
    alive_mask: np.ndarray,
    *,
    query_is_object: np.ndarray,
    query_is_controller: np.ndarray,
    query_controller_instance_id: np.ndarray,
) -> tuple[int, int, int, int]:
    """Return the remaining query class counts."""
    alive = np.asarray(alive_mask, dtype=bool).reshape(-1)
    count = int(alive.shape[0])
    is_object = _fit_bool_array(query_is_object, count)
    is_controller = _fit_bool_array(query_is_controller, count)
    # Instance ids fitted to the alive-mask length (truncate or zero-pad), int analog of _fit_bool_array.
    instance_id = np.zeros((count,), dtype=np.int64)
    ids = np.asarray(query_controller_instance_id, dtype=np.int64).reshape(-1)
    fit_count = min(len(ids), count)
    if fit_count:
        instance_id[:fit_count] = ids[:fit_count]
    hand_a = alive & (instance_id == QUERY_CONTROLLER_INSTANCE_HAND_A)
    hand_b = alive & (instance_id == QUERY_CONTROLLER_INSTANCE_HAND_B)
    controller = alive & (is_controller | hand_a | hand_b)
    obj = alive & is_object & ~controller
    return (
        int(np.count_nonzero(obj)),
        int(np.count_nonzero(controller)),
        int(np.count_nonzero(hand_a)),
        int(np.count_nonzero(hand_b)),
    )


@dataclass(frozen=True)
class TrackerMarkerPacket:
    seq: int
    marker_xyz_m: np.ndarray
    marker_colors_rgb_u8: np.ndarray
    query_rgb_u8: np.ndarray
    query_points_yx: np.ndarray
    tracks_yx: np.ndarray
    visibility: np.ndarray
    query_is_object: np.ndarray
    query_is_controller: np.ndarray
    receive_perf_s: float
    process_done_perf_s: float
    query_count: int
    consistent_visible_count: int = 0
    model_ms: float = 0.0
    lift_ms: float = 0.0
    e2e_ms: float = 0.0
    backend: str = TRACKER_BACKEND_TAPNEXTPP
    display_scope: str = DEFAULT_TRACKER_DISPLAY_SCOPE
    query_indices: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    query_target_id: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    query_controller_instance_id: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    query_all_target_id: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    query_all_controller_instance_id: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    hand_a_query_count: int = 0
    hand_b_query_count: int = 0
    object_query_count: int = 0
    marker_pixels_yx: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int64))
    marker_residual_valid: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=bool))
    marker_residual_violation: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=bool))
    marker_residual_checked_count: int = 0
    marker_residual_violation_count: int = 0
    marker_residual_gate: str = TRACKER_MARKER_GATE_PCD_FILTER_RESIDUAL_TABLE_Z
    query_alive_mask: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=bool))
    remaining_query_count: int = -1
    remaining_object_query_count: int = -1
    remaining_controller_query_count: int = -1
    remaining_hand_a_query_count: int = -1
    remaining_hand_b_query_count: int = -1
    retired_query_count: int = -1
    all_tracks_yx: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    all_tracker_visibility: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    coordinate_frame: str = COORDINATE_FRAME

    def __post_init__(self) -> None:
        """Validate and normalize the dataclass state after initialization."""
        alive = np.asarray(self.query_alive_mask, dtype=bool).reshape(-1)
        query_count = max(0, int(self.query_count))
        if alive.size == 0 and query_count > 0:
            alive = np.ones((query_count,), dtype=bool)
        elif alive.size != query_count and query_count > 0:
            fitted = np.zeros((query_count,), dtype=bool)
            count = min(int(alive.size), query_count)
            if count:
                fitted[:count] = alive[:count]
            alive = fitted
        alive = np.ascontiguousarray(alive, dtype=bool)
        object.__setattr__(self, "query_alive_mask", alive)
        if int(self.remaining_query_count) < 0:
            object.__setattr__(self, "remaining_query_count", int(np.count_nonzero(alive)))
        if int(self.retired_query_count) < 0:
            object.__setattr__(
                self,
                "retired_query_count",
                max(0, query_count - int(np.count_nonzero(alive))),
            )
        if (
            int(self.remaining_object_query_count) < 0
            or int(self.remaining_controller_query_count) < 0
            or int(self.remaining_hand_a_query_count) < 0
            or int(self.remaining_hand_b_query_count) < 0
        ):
            object_count, controller_count, hand_a_count, hand_b_count = _remaining_query_class_counts(
                alive,
                query_is_object=np.empty((0,), dtype=bool),
                query_is_controller=np.empty((0,), dtype=bool),
                query_controller_instance_id=self.query_all_controller_instance_id,
            )
            if int(self.remaining_object_query_count) < 0:
                object.__setattr__(self, "remaining_object_query_count", object_count)
            if int(self.remaining_controller_query_count) < 0:
                object.__setattr__(self, "remaining_controller_query_count", controller_count)
            if int(self.remaining_hand_a_query_count) < 0:
                object.__setattr__(self, "remaining_hand_a_query_count", hand_a_count)
            if int(self.remaining_hand_b_query_count) < 0:
                object.__setattr__(self, "remaining_hand_b_query_count", hand_b_count)

    @property
    def marker_count(self) -> int:
        """Return the marker count."""
        return int(self.marker_xyz_m.shape[0])


def _formal_chunk_rows_gated(*, warmup_anchor_written: bool, shape_prior_status: str) -> bool:
    """design_spec.md warmup/formal timeline split.

    Rows always write until a chunk-ready warmup anchor row has landed (live
    RealSense can emit an invalid strict pair before color-aligned PCD is
    ready; the bridge trims those, so they must not consume the frame-0
    slot). After the anchor, frames processed while the shape prior is still
    computing stay OUT of the formal final_data chunk timeline (they keep
    feeding the trackers and the left preview, which pace by
    input_frames.jsonl). The first frame after the prior is ready becomes
    output frame 1, stitched directly after warmup frame 0 under the
    operator hold-still convention. Terminal states (ready/disabled/failed)
    lift the gate — a failed prior must surface through the chunk bridge's
    shape-prior error path instead of silently stalling the row stream.
    """
    if not warmup_anchor_written:
        return False
    return str(shape_prior_status) in (
        shape_prior_warmup.STATUS_PENDING,
        shape_prior_warmup.STATUS_RUNNING,
    )


def _full_tracker_arrays_for_prepared_frame(packet: TrackerMarkerPacket) -> tuple[np.ndarray, np.ndarray]:
    """Return the full tracker arrays for prepared frame."""
    query_count = int(np.asarray(packet.query_points_yx, dtype=np.float32).reshape(-1, 2).shape[0])
    all_tracks = np.asarray(packet.all_tracks_yx, dtype=np.float32).reshape(-1, 2)
    all_visibility = np.asarray(packet.all_tracker_visibility, dtype=bool).reshape(-1)
    if all_tracks.shape[0] == query_count and all_visibility.shape[0] == query_count:
        return (
            np.ascontiguousarray(all_tracks, dtype=np.float32),
            np.ascontiguousarray(all_visibility, dtype=bool),
        )

    active_tracks = np.asarray(packet.tracks_yx, dtype=np.float32).reshape(-1, 2)
    active_visibility = np.asarray(packet.visibility, dtype=bool).reshape(-1)
    if active_tracks.shape[0] == query_count and active_visibility.shape[0] == query_count:
        return (
            np.ascontiguousarray(active_tracks, dtype=np.float32),
            np.ascontiguousarray(active_visibility, dtype=bool),
        )

    indices = np.asarray(packet.query_indices, dtype=np.int64).reshape(-1)
    if indices.shape[0] != active_tracks.shape[0] or active_tracks.shape[0] != active_visibility.shape[0]:
        raise ValueError("sparse tracker packet must have query_indices, tracks_yx, and visibility with matching lengths")
    if np.any(indices < 0) or np.any(indices >= query_count):
        raise ValueError("tracker packet query_indices contains out-of-range values")
    tracks = np.zeros((query_count, 2), dtype=np.float32)
    visibility = np.zeros((query_count,), dtype=bool)
    tracks[indices] = active_tracks
    visibility[indices] = active_visibility
    return np.ascontiguousarray(tracks, dtype=np.float32), np.ascontiguousarray(visibility, dtype=bool)


@dataclass(frozen=True)
class PairedRenderPacket:
    seq: int
    pcd_packet: MaskedPcdPacket
    tracker_packet: TrackerMarkerPacket
    mask_packet: MaskPacket | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the dataclass state after initialization."""
        pcd_seq = int(self.pcd_packet.seq)
        tracker_seq = int(self.tracker_packet.seq)
        mask_seq = None if self.mask_packet is None else int(self.mask_packet.seq)
        seq = int(self.seq)
        if pcd_seq != tracker_seq or seq != pcd_seq or (mask_seq is not None and mask_seq != seq):
            raise ValueError(
                "strict same-seq render packet mismatch: "
                f"pair={seq} pcd={pcd_seq} tracker={tracker_seq} mask={mask_seq}"
            )


@dataclass(frozen=True)
class PcdBuildResult:
    packet: MaskedPcdPacket
    depth_m: np.ndarray | None
    mask_packet: MaskPacket
    controller_pcd_mask: np.ndarray | None = None
    object_pcd_mask: np.ndarray | None = None
    object_observation_mask: np.ndarray | None = None
    pcd_stride: int = 1
    pcd_mask_erode_pixels: int = 0
    object_pcd_mask_erode_pixels: int = 0
    controller_pcd_mask_erode_pixels: int = 0
    world_z_diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PcdFilterTelemetry:
    enabled: bool = False
    mode: str = PCD_FILTER_NONE
    render_using_filtered: bool = False
    filter_seq: int = -1
    filter_age_frames: int = 0
    filter_age_ms: float = 0.0
    filter_ms: float = 0.0
    object_filter_ms: float = 0.0
    controller_filter_ms: float = 0.0
    object_raw_points: int = 0
    object_cap_points: int = 0
    object_output_points: int = 0
    object_prefallback_points: int = 0
    object_raw_retain_ratio: float = 0.0
    object_fallback_reason: str = ""
    controller_raw_points: int = 0
    controller_cap_points: int = 0
    controller_output_points: int = 0
    controller_prefallback_points: int = 0
    controller_raw_retain_ratio: float = 0.0
    controller_fallback_reason: str = ""
    object_filter_cap: int = 0
    controller_filter_cap: int = 0
    filter_submit_fps: float = 0.0
    filter_output_fps: float = 0.0
    filter_queue_drop: int = 0
    filter_busy: bool = False


@dataclass(frozen=True)
class DepthProfilePacket:
    seq: int
    receive_perf_s: float
    process_done_perf_s: float
    dropped_capture_frames: int
    timing: PipelineTiming


@dataclass(frozen=True)
class PairedBuildResult:
    seq: int
    pcd_result: PcdBuildResult
    tracker_packet: TrackerMarkerPacket

    @property
    def render_packet(self) -> PairedRenderPacket:
        """Return the paired packet used by the renderer."""
        return PairedRenderPacket(
            seq=int(self.seq),
            pcd_packet=self.pcd_result.packet,
            tracker_packet=self.tracker_packet,
            mask_packet=self.pcd_result.mask_packet,
        )


__all__ = [
    "PipelineTiming",
    "RealtimeCameraRuntime",
    "FramePacket",
    "LiveLatestFrameSampler",
    "FatalWorkerError",
    "RecordedRgbdFrameRef",
    "_NoopPipeline",
    "MaskPacket",
    "MaskedPcdPacket",
    "MarkerResidualAudit",
    "_fit_bool_array",
    "_remaining_query_class_counts",
    "TrackerMarkerPacket",
    "_formal_chunk_rows_gated",
    "_full_tracker_arrays_for_prepared_frame",
    "PairedRenderPacket",
    "PcdBuildResult",
    "PcdFilterTelemetry",
    "DepthProfilePacket",
    "PairedBuildResult",
]
