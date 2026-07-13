"""Shared dataclasses & packet types flowing between pipeline stages."""

from __future__ import annotations

from demo_v6_2.mdp_constants import *  # noqa: F401,F403


@dataclass(frozen=True)
class PipelineTiming:
    wait_ms: float = 0.0  # Time waiting for the source frame.
    align_ms: float = 0.0  # Time aligning native depth to color.
    frame_copy_ms: float = 0.0  # Time copying source arrays into a packet.
    ffs_ms: float = 0.0  # Total Fast Foundation Stereo inference time.
    ffs_align_ms: float = 0.0  # Time aligning FFS depth to the color camera.
    remote_rtt_ms: float = 0.0  # Round-trip time for a remote FFS request.
    remote_server_total_ms: float = 0.0  # Processing time reported by the server.
    remote_request_kb: float = 0.0  # Encoded remote request size in KiB.
    remote_response_kb: float = 0.0  # Encoded remote response size in KiB.
    depth_convert_ms: float = 0.0  # Time converting depth into metric units.
    preprocess_ms: float = 0.0  # Model input preprocessing time.
    prompt_ms: float = 0.0  # Segmentation prompt preparation time.
    model_ms: float = 0.0  # Primary model execution time.
    wall_model_ms: float = 0.0  # Wall-clock model execution time.
    cuda_event_model_ms: float = 0.0  # CUDA-event model execution time.
    pre_sync_wait_ms: float = 0.0  # Synchronization wait before model execution.
    post_sync_wait_ms: float = 0.0  # Synchronization wait after model execution.
    postprocess_ms: float = 0.0  # Model output postprocessing time.
    mask_ms: float = 0.0  # Time constructing semantic masks.
    pcd_ms: float = 0.0  # Canonical geometry + PT mask + class PCD time.
    receive_to_render_ms: float = 0.0  # End-to-end receive-to-render latency.


@dataclass(frozen=True)
class RealtimeCameraRuntime:
    pipeline: object | None  # RealSense pipeline; None for fake-live replay.
    align: object | None  # Native-depth-to-color aligner when required.
    serial: str  # Physical or recorded camera serial number.
    intrinsics: CameraIntrinsics  # Color-camera fx, fy, cx, and cy in pixels.
    depth_scale_m_per_unit: float  # Meters represented by one depth integer unit.
    k_color: np.ndarray  # Color-camera intrinsic matrix, shape (3, 3).
    k_ir_left: np.ndarray | None = None  # Left-IR intrinsic matrix, shape (3, 3).
    t_ir_left_to_color: np.ndarray | None = None  # Left-IR-to-color 4x4 transform.
    ir_baseline_m: float = 0.0  # Left-to-right IR baseline in meters.


@dataclass(frozen=True)
class FramePacket:
    seq: int  # Monotonic packet sequence within the current run.
    color_bgr: np.ndarray  # Color image, uint8 shape (H, W, 3) in BGR order.
    depth_source: str  # Selected depth backend: realsense, ffs, or none.
    intrinsics: CameraIntrinsics  # Color-camera fx, fy, cx, and cy in pixels.
    depth_scale_m_per_unit: float  # Meters represented by one depth integer unit.
    receive_perf_s: float  # Local monotonic time when the frame was received.
    timing: PipelineTiming  # Per-stage latency measurements for this frame.
    depth_u16: np.ndarray | None = None  # Native depth image, uint16 shape (H, W).
    ir_left_u8: np.ndarray | None = None  # Left IR image, uint8 shape (H, W).
    ir_right_u8: np.ndarray | None = None  # Right IR image, uint8 shape (H, W).
    k_ir_left: np.ndarray | None = None  # Left-IR intrinsic matrix, shape (3, 3).
    t_ir_left_to_color: np.ndarray | None = None  # Left-IR-to-color 4x4 transform.
    k_color: np.ndarray | None = None  # Color-camera intrinsic matrix, shape (3, 3).
    ir_baseline_m: float = 0.0  # Left-to-right IR baseline in meters.
    source_timestamp_s: float | None = None  # Original capture timestamp in seconds.
    source_frame_index: int | None = None  # Zero-based index in the source stream.
    source_step: int | None = None  # Original recording step or filename stem.


@dataclass(frozen=True)
class FatalWorkerError:
    stage: str  # Pipeline stage whose worker failed.
    exc_type: str  # Exception class name.
    message: str  # Human-readable exception message.

    def log_message(self) -> str:
        """Format the worker failure for logs and HUD output."""
        return f"{self.stage} failed: {self.exc_type}: {self.message}"


@dataclass(frozen=True)
class RecordedRgbdFrameRef:
    step: int  # Original recording step and file stem.
    timestamp_s: float  # Source capture timestamp in seconds.
    color_path: Path  # Path to the RGB PNG for this step.
    depth_path: Path | None = None  # Path to the native uint16 depth array.
    ir_left_path: Path | None = None  # Path to the left-IR grayscale PNG.
    ir_right_path: Path | None = None  # Path to the right-IR grayscale PNG.


@dataclass(frozen=True)
class MaskPacket:
    seq: int  # Source FramePacket sequence preserved through segmentation.
    color_bgr: np.ndarray  # Color image, uint8 shape (H, W, 3) in BGR order.
    depth_source: str  # Selected depth backend: realsense, ffs, or none.
    intrinsics: CameraIntrinsics  # Color-camera fx, fy, cx, and cy in pixels.
    depth_scale_m_per_unit: float  # Meters represented by one depth integer unit.
    receive_perf_s: float  # Local monotonic time when the frame was received.
    process_done_perf_s: float  # Monotonic time when segmentation completed.
    dropped_capture_frames: int  # Capture frames skipped before this packet.
    timing: PipelineTiming  # Accumulated latency measurements for this frame.
    controller_mask: np.ndarray  # Controller boolean mask, shape (H, W).
    object_mask: np.ndarray  # Object boolean mask, shape (H, W).
    hand_a_mask: np.ndarray | None = None  # First-hand mask, shape (H, W).
    hand_b_mask: np.ndarray | None = None  # Second-hand mask, shape (H, W).
    depth_u16: np.ndarray | None = None  # Native depth image, uint16 shape (H, W).
    ir_left_u8: np.ndarray | None = None  # Left IR image, uint8 shape (H, W).
    ir_right_u8: np.ndarray | None = None  # Right IR image, uint8 shape (H, W).
    k_ir_left: np.ndarray | None = None  # Left-IR intrinsic matrix, shape (3, 3).
    t_ir_left_to_color: np.ndarray | None = None  # Left-IR-to-color 4x4 transform.
    k_color: np.ndarray | None = None  # Color-camera intrinsic matrix, shape (3, 3).
    ir_baseline_m: float = 0.0  # Left-to-right IR baseline in meters.
    source_timestamp_s: float | None = None  # Original capture timestamp in seconds.
    source_frame_index: int | None = None  # Zero-based index in the source stream.
    source_step: int | None = None  # Original recording step or filename stem.


@dataclass(frozen=True)
class ProcessedFramePacket:
    """Canonical origin-style geometry and masks for one formal frame."""

    seq: int
    mask_packet: MaskPacket  # Contains the canonical processed masks.
    depth_m: np.ndarray  # Color-aligned metric depth, shape (H, W).
    depth_valid_mask: np.ndarray  # Origin depth gate, shape (H, W).
    pcd_points: np.ndarray  # Dense world grid, shape (1, H, W, 3).
    pcd_colors: np.ndarray  # Dense RGB grid, shape (1, H, W, 3).

    def __post_init__(self) -> None:
        """Reject mixed-sequence or shape-inconsistent canonical frames."""
        if int(self.seq) != int(self.mask_packet.seq):
            raise ValueError(
                "processed frame sequence mismatch: "
                f"frame={self.seq} mask={self.mask_packet.seq}"
            )
        depth_shape = np.asarray(self.depth_m).shape
        if len(depth_shape) != 2:
            raise ValueError(f"processed depth must have shape HxW; got {depth_shape}")
        if np.asarray(self.depth_valid_mask).shape != depth_shape:
            raise ValueError("processed depth-valid mask must match depth shape")
        depth_valid = np.asarray(self.depth_valid_mask, dtype=bool)
        expected_grid_shape = (1, *depth_shape, 3)
        if np.asarray(self.pcd_points).shape != expected_grid_shape:
            raise ValueError(
                f"processed PCD points must have shape {expected_grid_shape}"
            )
        if np.asarray(self.pcd_colors).shape != expected_grid_shape:
            raise ValueError(
                f"processed PCD colors must have shape {expected_grid_shape}"
            )
        points_grid = np.asarray(self.pcd_points, dtype=np.float32)[0]
        for name, mask in (
            ("object", self.mask_packet.object_mask),
            ("controller", self.mask_packet.controller_mask),
        ):
            mask_bool = np.asarray(mask, dtype=bool)
            if mask_bool.shape != depth_shape:
                raise ValueError(
                    f"processed {name} mask shape {mask_bool.shape} "
                    f"does not match depth shape {depth_shape}"
                )
            if not np.any(mask_bool):
                raise ValueError(f"processed {name} mask is empty")
            if np.any(mask_bool & ~depth_valid):
                raise ValueError(f"processed {name} mask contains depth-invalid pixels")
            if not np.isfinite(points_grid[mask_bool]).all():
                raise ValueError(f"processed {name} mask contains non-finite 3D points")
        controller = np.asarray(self.mask_packet.controller_mask, dtype=bool)
        for name, hand_mask in (
            ("hand_a", self.mask_packet.hand_a_mask),
            ("hand_b", self.mask_packet.hand_b_mask),
        ):
            if hand_mask is None:
                continue
            hand = np.asarray(hand_mask, dtype=bool)
            if hand.shape != depth_shape:
                raise ValueError(
                    f"processed {name} mask shape {hand.shape} does not match "
                    f"depth shape {depth_shape}"
                )
            if np.any(hand & ~controller):
                raise ValueError(
                    f"processed {name} mask must be a controller-mask subset"
                )


@dataclass(frozen=True)
class MaskedPcdPacket:
    seq: int  # Source packet sequence preserved through PCD construction.
    controller_xyz_m: np.ndarray  # Controller points in meters, shape (N, 3).
    controller_colors_rgb_u8: np.ndarray  # Controller RGB colors, shape (N, 3).
    object_xyz_m: np.ndarray  # Object points in meters, shape (M, 3).
    object_colors_rgb_u8: np.ndarray  # Object RGB colors, shape (M, 3).
    intrinsics: CameraIntrinsics  # Color-camera fx, fy, cx, and cy in pixels.
    receive_perf_s: float  # Monotonic time when the source frame was received.
    process_done_perf_s: float  # Monotonic time when PCD processing completed.
    dropped_capture_frames: int  # Capture frames skipped before this packet.
    dropped_seg_frames: int  # Segmentation packets skipped before this packet.
    timing: PipelineTiming  # Accumulated latency measurements for this frame.
    coordinate_frame: str = COORDINATE_FRAME  # Coordinate frame of all XYZ arrays.
    source_timestamp_s: float | None = None  # Original capture timestamp in seconds.
    source_frame_index: int | None = None  # Zero-based index in the source stream.
    source_step: int | None = None  # Original recording step or filename stem.
    shape_prior_points_m: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )  # Shape-prior points in meters, shape (P, 3).
    shape_prior_colors_rgb_u8: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.uint8)
    )  # Shape-prior RGB colors, shape (P, 3).
    shape_prior_status: str = (
        shape_prior_warmup.STATUS_DISABLED
    )  # Current shape-prior warmup state.
    shape_prior_profile: dict[str, Any] = field(
        default_factory=dict
    )  # Shape-prior profiling and diagnostic values.

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
        return int(
            np.asarray(self.shape_prior_points_m, dtype=np.float32)
            .reshape(-1, 3)
            .shape[0]
        )


@dataclass(frozen=True)
class TrackerMarkerPacket:
    seq: int  # Source packet sequence preserved through tracking.
    marker_xyz_m: np.ndarray  # Visible marker points in meters, shape (N, 3).
    marker_colors_rgb_u8: np.ndarray  # Visible marker RGB colors, shape (N, 3).
    query_rgb_u8: np.ndarray  # Frozen query-point RGB colors, shape (Q, 3).
    query_points_yx: np.ndarray  # Frozen query image coordinates, shape (Q, 2).
    tracks_yx: np.ndarray  # Current active track coordinates, shape (A, 2).
    visibility: np.ndarray  # Current visibility flags for active tracks.
    query_is_object: np.ndarray  # Object-class flags for active queries.
    query_is_controller: np.ndarray  # Controller-class flags for active queries.
    receive_perf_s: float  # Monotonic time when the source frame was received.
    process_done_perf_s: float  # Monotonic time when tracking completed.
    query_count: int  # Total frozen query count.
    consistent_visible_count: int = 0  # Queries visible under consistency checks.
    model_ms: float = 0.0  # Tracker model execution time in milliseconds.
    lift_ms: float = 0.0  # Time lifting 2D tracks into 3D markers.
    e2e_ms: float = 0.0  # End-to-end tracker latency in milliseconds.
    backend: str = TRACKER_BACKEND_TAPNEXTPP  # Tracker backend identifier.
    display_scope: str = DEFAULT_TRACKER_DISPLAY_SCOPE  # Marker display policy.
    query_indices: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )  # Full-query indices represented by sparse active arrays.
    query_target_id: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )  # Target IDs for active queries.
    query_controller_instance_id: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )  # Controller instance IDs for active queries.
    hand_a_query_count: int = 0  # Frozen query count assigned to the first hand.
    hand_b_query_count: int = 0  # Frozen query count assigned to the second hand.
    object_query_count: int = 0  # Frozen query count assigned to the object.
    all_tracks_yx: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )  # Current coordinates for the complete query table, shape (Q, 2).
    all_tracker_visibility: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float32)
    )  # Current visibility values for the complete query table.
    all_observation_visibility: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=bool)
    )  # Full-query visibility after processed-mask and depth gates.
    coordinate_frame: str = COORDINATE_FRAME  # Coordinate frame of marker_xyz_m.

    @property
    def marker_count(self) -> int:
        """Return the marker count."""
        return int(self.marker_xyz_m.shape[0])


def _formal_chunk_rows_gated(
    *, warmup_anchor_written: bool, shape_prior_status: str
) -> bool:
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


def _full_tracker_arrays_for_prepared_frame(
    packet: TrackerMarkerPacket,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the full tracker arrays for prepared frame."""
    query_count = int(
        np.asarray(packet.query_points_yx, dtype=np.float32).reshape(-1, 2).shape[0]
    )
    all_tracks = np.asarray(packet.all_tracks_yx, dtype=np.float32).reshape(-1, 2)
    all_visibility = np.asarray(packet.all_observation_visibility, dtype=bool).reshape(
        -1
    )
    if all_tracks.shape[0] != query_count or all_visibility.shape[0] != query_count:
        raise ValueError(
            "prepared frame requires full processed tracker arrays: "
            f"queries={query_count} tracks={all_tracks.shape[0]} "
            f"visibility={all_visibility.shape[0]}"
        )
    return (
        np.ascontiguousarray(all_tracks, dtype=np.float32),
        np.ascontiguousarray(all_visibility, dtype=bool),
    )


@dataclass(frozen=True)
class PcdBuildResult:
    packet: MaskedPcdPacket  # Materialized point-cloud packet.
    processed_frame: ProcessedFramePacket  # Shared canonical frame.


@dataclass(frozen=True)
class DepthProfilePacket:
    seq: int  # Source packet sequence preserved through depth profiling.
    receive_perf_s: float  # Monotonic time when the source frame was received.
    process_done_perf_s: float  # Monotonic time when depth processing completed.
    dropped_capture_frames: int  # Capture frames skipped before this packet.
    timing: PipelineTiming  # Accumulated depth-stage latency measurements.


@dataclass(frozen=True)
class PairedBuildResult:
    seq: int  # Sequence shared by the PCD and tracker results.
    pcd_result: PcdBuildResult  # Point-cloud build result for this sequence.
    tracker_packet: TrackerMarkerPacket  # Tracker result for this sequence.

    def __post_init__(self) -> None:
        """Reject PCD, mask, and tracker results from different frames."""
        seq = int(self.seq)
        pcd_seq = int(self.pcd_result.packet.seq)
        mask_seq = int(self.pcd_result.processed_frame.mask_packet.seq)
        tracker_seq = int(self.tracker_packet.seq)
        if seq != pcd_seq or seq != mask_seq or seq != tracker_seq:
            raise ValueError(
                "strict same-seq build result mismatch: "
                f"pair={seq} pcd={pcd_seq} mask={mask_seq} tracker={tracker_seq}"
            )


__all__ = [
    "PipelineTiming",
    "RealtimeCameraRuntime",
    "FramePacket",
    "FatalWorkerError",
    "RecordedRgbdFrameRef",
    "MaskPacket",
    "ProcessedFramePacket",
    "MaskedPcdPacket",
    "TrackerMarkerPacket",
    "_formal_chunk_rows_gated",
    "_full_tracker_arrays_for_prepared_frame",
    "PcdBuildResult",
    "DepthProfilePacket",
    "PairedBuildResult",
]
