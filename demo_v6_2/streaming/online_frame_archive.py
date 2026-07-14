"""Per-frame RGB-D archive for the published online stream.

Pipeline questions Q21-Q23 (see PIPELINE.md): this is the raw RGB-D side of the
schema the training side reads. Alongside ``online_data/chunks`` (which stays
byte-identical to the existing contract), every published online frame k also
lands as raw sensor products in the offline PhysTwin case layout
(``color/0/{k}.png``, ``depth/0/{k}.npy`` uint16 mm, ``calibrate.pkl``,
``metadata.json``, ``enhance_metadata.json``). In the live stream path the
color/depth files are written in REAL TIME (``stream_frame``: one fsynced pair
the moment each capture row is accepted, at frame cadence) — they do not wait
for chunk materialization. ``metadata.json`` is still rewritten atomically only
after the owning chunk commits, so ``frame_num`` never points at a file that
does not exist, never counts an uncommitted frame, and the ``phystwin_shen``
trainer can start reading from the first committed chunk; live consumers that
want the sub-chunk latency watch the color/depth directories directly. When the
stream ends, streamed frames whose chunk never committed are deleted
(``discard_streamed_tail``), so the final tree contains exactly the committed
frames. Layout::

    online_data/
        color/0/{k}.png        # original RGB of the frame the chunk consumed
        depth/0/{k}.npy        # (H, W) uint16 millimeters, invalid = 0
        calibrate.pkl          # [4x4 camera-to-world] (single camera)
        metadata.json          # intrinsics/WH/frame_num/serial_numbers
        enhance_metadata.json  # online-frame -> source-frame mapping table

Filenames use the continuous online frame index (0..N-1) — exactly the frames
that were processed and published, including the chunk-0 frame-0 shape-prior
warmup anchor. This differs from ``capture/input_rgb``, which logs every
received input frame whether or not it entered a chunk.

Depth format is identical for every backend: RealSense raw units at the
standard 0.001 m/unit scale round-trip bit-exactly through
``depth_m_to_mm_u16`` (an effective direct copy), and FFS-generated float
meters quantize through the same conversion. ``metadata.json``/``calibrate.pkl``
mirror the recording-case schema that data_process_origin/data_process_pcd.py
reads: ``np.load(depth)/1000.0``, ``cv2.imread`` color, ``intrinsics`` shaped
(num_cam, 3, 3), and contiguous integer filenames ``0..frame_num-1``.

A capture row without its prepared frame is rejected before materialization.
Legacy prepared NPZs without depth and misaligned indices also fail fast, so an
online chunk is never committed without its archived frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from demo_v6_2.phystwin_strict_product import PreparedPhysTwinFrame
from demo_v6_2.utils.atomic_io import (
    atomic_json_dump,
    atomic_open,
    atomic_pickle_dump,
)

CAMERA_INDEX = 0


class OnlineFrameArchiveError(RuntimeError):
    """A published chunk frame could not be archived as raw color/depth."""


def _square_matrix(value: Any, *, size: int, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
        raise OnlineFrameArchiveError(
            f"capture metadata {name} must be a finite {size}x{size} matrix, "
            f"got shape {matrix.shape}"
        )
    return np.ascontiguousarray(matrix)


def _metadata_k_color(metadata: Mapping[str, Any]) -> np.ndarray:
    """Accept k_color or the fx/fy/cx/cy intrinsics mapping, like the chunk
    stream's _intrinsics_matrix, so the archive supports every capture the
    chunk pipeline supports."""
    k_color = metadata.get("k_color")
    if k_color is not None:
        return _square_matrix(k_color, size=3, name="k_color")
    intrinsics = metadata.get("intrinsics")
    if isinstance(intrinsics, Mapping):
        return _square_matrix(
            [
                [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
                [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                [0.0, 0.0, 1.0],
            ],
            size=3,
            name="intrinsics",
        )
    raise OnlineFrameArchiveError(
        "capture metadata has neither k_color nor an fx/fy/cx/cy intrinsics "
        "mapping for the online_data frame archive"
    )


def _metadata_c2w(metadata: Mapping[str, Any]) -> np.ndarray:
    """Missing table calibration falls back to identity, matching the chunk
    stream's _camera_to_world — the committed chunks' pcd_points were lifted
    with the same identity, so the archived calibration stays consistent with
    the published world frame."""
    value = metadata.get("camera_to_world_c2w")
    if value is None:
        return np.eye(4, dtype=np.float64)
    return _square_matrix(value, size=4, name="camera_to_world_c2w")


class OnlineFrameArchive:
    """Session-lived writer for the online_data per-frame RGB-D products.

    Construction clears stale per-frame outputs (color/, depth/,
    metadata.json, calibrate.pkl, enhance_metadata.json) without touching
    chunks/ or manifest.json; ``initialize_case`` publishes the calibration
    from capture metadata before any frame streams.
    """

    def __init__(self, *, base_path: str | Path, fps: int) -> None:
        """Initialize OnlineFrameArchive and clear stale per-frame outputs."""
        self.online_dir = Path(base_path) / "online_data"
        self.fps = int(fps)
        self.color_dir = self.online_dir / "color" / str(CAMERA_INDEX)
        self.depth_dir = self.online_dir / "depth" / str(CAMERA_INDEX)
        self.metadata_path = self.online_dir / "metadata.json"
        self.calibrate_path = self.online_dir / "calibrate.pkl"
        self.enhance_metadata_path = self.online_dir / "enhance_metadata.json"
        self._clear_stale_outputs()
        self.color_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.frames_committed = 0
        # Real-time streaming state: ``frames_streamed`` counts frames whose
        # color/depth files are already on disk (written per frame as capture
        # rows are accepted), while ``frames_committed`` counts only frames
        # whose owning chunk committed — metadata.json's frame_num contract is
        # unchanged. ``_streamed_info[i]`` records the identity of streamed
        # online frame i so archive_chunk can verify instead of rewriting.
        self.frames_streamed = 0
        self._streamed_info: list[dict[str, Any]] = []
        self._frame_mapping: list[dict[str, Any]] = []
        self._calibration: dict[str, Any] | None = None

    def _clear_stale_outputs(self) -> None:
        """Remove previous-run per-frame outputs, keeping chunks/ untouched.

        metadata.json goes first: an absent metadata.json is the unambiguous
        "no data" signal for case readers, so an interruption mid-clear never
        leaves a frame_num pointing at already-deleted frame files.
        """
        import shutil  # noqa: PLC0415

        for file_path in (
            self.metadata_path,
            self.enhance_metadata_path,
            self.calibrate_path,
        ):
            if file_path.exists():
                file_path.unlink()
        for directory in (self.online_dir / "color", self.online_dir / "depth"):
            if directory.is_dir():
                shutil.rmtree(directory)

    def _initialize_calibration(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        """Extract the single-camera calibration and publish calibrate.pkl."""
        k_color = _metadata_k_color(metadata)
        c2w = _metadata_c2w(metadata)
        width = metadata.get("width")
        height = metadata.get("height")
        if width is None or height is None:
            raise OnlineFrameArchiveError(
                "capture metadata is missing width/height for the online_data "
                "frame archive"
            )
        serial_number = str(metadata.get("serial") or "").strip()
        if not serial_number:
            raise OnlineFrameArchiveError(
                "capture metadata is missing the single-camera serial"
            )
        calibration = {
            "k_color": k_color,
            "c2w": c2w,
            "width": int(width),
            "height": int(height),
            "serial_number": serial_number,
        }
        # data_process_origin/data_process_pcd.py:166 unpickles an indexable
        # sequence of per-camera 4x4 camera-to-world matrices.
        atomic_pickle_dump([c2w], self.calibrate_path)
        return calibration

    def initialize_case(self, metadata: Mapping[str, Any]) -> None:
        """Seed the case dir before any chunk exists (frame_num = 0).

        Downstream consumers launched at shape-prior-ready time
        (downstream.mode: phystwin_shen) read calibrate.pkl/metadata.json at
        startup, before the first chunk commits — so those files must exist
        as soon as capture metadata is known. Idempotent: once frames are
        archived this republishes the current committed state.
        """
        if self._calibration is None:
            self._calibration = self._initialize_calibration(metadata)
        self._write_metadata(self._calibration)
        self._write_enhance_metadata()

    def _archive_one_frame(
        self,
        *,
        online_frame_index: int,
        frame: PreparedPhysTwinFrame,
        calibration: Mapping[str, Any],
        context: str,
    ) -> None:
        """Write one color png + depth npy pair for a published frame."""
        height = int(calibration["height"])
        width = int(calibration["width"])
        rgb = np.asarray(frame.rgb_frame)
        if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
            raise OnlineFrameArchiveError(
                f"{context}: rgb_frame must be ({height}, {width}, 3) uint8, "
                f"got shape {rgb.shape} dtype {rgb.dtype}"
            )
        if frame.depth_mm_u16 is None:
            raise OnlineFrameArchiveError(
                f"{context}: prepared frame has no depth_mm_u16; the capture "
                "predates the online_data color/depth contract — re-run the "
                "capture with the current demo_v6_1"
            )
        depth = np.asarray(frame.depth_mm_u16)
        if depth.shape != (height, width) or depth.dtype != np.uint16:
            raise OnlineFrameArchiveError(
                f"{context}: depth_mm_u16 must be ({height}, {width}) uint16, "
                f"got shape {depth.shape} dtype {depth.dtype}"
            )
        # data_process_origin reads color with cv2.imread (BGR on disk).
        ok, png = cv2.imencode(".png", np.ascontiguousarray(rgb[:, :, ::-1]))
        if not ok:
            raise OnlineFrameArchiveError(f"{context}: PNG encoding failed")
        # metadata.json is fsync'd, so the frame files it points at must
        # survive a power loss too.
        with atomic_open(self.color_dir / f"{online_frame_index}.png") as handle:
            handle.write(png.tobytes())
        with atomic_open(self.depth_dir / f"{online_frame_index}.npy") as handle:
            np.save(handle, depth)

    def stream_frame(self, frame: PreparedPhysTwinFrame) -> int:
        """Write one frame's color/depth immediately, without waiting for a chunk.

        The stream bridge calls this the moment a capture row is accepted, so
        ``color/0/{k}.png`` + ``depth/0/{k}.npy`` appear at frame cadence
        instead of chunk-commit cadence. ``metadata.json``'s ``frame_num``
        still advances only in ``publish_metadata`` after the owning chunk
        commits: strict case readers keep the committed-only contract, while
        live consumers may watch the color/depth directories directly.
        ``archive_chunk`` later verifies the streamed files' identity instead
        of rewriting them, and ``discard_streamed_tail`` removes frames whose
        chunk never committed when the stream ends.
        """
        if self._calibration is None:
            raise OnlineFrameArchiveError(
                "stream_frame requires initialize_case to have published the "
                "camera calibration first"
            )
        online_frame_index = int(self.frames_streamed)
        self._archive_one_frame(
            online_frame_index=online_frame_index,
            frame=frame,
            calibration=self._calibration,
            context=f"streamed frame (online frame {online_frame_index})",
        )
        # Advance the streamed counter with the frame's identity record.
        self._streamed_info.append(
            {
                "seq": int(frame.seq),
                "source_frame_index": (
                    None
                    if frame.source_frame_index is None
                    else int(frame.source_frame_index)
                ),
            }
        )
        self.frames_streamed += 1
        return online_frame_index

    def discard_streamed_tail(self) -> int:
        """Delete streamed frames whose chunk never committed; return the count.

        Called when the stream ends. Color/depth files remain exactly for
        frames of committed chunks, and ``frame_num`` keeps pointing only at
        files that exist.
        """
        discarded = 0
        for index in range(int(self.frames_committed), int(self.frames_streamed)):
            for path in (
                self.color_dir / f"{index}.png",
                self.depth_dir / f"{index}.npy",
            ):
                if path.exists():
                    path.unlink()
            discarded += 1
        del self._streamed_info[int(self.frames_committed) :]
        self.frames_streamed = int(self.frames_committed)
        return discarded

    def archive_chunk(
        self,
        *,
        chunk_id: int,
        frames: Sequence[PreparedPhysTwinFrame],
        source_frame_indices: Sequence[int],
        online_start_frame: int,
    ) -> dict[str, Any]:
        """Archive every published frame of one chunk.

        Frame files must already have landed through ``stream_frame``. This
        method verifies their identity before the caller commits the online
        chunk; metadata.json/enhance_metadata.json only advance in
        ``publish_metadata`` after that commit succeeds. At every interruption
        point ``frame_num`` therefore counts only frames that both exist on
        disk and belong to a committed chunk.
        """
        if int(online_start_frame) != int(self.frames_committed):
            raise OnlineFrameArchiveError(
                f"chunk {chunk_id}: online frame index discontinuity — "
                f"archive has {self.frames_committed} frames but the chunk "
                f"starts at online frame {online_start_frame}"
            )
        if len(frames) != len(source_frame_indices):
            raise OnlineFrameArchiveError(
                f"chunk {chunk_id}: {len(frames)} prepared frames but "
                f"{len(source_frame_indices)} source_frame_indices"
            )
        for local_index, frame in enumerate(frames):
            online_frame_index = int(online_start_frame) + local_index
            source_frame_index = int(source_frame_indices[local_index])
            context = (
                f"chunk {chunk_id} frame {local_index} "
                f"(online frame {online_frame_index}, "
                f"source frame {source_frame_index})"
            )
            if (
                frame.source_frame_index is not None
                and int(frame.source_frame_index) != source_frame_index
            ):
                raise OnlineFrameArchiveError(
                    f"{context}: prepared frame carries source_frame_index "
                    f"{frame.source_frame_index}, chunk says {source_frame_index}"
                )
            if online_frame_index >= self.frames_streamed:
                raise OnlineFrameArchiveError(
                    f"{context}: frame was not streamed before chunk commit"
                )
            streamed = self._streamed_info[online_frame_index]
            if int(streamed["seq"]) != int(frame.seq):
                raise OnlineFrameArchiveError(
                    f"{context}: streamed file was written for seq "
                    f"{streamed['seq']} but the committed chunk carries "
                    f"seq {frame.seq}"
                )
            self._frame_mapping.append(
                {
                    "online_frame_index": online_frame_index,
                    "seq": int(frame.seq),
                    "source_frame_index": source_frame_index,
                    "depth_path": f"depth/{CAMERA_INDEX}/{online_frame_index}.npy",
                }
            )
        self.frames_committed += len(frames)
        return {
            "online_frame_archive_frames": int(self.frames_committed),
            "online_frame_archive_dir": str(self.online_dir),
        }

    def publish_metadata(self) -> None:
        """Advance metadata.json/enhance_metadata.json to the archived frames.

        Called after the corresponding chunk commit succeeds, so a failed
        commit leaves ``frame_num`` at the previous committed value — the
        extra frame files on disk are harmless, while the reverse (metadata
        claiming frames of an uncommitted chunk) would silently diverge case
        readers from the published chunk stream.
        """
        if self._calibration is None:
            raise OnlineFrameArchiveError(
                "publish_metadata called before any chunk was archived"
            )
        self._write_metadata(self._calibration)
        self._write_enhance_metadata()

    def _write_metadata(self, calibration: Mapping[str, Any]) -> None:
        """Rewrite metadata.json in the data_process_origin case schema."""
        metadata = {
            "serial_numbers": [str(calibration["serial_number"])],
            "WH": [int(calibration["width"]), int(calibration["height"])],
            "intrinsics": [np.asarray(calibration["k_color"]).tolist()],
            "frame_num": int(self.frames_committed),
            "fps": int(self.fps),
        }
        atomic_json_dump(metadata, self.metadata_path)

    def _write_enhance_metadata(self) -> None:
        """Rewrite the online-frame -> source-frame mapping table.

        The full table is rewritten per chunk (~55 bytes/frame), which is
        O(total_frames) per publish — negligible for the minutes-long demo
        sessions this targets (a 1 h run rewrites ~1 MB every 7 s chunk).
        Switch to an append-only sidecar before running for many hours.
        """
        payload = {
            "frame_mapping": list(self._frame_mapping),
        }
        atomic_json_dump(payload, self.enhance_metadata_path)


__all__ = [
    "OnlineFrameArchive",
    "OnlineFrameArchiveError",
]
