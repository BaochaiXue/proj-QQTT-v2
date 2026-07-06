"""Tests for the Demo v6 online_data per-frame RGB-D archive."""

from __future__ import annotations

import dataclasses
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from demo_v6.online_frame_archive import (
    OnlineFrameArchive,
    OnlineFrameArchiveError,
)
from demo_v6.phystwin_strict_product import (
    PreparedPhysTwinFrame,
    depth_m_to_mm_u16,
    load_prepared_phystwin_frame,
    prepare_phystwin_frame,
    write_prepared_phystwin_frame,
)

# The RealSense D4xx default depth scale as reported by the device (float32).
REALSENSE_DEPTH_SCALE = np.float32(0.0010000000474974513)

HEIGHT = 6
WIDTH = 8
K_COLOR = [[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]]
C2W = [
    [1.0, 0.0, 0.0, 0.1],
    [0.0, -1.0, 0.0, 0.2],
    [0.0, 0.0, -1.0, 0.7],
    [0.0, 0.0, 0.0, 1.0],
]


def _capture_metadata() -> dict:
    return {
        "k_color": K_COLOR,
        "camera_to_world_c2w": C2W,
        "width": WIDTH,
        "height": HEIGHT,
        "serial": "test-serial-001",
    }


def _make_frame(
    *,
    seq: int,
    source_frame_index: int | None,
    depth_mm: np.ndarray | None = None,
    rgb: np.ndarray | None = None,
) -> PreparedPhysTwinFrame:
    rng = np.random.default_rng(seq)
    if rgb is None:
        rgb = rng.integers(0, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    if depth_mm is None:
        depth_mm = rng.integers(0, 2000, size=(HEIGHT, WIDTH), dtype=np.uint16)
    grid = np.zeros((1, HEIGHT, WIDTH, 3), dtype=np.float32)
    return PreparedPhysTwinFrame(
        seq=seq,
        rgb_frame=np.ascontiguousarray(rgb),
        processed_mask_frame={
            "object": np.ones((HEIGHT, WIDTH), dtype=bool),
            "controller": np.zeros((HEIGHT, WIDTH), dtype=bool),
        },
        pcd_points=grid,
        pcd_colors=np.zeros((1, HEIGHT, WIDTH, 3), dtype=np.uint8),
        tracks_yx=np.zeros((4, 2), dtype=np.float32),
        visibility=np.ones((4,), dtype=bool),
        query_points_yx=np.zeros((4, 2), dtype=np.float32),
        source_timestamp_s=100.0 + seq,
        source_frame_index=source_frame_index,
        source_step=None if source_frame_index is None else source_frame_index + 117,
        depth_mm_u16=depth_mm,
    )


class DepthConversionTests(unittest.TestCase):
    def test_realsense_units_round_trip_bit_exact(self) -> None:
        rng = np.random.default_rng(0)
        units = rng.integers(0, 65536, size=(32, 48), dtype=np.uint16)
        units[0, 0] = 0
        units[0, 1] = np.uint16(65535)
        depth_m = units.astype(np.float32) * REALSENSE_DEPTH_SCALE
        np.testing.assert_array_equal(depth_m_to_mm_u16(depth_m), units)

    def test_invalid_values_map_to_zero(self) -> None:
        # 70 m overflows uint16 millimeters (FFS far-field garbage) and must
        # become the invalid-0 sentinel, not saturate to a "valid" 65535.
        depth_m = np.array(
            [[np.nan, np.inf, -np.inf], [-0.5, 0.0, 70.0]], dtype=np.float32
        )
        result = depth_m_to_mm_u16(depth_m)
        self.assertEqual(result.dtype, np.uint16)
        np.testing.assert_array_equal(result, np.zeros((2, 3), dtype=np.uint16))
        np.testing.assert_array_equal(
            depth_m_to_mm_u16(np.array([[65.535, 65.536]], dtype=np.float32)),
            np.array([[65535, 0]], dtype=np.uint16),
        )

    def test_requires_2d(self) -> None:
        with self.assertRaises(ValueError):
            depth_m_to_mm_u16(np.zeros((2, 3, 1), dtype=np.float32))


class PreparedFrameDepthTests(unittest.TestCase):
    def test_prepare_populates_depth_mm_u16(self) -> None:
        depth_m = np.full((HEIGHT, WIDTH), 0.5, dtype=np.float32)
        depth_m[0, 0] = np.nan
        frame = prepare_phystwin_frame(
            seq=3,
            rgb_frame=np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8),
            depth_m=depth_m,
            mask_frame={"object": np.ones((HEIGHT, WIDTH), dtype=bool)},
            tracks_yx=np.zeros((2, 2), dtype=np.float32),
            visibility=np.ones((2,), dtype=bool),
            query_points_yx=np.zeros((2, 2), dtype=np.float32),
            intrinsics=K_COLOR,
            c2w=np.asarray(C2W),
            mask_radius_outlier_filter=False,
        )
        self.assertIsNotNone(frame.depth_mm_u16)
        self.assertEqual(frame.depth_mm_u16.dtype, np.uint16)
        self.assertEqual(frame.depth_mm_u16.shape, (HEIGHT, WIDTH))
        self.assertEqual(int(frame.depth_mm_u16[0, 0]), 0)
        self.assertEqual(int(frame.depth_mm_u16[1, 1]), 500)

    def test_npz_round_trip_preserves_depth(self) -> None:
        frame = _make_frame(seq=5, source_frame_index=42)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.npz"
            write_prepared_phystwin_frame(path, frame)
            loaded = load_prepared_phystwin_frame(path)
        np.testing.assert_array_equal(loaded.depth_mm_u16, frame.depth_mm_u16)

    def test_legacy_npz_without_depth_loads_none(self) -> None:
        frame = _make_frame(seq=6, source_frame_index=43)
        legacy = dataclasses.replace(frame, depth_mm_u16=None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.npz"
            write_prepared_phystwin_frame(path, legacy)
            loaded = load_prepared_phystwin_frame(path)
        self.assertIsNone(loaded.depth_mm_u16)


class OnlineFrameArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base_path = Path(self._tmp.name)
        self.online_dir = self.base_path / "online_data"
        self.archive = OnlineFrameArchive(
            base_path=self.base_path, case_name="test_case", fps=5
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _archive(
        self,
        frames,
        *,
        chunk_id,
        start,
        indices=None,
        timestamps=None,
        metadata=None,
        publish=True,
    ):
        if indices is None:
            indices = [
                frame.source_frame_index for frame in frames if frame is not None
            ]
        summary = self.archive.archive_chunk(
            chunk_id=chunk_id,
            metadata=_capture_metadata() if metadata is None else metadata,
            serial_number="fallback-serial",
            frames=frames,
            source_frame_indices=indices,
            source_timestamps_s=timestamps,
            online_start_frame=start,
        )
        if publish:
            # Mirrors _write_chunk_from_rows: metadata advances only after
            # the chunk commit succeeds.
            self.archive.publish_metadata()
        return summary

    def test_archive_writes_case_layout(self) -> None:
        # Chunk 0 starts with the warmup anchor (source frame 0); chunk 1
        # continues the online index without a gap.
        chunk0 = [_make_frame(seq=0, source_frame_index=0),
                  _make_frame(seq=7, source_frame_index=2387)]
        chunk1 = [_make_frame(seq=8, source_frame_index=2393),
                  _make_frame(seq=9, source_frame_index=2399)]
        summary = self._archive(chunk0, chunk_id=0, start=0, timestamps=[1.0, 2.0])
        self.assertEqual(summary["online_frame_archive_frames"], 2)
        self._archive(chunk1, chunk_id=1, start=2)

        # data_process_origin-style reads: contiguous integer filenames,
        # cv2.imread color, np.load(depth)/1000.0, indexable c2w pickle.
        metadata = json.loads((self.online_dir / "metadata.json").read_text())
        self.assertEqual(metadata["frame_num"], 4)
        self.assertEqual(metadata["WH"], [WIDTH, HEIGHT])
        self.assertEqual(metadata["serial_numbers"], ["test-serial-001"])
        intrinsics = np.array(metadata["intrinsics"])
        self.assertEqual(intrinsics.shape, (1, 3, 3))
        np.testing.assert_allclose(intrinsics[0], np.asarray(K_COLOR))
        with (self.online_dir / "calibrate.pkl").open("rb") as handle:
            c2ws = pickle.load(handle)
        np.testing.assert_allclose(c2ws[0], np.asarray(C2W))
        all_frames = chunk0 + chunk1
        for online_idx in range(metadata["frame_num"]):
            color = cv2.imread(str(self.online_dir / "color" / "0" / f"{online_idx}.png"))
            self.assertIsNotNone(color)
            np.testing.assert_array_equal(
                cv2.cvtColor(color, cv2.COLOR_BGR2RGB),
                all_frames[online_idx].rgb_frame,
            )
            depth_mm = np.load(self.online_dir / "depth" / "0" / f"{online_idx}.npy")
            self.assertEqual(depth_mm.dtype, np.uint16)
            np.testing.assert_array_equal(depth_mm, all_frames[online_idx].depth_mm_u16)
            depth_m = depth_mm / 1000.0
            self.assertEqual(depth_m.shape, (HEIGHT, WIDTH))

        enhance = json.loads((self.online_dir / "enhance_metadata.json").read_text())
        mapping = enhance["frame_mapping"]
        self.assertEqual(len(mapping), 4)
        # The warmup anchor is online frame 0 and maps back to source frame 0.
        self.assertEqual(mapping[0]["online_frame_index"], 0)
        self.assertEqual(mapping[0]["source_frame_index"], 0)
        self.assertEqual(mapping[0]["chunk_id"], 0)
        self.assertEqual(mapping[0]["color_path"], "color/0/0.png")
        self.assertEqual(mapping[0]["source_timestamp_s"], 1.0)
        self.assertEqual(mapping[2]["online_frame_index"], 2)
        self.assertEqual(mapping[2]["chunk_id"], 1)
        self.assertEqual(mapping[2]["chunk_frame_index"], 0)
        self.assertEqual(mapping[2]["source_frame_index"], 2393)
        self.assertIsNone(mapping[2]["source_timestamp_s"])
        self.assertEqual(mapping[3]["depth_path"], "depth/0/3.npy")

    def test_fail_fast_on_missing_depth(self) -> None:
        frame = _make_frame(seq=0, source_frame_index=0)
        no_depth = dataclasses.replace(frame, depth_mm_u16=None)
        with self.assertRaisesRegex(OnlineFrameArchiveError, "depth_mm_u16"):
            self._archive([no_depth], chunk_id=0, start=0)

    def test_fail_fast_on_legacy_reprocess_path(self) -> None:
        with self.assertRaisesRegex(OnlineFrameArchiveError, "prepared"):
            self._archive(None, chunk_id=0, start=0, indices=[0])

    def test_fail_fast_on_online_index_discontinuity(self) -> None:
        self._archive([_make_frame(seq=0, source_frame_index=0)], chunk_id=0, start=0)
        with self.assertRaisesRegex(OnlineFrameArchiveError, "discontinuity"):
            self._archive(
                [_make_frame(seq=1, source_frame_index=5)], chunk_id=1, start=2
            )

    def test_fail_fast_on_source_index_mismatch(self) -> None:
        frame = _make_frame(seq=0, source_frame_index=10)
        with self.assertRaisesRegex(OnlineFrameArchiveError, "source_frame_index"):
            self._archive([frame], chunk_id=0, start=0, indices=[11])

    def test_metadata_only_advances_on_publish(self) -> None:
        # A failed chunk commit must leave frame_num at the previous value:
        # extra frame files are harmless, metadata pointing at frames of an
        # uncommitted chunk is not.
        self._archive(
            [_make_frame(seq=0, source_frame_index=0)],
            chunk_id=0,
            start=0,
            publish=False,
        )
        self.assertTrue((self.online_dir / "color" / "0" / "0.png").exists())
        self.assertFalse((self.online_dir / "metadata.json").exists())
        self.archive.publish_metadata()
        metadata = json.loads((self.online_dir / "metadata.json").read_text())
        self.assertEqual(metadata["frame_num"], 1)

    def test_publish_before_archive_fails(self) -> None:
        with self.assertRaisesRegex(OnlineFrameArchiveError, "publish_metadata"):
            self.archive.publish_metadata()

    def test_calibration_fallbacks_match_chunk_stream(self) -> None:
        # No table calibration -> identity c2w; fx/fy/cx/cy intrinsics form
        # accepted — the archive supports every capture the chunk stream does.
        metadata = {
            "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 4.0, "cy": 3.0},
            "camera_to_world_c2w": None,
            "width": WIDTH,
            "height": HEIGHT,
        }
        self._archive(
            [_make_frame(seq=0, source_frame_index=0)],
            chunk_id=0,
            start=0,
            metadata=metadata,
        )
        with (self.online_dir / "calibrate.pkl").open("rb") as handle:
            c2ws = pickle.load(handle)
        np.testing.assert_array_equal(c2ws[0], np.eye(4))
        online_meta = json.loads((self.online_dir / "metadata.json").read_text())
        np.testing.assert_allclose(
            np.array(online_meta["intrinsics"])[0], np.asarray(K_COLOR)
        )
        self.assertEqual(online_meta["serial_numbers"], ["fallback-serial"])

    def test_clears_stale_outputs_and_keeps_chunks(self) -> None:
        chunks_dir = self.online_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        (chunks_dir / "chunk_000000.pkl").write_bytes(b"chunk-bytes")
        stale_color = self.online_dir / "color" / "0" / "99.png"
        stale_color.parent.mkdir(parents=True, exist_ok=True)
        stale_color.write_bytes(b"stale")
        (self.online_dir / "metadata.json").write_text("{}")
        (self.online_dir / "calibrate.pkl").write_bytes(b"stale")
        (self.online_dir / "enhance_metadata.json").write_text("{}")
        OnlineFrameArchive(base_path=self.base_path, case_name="test_case", fps=5)
        self.assertFalse(stale_color.exists())
        self.assertFalse((self.online_dir / "metadata.json").exists())
        self.assertFalse((self.online_dir / "calibrate.pkl").exists())
        self.assertFalse((self.online_dir / "enhance_metadata.json").exists())
        self.assertEqual(
            (chunks_dir / "chunk_000000.pkl").read_bytes(), b"chunk-bytes"
        )


if __name__ == "__main__":
    unittest.main()
