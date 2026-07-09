"""Tests for the canonical Demo v6.2 tracking-video loader."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from demo_v6_2.others import visualize_object_controller_tracking as visualization


def _chunk_payload(
    *,
    start_frame: int,
    source_start: int,
    status: str,
) -> dict[str, object]:
    frame_count = 2
    object_points = np.arange(
        frame_count * 2 * 3,
        dtype=np.float32,
    ).reshape(frame_count, 2, 3)
    controller_points = np.arange(
        frame_count * 3,
        dtype=np.float32,
    ).reshape(frame_count, 1, 3)
    return {
        "start_frame": start_frame,
        "end_frame": start_frame + frame_count,
        "object_points": object_points,
        "object_visibilities": np.asarray(
            [[True, True], [True, False]],
            dtype=bool,
        ),
        "object_motions_valid": np.asarray(
            [[True, False], [True, True]],
            dtype=bool,
        ),
        "controller_points": controller_points,
        "source_frame_indices": [source_start, source_start + 1],
        "query_schema_hash": "stable-schema",
        "track_process_status": status,
    }


def _write_chunk(path: Path, payload: dict[str, object]) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


class TrackingSequenceTests(unittest.TestCase):
    def test_loads_contiguous_chunks_and_applies_one_render_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chunks_dir = Path(tmp)
            _write_chunk(
                chunks_dir / "chunk_000000.pkl",
                _chunk_payload(start_frame=0, source_start=10, status="normal"),
            )
            _write_chunk(
                chunks_dir / "chunk_000001.pkl",
                _chunk_payload(start_frame=2, source_start=12, status="degraded"),
            )

            tracking = visualization.load_tracking_sequence(chunks_dir)

        self.assertEqual(tracking.frame_count, 4)
        self.assertEqual(tracking.object_points.shape, (4, 2, 3))
        self.assertEqual(tracking.controller_points.shape, (4, 1, 3))
        self.assertEqual(tracking.source_frame_indices, (10, 11, 12, 13))
        self.assertEqual(
            tracking.track_status_counts,
            {"normal": 1, "degraded": 1},
        )
        np.testing.assert_array_equal(
            tracking.rendered_object_mask,
            np.asarray(
                [
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                ],
                dtype=bool,
            ),
        )

    def test_rejects_noncontiguous_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chunks_dir = Path(tmp)
            _write_chunk(
                chunks_dir / "chunk_000000.pkl",
                _chunk_payload(start_frame=0, source_start=10, status="normal"),
            )
            _write_chunk(
                chunks_dir / "chunk_000001.pkl",
                _chunk_payload(start_frame=3, source_start=12, status="normal"),
            )

            with self.assertRaisesRegex(ValueError, "expected 2"):
                visualization.load_tracking_sequence(chunks_dir)

    def test_rejects_nonboolean_tracking_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chunks_dir = Path(tmp)
            payload = _chunk_payload(
                start_frame=0,
                source_start=10,
                status="normal",
            )
            payload["object_visibilities"] = np.ones((2, 2), dtype=np.uint8)
            _write_chunk(chunks_dir / "chunk_000000.pkl", payload)

            with self.assertRaisesRegex(TypeError, "must be boolean"):
                visualization.load_tracking_sequence(chunks_dir)

    def test_summary_reports_current_render_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_dir = root / "chunks"
            chunks_dir.mkdir()
            _write_chunk(
                chunks_dir / "chunk_000000.pkl",
                _chunk_payload(start_frame=0, source_start=10, status="normal"),
            )
            tracking = visualization.load_tracking_sequence(chunks_dir)
            output_path = root / "tracking.mp4"
            output_path.write_bytes(b"video")

            summary = visualization.build_summary(
                tracking,
                output_path=output_path,
                fps=5.0,
                width=1280,
                height=900,
                controller_radius_m=0.01,
            )

        self.assertEqual(summary["render_policy"], "visibility_and_motion_valid")
        self.assertEqual(summary["frame_count"], 2)
        self.assertEqual(summary["video_size_bytes"], 5)
        self.assertIn("demo_v6_2", str(visualization.DEFAULT_OUTPUT_PATH))


if __name__ == "__main__":
    unittest.main()
