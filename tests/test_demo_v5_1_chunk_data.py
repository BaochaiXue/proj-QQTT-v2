from __future__ import annotations

import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np


def _minimal_chunk_data_window(
    chunk_data_payload,
    *,
    frame_count: int = 1,
    chunk_index: int = 0,
    source_frame_indices: list[int] | None = None,
):
    object_points = np.asarray(
        [
            [[0.05 + 0.01 * float(frame_idx), 0.0, 1.0]]
            for frame_idx in range(int(frame_count))
        ],
        dtype=np.float32,
    )
    controller_points = np.asarray(
        [
            [[0.20 + 0.01 * float(frame_idx), 0.0, 1.0]]
            for frame_idx in range(int(frame_count))
        ],
        dtype=np.float32,
    )
    track_process_data = {
        "controller_mask": np.asarray([True], dtype=bool),
        "controller_points": controller_points,
        "object_colors": np.tile(
            np.asarray([[[0.7, 0.2, 0.1]]], dtype=np.float32),
            (int(frame_count), 1, 1),
        ),
        "object_motions_valid": np.ones((int(frame_count), 1), dtype=bool),
        "object_points": object_points,
        "object_visibilities": np.ones((int(frame_count), 1), dtype=bool),
        "query_ids": np.asarray([10, 20], dtype=np.int64),
        "query_semantic_labels": np.asarray([1, 2], dtype=np.int8),
        "object_query_indices": np.asarray([10], dtype=np.int64),
        "controller_query_indices": np.asarray([20], dtype=np.int64),
        "object_track_query_indices": np.asarray([10], dtype=np.int64),
        "controller_track_query_indices": np.asarray([20], dtype=np.int64),
    }
    return chunk_data_payload.ChunkDataWindow(
        track_process_data=track_process_data,
        fps=5,
        serial_number="test-camera",
        depth_backend="test-depth",
        depth_source_internal="test-depth",
        chunk_index=int(chunk_index),
        source_frame_indices=(
            list(range(int(frame_count)))
            if source_frame_indices is None
            else source_frame_indices
        ),
    )


class DemoV51ChunkDataTest(unittest.TestCase):
    def test_chunk_data_payload_builder_is_memory_only(self) -> None:
        from demo_v5_1 import chunk_data_payload

        self.assertFalse(hasattr(chunk_data_payload, "write_data_process_chunk_case"))
        self.assertFalse(hasattr(chunk_data_payload, "validate_data_process_case"))
        final_data, track_process, manifest = (
            chunk_data_payload.build_chunk_data_payload(
                _minimal_chunk_data_window(chunk_data_payload)
            )
        )

        self.assertIn("object_points", final_data)
        self.assertIn("controller_points", final_data)
        self.assertEqual(
            final_data["query_schema_hash"], track_process["query_schema_hash"]
        )
        self.assertNotIn("data_process_case_root", manifest)
        self.assertEqual("online_final_data_chunk", manifest["publish_contract"])

    def test_chunk_data_writer_does_not_emit_data_process_chunk_case(self) -> None:
        from demo_v5_1 import chunk_data_output
        from demo_v5_1 import chunk_data_payload

        final_data, track_process, _ = chunk_data_payload.build_chunk_data_payload(
            _minimal_chunk_data_window(chunk_data_payload)
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = chunk_data_output.ChunkDataWriter(
                base_path=tmpdir,
                case_name="demo_v5_1",
                chunk_size=1,
                num_frames_total=1,
            )
            result = output.commit_chunk_data(
                final_data,
                track_process,
                source_frame_indices=[0],
                status="recording",
            )
            output.finish()

            root = Path(tmpdir)
            self.assertEqual(
                root / "online_data" / "chunks" / "chunk_000000.pkl",
                Path(result["online_chunk_path"]),
            )
            self.assertEqual(
                root / "online_data" / "manifest.json",
                Path(result["online_manifest_path"]),
            )
            self.assertEqual(
                root / "data" / "final_data.pkl",
                Path(result["static_data_path"]),
            )
            self.assertTrue(Path(result["online_chunk_path"]).is_file())
            self.assertTrue(Path(result["online_manifest_path"]).is_file())
            self.assertTrue(Path(result["static_data_path"]).is_file())
            with Path(result["online_chunk_path"]).open("rb") as handle:
                chunk = pickle.load(handle)
            self.assertEqual(0, chunk["chunk_id"])
            self.assertEqual(0, chunk["start_frame"])
            self.assertEqual(1, chunk["end_frame"])
            self.assertEqual([], sorted(root.glob("demo_v5_1_chunk_*")))
            self.assertFalse((root / "online_data" / "demo_v5_1").exists())
            self.assertFalse((root / "data" / "demo_v5_1").exists())

    def test_warmup_trim_returns_delayed_source_frame_zero(self) -> None:
        from demo_v5_1 import chunk_data_stream

        result = chunk_data_stream._trim_warmup_delayed_rows(
            [
                {
                    "seq": 0,
                    "source_frame_index": 0,
                    "source_timestamp_s": 0.0,
                    "startup_hold_s": 4.0,
                    "pipeline_latency_ms": 2500.0,
                    "controller_point_count": 30,
                    "object_point_count": 1,
                },
                {
                    "seq": 42,
                    "source_frame_index": 42,
                    "source_timestamp_s": 4.5,
                    "startup_hold_s": 4.0,
                    "pipeline_latency_ms": 30.0,
                    "controller_point_count": 30,
                    "object_point_count": 1,
                },
                {
                    "seq": 43,
                    "source_frame_index": 43,
                    "source_timestamp_s": 4.7,
                    "startup_hold_s": 4.0,
                    "pipeline_latency_ms": 30.0,
                    "controller_point_count": 30,
                    "object_point_count": 1,
                },
            ]
        )

        self.assertEqual(0, result.skipped_count)
        self.assertIsNotNone(result.warmup_row)
        self.assertEqual(0, result.warmup_row["source_frame_index"])
        self.assertEqual(
            [0, 42, 43],
            [row["source_frame_index"] for row in result.rows],
        )

    def test_warmup_frame_is_first_frame_of_chunk_zero(self) -> None:
        from demo_v5_1 import chunk_data_output
        from demo_v5_1 import chunk_data_payload

        final_data, track_process, _ = chunk_data_payload.build_chunk_data_payload(
            _minimal_chunk_data_window(
                chunk_data_payload,
                frame_count=2,
                chunk_index=0,
                source_frame_indices=[0, 42],
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = chunk_data_output.ChunkDataWriter(
                base_path=tmpdir,
                case_name="demo_v5_1",
                chunk_size=2,
                num_frames_total=2,
            )
            result = output.commit_chunk_data(
                final_data,
                track_process,
                source_frame_indices=[0, 42],
                source_timestamps_s=[1.25, 4.5],
                status="recording",
            )
            output.finish()

            root = Path(tmpdir)
            chunk_path = root / "online_data" / "chunks" / "chunk_000000.pkl"
            static_data_path = root / "data" / "final_data.pkl"
            self.assertEqual(chunk_path, Path(result["online_chunk_path"]))
            self.assertTrue(chunk_path.is_file())
            self.assertTrue(static_data_path.is_file())
            self.assertFalse(
                (root / "online_data" / "chunks" / "chunk_warmup.pkl").exists()
            )
            with chunk_path.open("rb") as handle:
                chunk = pickle.load(handle)
            with static_data_path.open("rb") as handle:
                static_data = pickle.load(handle)
            self.assertNotIn("chunk_role", chunk)
            self.assertNotIn("is_warmup_chunk", chunk)
            self.assertEqual(0, chunk["chunk_id"])
            self.assertEqual(0, chunk["start_frame"])
            self.assertEqual(2, chunk["end_frame"])
            self.assertEqual([0, 42], chunk["source_frame_indices"])
            self.assertEqual([1.25, 4.5], chunk["source_timestamps_s"])
            self.assertEqual(2, int(np.asarray(chunk["object_points"]).shape[0]))
            np.testing.assert_allclose(
                np.asarray(chunk["object_points"])[0],
                np.asarray(final_data["object_points"])[0],
            )
            np.testing.assert_allclose(
                np.asarray(static_data["object_points"])[0],
                np.asarray(final_data["object_points"])[0],
            )


if __name__ == "__main__":
    unittest.main()
