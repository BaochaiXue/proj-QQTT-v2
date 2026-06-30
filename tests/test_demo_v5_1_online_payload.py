from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np


def _minimal_chunk(chunk_payload):
    object_points = np.asarray([[[0.05, 0.0, 1.0]]], dtype=np.float32)
    controller_points = np.asarray([[[0.20, 0.0, 1.0]]], dtype=np.float32)
    track_process_data = {
        "controller_mask": np.asarray([True], dtype=bool),
        "controller_points": controller_points,
        "object_colors": np.asarray([[[0.7, 0.2, 0.1]]], dtype=np.float32),
        "object_motions_valid": np.asarray([[True]], dtype=bool),
        "object_points": object_points,
        "object_visibilities": np.asarray([[True]], dtype=bool),
        "query_ids": np.asarray([10, 20], dtype=np.int64),
        "query_semantic_labels": np.asarray([1, 2], dtype=np.int8),
        "object_query_indices": np.asarray([10], dtype=np.int64),
        "controller_query_indices": np.asarray([20], dtype=np.int64),
        "object_track_query_indices": np.asarray([10], dtype=np.int64),
        "controller_track_query_indices": np.asarray([20], dtype=np.int64),
    }
    return chunk_payload.DataProcessChunk(
        track_process_data=track_process_data,
        fps=5,
        serial_number="test-camera",
        depth_backend="test-depth",
        depth_source_internal="test-depth",
        chunk_index=1,
        source_frame_indices=[0],
    )


class DemoV51OnlinePayloadTest(unittest.TestCase):
    def test_chunk_payload_builder_is_memory_only(self) -> None:
        from demo_v5_1 import data_process_chunk_payload

        self.assertFalse(
            hasattr(data_process_chunk_payload, "write_data_process_chunk_case")
        )
        self.assertFalse(
            hasattr(data_process_chunk_payload, "validate_data_process_case")
        )
        final_data, track_process, manifest = (
            data_process_chunk_payload.build_data_process_chunk_payload(
                _minimal_chunk(data_process_chunk_payload)
            )
        )

        self.assertIn("object_points", final_data)
        self.assertIn("controller_points", final_data)
        self.assertEqual(
            final_data["query_schema_hash"], track_process["query_schema_hash"]
        )
        self.assertNotIn("data_process_case_root", manifest)
        self.assertEqual("online_final_data_chunk", manifest["publish_contract"])

    def test_online_writer_does_not_emit_data_process_chunk_case(self) -> None:
        from demo_v5_1 import chunked_final_data_output as online
        from demo_v5_1 import data_process_chunk_payload

        final_data, track_process, _ = (
            data_process_chunk_payload.build_data_process_chunk_payload(
                _minimal_chunk(data_process_chunk_payload)
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = online.ChunkedFinalDataWriter(
                base_path=tmpdir,
                case_name="demo_v5_1",
                chunk_size=1,
                num_frames_total=1,
            )
            result = output.commit_final_data_with_track(
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
            self.assertEqual([], sorted(root.glob("demo_v5_1_chunk_*")))
            self.assertFalse((root / "online_data" / "demo_v5_1").exists())
            self.assertFalse((root / "data" / "demo_v5_1").exists())


if __name__ == "__main__":
    unittest.main()
