from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


def _minimal_chunk(writer):
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
    return writer.DataProcessChunk(
        rgb_frames=[np.zeros((2, 2, 3), dtype=np.uint8)],
        processed_masks=[
            [
                {
                    "object": np.asarray([[True, False], [False, False]]),
                    "controller": np.asarray([[False, True], [False, False]]),
                }
            ]
        ],
        track_process_data=track_process_data,
        intrinsics=np.eye(3, dtype=np.float32),
        camera_to_world_c2w=np.eye(4, dtype=np.float32),
        fps=5,
    )


class DemoV51SplitPayloadTest(unittest.TestCase):
    def test_chunk_writer_does_not_emit_split_json(self) -> None:
        from demo_v5_1 import data_process_chunk_writer as writer

        self.assertFalse(hasattr(writer, "_split_payload"))
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = writer.write_data_process_chunk_case(
                tmpdir,
                "chunk_000000",
                _minimal_chunk(writer),
            )
            case_dir = Path(manifest["data_process_case_root"])

            self.assertFalse((case_dir / "split.json").exists())
            validation = writer.validate_data_process_case(case_dir)

        self.assertEqual(1, validation["frame_count"])

    def test_aggregate_writes_train_only_split_without_generic_helper(self) -> None:
        from demo_v5_1 import chunked_final_data_aggregate as aggregate
        from demo_v5_1 import data_process_chunk_writer as writer

        self.assertFalse(hasattr(aggregate, "_split_payload"))
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_case_dirs = []
            for chunk_idx in range(2):
                manifest = writer.write_data_process_chunk_case(
                    tmpdir,
                    f"chunk_{chunk_idx:06d}",
                    _minimal_chunk(writer),
                )
                chunk_case_dirs.append(Path(manifest["data_process_case_root"]))

            aggregate_dir = Path(tmpdir) / "aggregate"
            aggregate.build_aggregate_case_from_chunk_cases(
                chunk_case_dirs,
                aggregate_dir,
                ready=True,
            )
            split = json.loads(
                (aggregate_dir / "split.json").read_text(encoding="utf-8")
            )

        self.assertEqual({"frame_len": 2, "train": [0, 2], "test": [2, 2]}, split)


if __name__ == "__main__":
    unittest.main()
