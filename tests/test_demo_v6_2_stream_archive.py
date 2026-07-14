from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from demo_v6_2.streaming.online_frame_archive import (
    OnlineFrameArchive,
    OnlineFrameArchiveError,
)
from demo_v6_2.phystwin_strict_product import PreparedPhysTwinFrame


class RealtimeArchiveContractTests(unittest.TestCase):
    def test_chunk_commit_requires_frame_to_be_streamed_first(self) -> None:
        height = 2
        width = 3
        metadata = {
            "k_color": [[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]],
            "camera_to_world_c2w": np.eye(4, dtype=np.float32).tolist(),
            "width": width,
            "height": height,
            "serial": "test-camera",
        }
        frame = PreparedPhysTwinFrame(
            seq=7,
            rgb_frame=np.zeros((height, width, 3), dtype=np.uint8),
            processed_mask_frame={
                "object": np.ones((height, width), dtype=bool),
            },
            pcd_points=np.zeros((1, height, width, 3), dtype=np.float32),
            pcd_colors=np.zeros((1, height, width, 3), dtype=np.uint8),
            tracks_yx=np.zeros((1, 2), dtype=np.float32),
            visibility=np.ones((1,), dtype=bool),
            query_points_yx=np.zeros((1, 2), dtype=np.float32),
            source_frame_index=42,
            depth_mm_u16=np.ones((height, width), dtype=np.uint16),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = OnlineFrameArchive(
                base_path=Path(temporary_directory),
                fps=5,
            )
            archive.initialize_case(metadata)
            with self.assertRaisesRegex(
                OnlineFrameArchiveError,
                "not streamed before chunk commit",
            ):
                archive.archive_chunk(
                    chunk_id=0,
                    frames=[frame],
                    source_frame_indices=[42],
                    online_start_frame=0,
                )

            archive.stream_frame(frame)
            summary = archive.archive_chunk(
                chunk_id=0,
                frames=[frame],
                source_frame_indices=[42],
                online_start_frame=0,
            )
            archive.publish_metadata()
            online_metadata = json.loads(
                archive.metadata_path.read_text(encoding="utf-8")
            )
            enhance_metadata = json.loads(
                archive.enhance_metadata_path.read_text(encoding="utf-8")
            )

        self.assertEqual(summary["online_frame_archive_frames"], 1)
        self.assertEqual(
            set(online_metadata),
            {"serial_numbers", "WH", "intrinsics", "frame_num", "fps"},
        )
        self.assertEqual(online_metadata["serial_numbers"], ["test-camera"])
        self.assertEqual(online_metadata["frame_num"], 1)
        self.assertEqual(set(enhance_metadata), {"frame_mapping"})
        self.assertEqual(len(enhance_metadata["frame_mapping"]), 1)
        self.assertEqual(
            set(enhance_metadata["frame_mapping"][0]),
            {"online_frame_index", "seq", "source_frame_index", "depth_path"},
        )


if __name__ == "__main__":
    unittest.main()
