from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from demo_v6_2.online_frame_archive import (
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
                case_name="test-case",
                fps=5,
            )
            archive.initialize_case(metadata, serial_number="test-camera")
            with self.assertRaisesRegex(
                OnlineFrameArchiveError,
                "not streamed before chunk commit",
            ):
                archive.archive_chunk(
                    chunk_id=0,
                    metadata=metadata,
                    serial_number="test-camera",
                    frames=[frame],
                    source_frame_indices=[42],
                    source_timestamps_s=None,
                    online_start_frame=0,
                )

            archive.stream_frame(frame)
            summary = archive.archive_chunk(
                chunk_id=0,
                metadata=metadata,
                serial_number="test-camera",
                frames=[frame],
                source_frame_indices=[42],
                source_timestamps_s=None,
                online_start_frame=0,
            )

        self.assertEqual(summary["online_frame_archive_frames"], 1)


if __name__ == "__main__":
    unittest.main()
