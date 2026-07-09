"""Tests for the approved Demo v6.2 legacy/fallback cleanup."""

from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from demo_v6_2 import chunk_materialize, chunk_window_builder, main_warmup
from demo_v6_2 import phystwin_strict_product as strict
from demo_v6_2.chunk_data_stream import (
    stream_chunk_data_from_headless_capture,
    write_chunk_data_from_headless_capture,
)
from demo_v6_2.main_cli import build_parser
from demo_v6_2.main_config import load_default_config
from demo_v6_2.online_frame_archive import OnlineFrameArchiveError


def _prepared_frame(seq: int = 7) -> strict.PreparedPhysTwinFrame:
    return strict.PreparedPhysTwinFrame(
        seq=seq,
        rgb_frame=np.zeros((1, 1, 3), dtype=np.uint8),
        processed_mask_frame={
            "object": np.zeros((1, 1), dtype=bool),
            "controller": np.zeros((1, 1), dtype=bool),
        },
        pcd_points=np.zeros((1, 1, 1, 3), dtype=np.float32),
        pcd_colors=np.zeros((1, 1, 1, 3), dtype=np.uint8),
        tracks_yx=np.empty((0, 2), dtype=np.float32),
        visibility=np.empty((0,), dtype=bool),
        query_points_yx=np.empty((0, 2), dtype=np.float32),
        source_frame_index=42,
        depth_mm_u16=np.zeros((1, 1), dtype=np.uint16),
    )


class PreparedFrameRequirementTests(unittest.TestCase):
    def test_capture_row_requires_prepared_frame_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                OnlineFrameArchiveError,
                "missing prepared_phystwin_frame_path",
            ) as raised:
                chunk_window_builder._prepared_frame_from_row(
                    Path(tmp),
                    {"seq": 7, "source_frame_index": 42},
                )

        self.assertIn("seq=7", str(raised.exception))
        self.assertIn("source_frame_index=42", str(raised.exception))

    def test_capture_row_loads_canonical_prepared_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp)
            relative_path = Path("prepared_phystwin") / "000007.npz"
            strict.write_prepared_phystwin_frame(
                capture / relative_path,
                _prepared_frame(),
            )

            loaded = chunk_window_builder._prepared_frame_from_row(
                capture,
                {
                    "seq": 7,
                    "source_frame_index": 42,
                    "prepared_phystwin_frame_path": str(relative_path),
                },
            )

        self.assertEqual(loaded.seq, 7)
        self.assertEqual(loaded.source_frame_index, 42)

    def test_chunk_rejects_row_and_prepared_frame_count_mismatch(self) -> None:
        with mock.patch.object(
            chunk_materialize,
            "_chunk_data_window_from_prepared_frames",
        ) as prepared_builder:
            with self.assertRaisesRegex(
                OnlineFrameArchiveError,
                "prepared frames",
            ):
                chunk_materialize._write_chunk_from_rows(
                    capture=Path("unused"),
                    metadata={},
                    rows=[{"seq": 7, "source_frame_index": 42}],
                    case_prefix="cleanup_test",
                    chunk_index=3,
                    row_start=0,
                    row_end=1,
                    fps=5,
                    serial_number="test-camera",
                    surface_points=np.empty((0, 3), dtype=np.float32),
                    interior_points=np.empty((0, 3), dtype=np.float32),
                    wall_time_origin_s=0.0,
                    window_closed_wall_s=0.0,
                    prepared_frames=[],
                )

        prepared_builder.assert_not_called()


class OpenCvComponentTests(unittest.TestCase):
    def test_connected_components_are_sorted_by_area(self) -> None:
        mask = np.zeros((6, 6), dtype=bool)
        mask[0, :3] = True
        mask[4, 4] = True

        components = main_warmup._connected_components_by_area(mask)

        self.assertEqual([int(component.sum()) for component in components], [3, 1])

    def test_connected_components_propagates_cv2_failure(self) -> None:
        fake_cv2 = mock.Mock()
        fake_cv2.connectedComponentsWithStats.side_effect = RuntimeError("cv2 sentinel")

        with mock.patch.dict(sys.modules, {"cv2": fake_cv2}):
            with self.assertRaisesRegex(RuntimeError, "cv2 sentinel"):
                main_warmup._connected_components_by_area(np.ones((2, 2), dtype=bool))

        fake_cv2.connectedComponentsWithStats.assert_called_once()


class RemovedCompatibilityTests(unittest.TestCase):
    def test_legacy_helpers_are_absent(self) -> None:
        removed = (
            (main_warmup, "run_sam31_first_frame_masks"),
            (main_warmup, "resolve_initial_masks"),
            (chunk_window_builder, "_chunk_data_window_from_rows"),
            (chunk_materialize, "_chunk_data_window_from_rows"),
        )
        for module, name in removed:
            with self.subTest(module=module.__name__, name=name):
                self.assertFalse(hasattr(module, name))

    def test_chunk_entry_points_have_no_legacy_mask_parameters(self) -> None:
        for entry_point in (
            write_chunk_data_from_headless_capture,
            stream_chunk_data_from_headless_capture,
        ):
            with self.subTest(entry_point=entry_point.__name__):
                parameters = inspect.signature(entry_point).parameters
                self.assertFalse(
                    any(name.startswith("mask_radius_outlier") for name in parameters)
                )

    def test_orchestrator_has_no_legacy_mask_options(self) -> None:
        option_strings = build_parser()._option_string_actions
        self.assertNotIn("--mask-radius-outlier-filter", option_strings)
        self.assertNotIn("--no-mask-radius-outlier-filter", option_strings)
        self.assertNotIn("--mask-radius-outlier-radius-m", option_strings)
        self.assertNotIn("--mask-radius-outlier-nb-points", option_strings)

        camera_defaults = load_default_config()["camera"]
        self.assertNotIn("mask_radius_outlier_radius_m", camera_defaults)
        self.assertNotIn("mask_radius_outlier_nb_points", camera_defaults)


if __name__ == "__main__":
    unittest.main()
