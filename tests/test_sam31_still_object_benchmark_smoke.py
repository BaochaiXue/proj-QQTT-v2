from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.harness.benchmark_sam31_still_object_views import (
    CameraBenchmarkTiming,
    _limited_frame_source,
    build_average_summary,
)
from scripts.harness.sam31_mask_helper import ColorSource


ROOT = Path(__file__).resolve().parents[1]


class Sam31StillObjectBenchmarkSmokeTest(unittest.TestCase):
    def test_cli_help_does_not_require_sam3_runtime(self) -> None:
        command = [sys.executable, "scripts/harness/benchmark_sam31_still_object_views.py", "--help"]
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Benchmark SAM 3.1 segmentation speed", result.stdout)

    def test_limited_frame_source_selects_exact_requested_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_dir = Path(tmp_dir)
            frame_paths = [frame_dir / f"{idx}.png" for idx in range(5)]
            source = ColorSource(camera_idx=1, mode="frames", path=frame_dir, frame_paths=frame_paths)

            limited = _limited_frame_source(source, frame_count=3)

            self.assertEqual(limited.camera_idx, 1)
            self.assertEqual(limited.frame_paths, frame_paths[:3])

    def test_limited_frame_source_rejects_short_or_video_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_dir = Path(tmp_dir)
            source = ColorSource(camera_idx=0, mode="frames", path=frame_dir, frame_paths=[frame_dir / "0.png"])
            with self.assertRaisesRegex(ValueError, "expected at least 2"):
                _limited_frame_source(source, frame_count=2)

            mp4_source = ColorSource(camera_idx=0, mode="mp4", path=frame_dir / "0.mp4")
            with self.assertRaisesRegex(ValueError, "requires frame-directory"):
                _limited_frame_source(mp4_source, frame_count=1)

    def test_average_summary_reports_per_frame_means(self) -> None:
        timings = [
            CameraBenchmarkTiming(
                camera_idx=0,
                source_path="/tmp/0",
                frame_count=10,
                frame_tokens=[],
                tracked_object_count=1,
                saved_frame_count=10,
                frame_prep_seconds=1.0,
                predictor_build_seconds=2.0,
                start_session_seconds=0.5,
                prompt_seconds=1.0,
                propagate_seconds=2.0,
                mask_write_seconds=0.0,
                close_seconds=0.5,
                segment_seconds=3.0,
                total_seconds=7.0,
                segment_ms_per_frame=300.0,
                total_ms_per_frame=700.0,
            ),
            CameraBenchmarkTiming(
                camera_idx=1,
                source_path="/tmp/1",
                frame_count=10,
                frame_tokens=[],
                tracked_object_count=1,
                saved_frame_count=10,
                frame_prep_seconds=1.0,
                predictor_build_seconds=4.0,
                start_session_seconds=0.5,
                prompt_seconds=2.0,
                propagate_seconds=2.0,
                mask_write_seconds=0.0,
                close_seconds=0.5,
                segment_seconds=4.0,
                total_seconds=10.0,
                segment_ms_per_frame=400.0,
                total_ms_per_frame=1000.0,
            ),
        ]

        summary = build_average_summary(timings)

        self.assertEqual(summary["camera_count"], 2.0)
        self.assertEqual(summary["segment_seconds_per_camera_mean"], 3.5)
        self.assertEqual(summary["segment_ms_per_frame_mean"], 350.0)
        self.assertEqual(summary["total_seconds_per_camera_mean"], 8.5)
        self.assertEqual(summary["predictor_build_seconds_mean"], 3.0)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
