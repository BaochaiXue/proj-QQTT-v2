from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from unittest import mock

from scripts.harness.cleanup_different_types_cases import cleanup_cases


def _write_case(root: Path, case_name: str, *, include_mp4: bool = True) -> Path:
    case_dir = root / case_name
    for stream in ("color", "depth", "ir_left", "ir_right", "depth_ffs_float_m", "depth_ffs_native_like_postprocess"):
        for camera_idx in range(4):
            stream_dir = case_dir / stream / str(camera_idx)
            stream_dir.mkdir(parents=True, exist_ok=True)
            suffix = ".png" if stream in {"color", "ir_left", "ir_right"} else ".npy"
            (stream_dir / f"0{suffix}").write_bytes(b"fixture")
    if include_mp4:
        for camera_idx in range(3):
            (case_dir / "color" / f"{camera_idx}.mp4").write_bytes(b"mp4fixture")
    (case_dir / "metadata.json").write_text(
        json.dumps(
            {
                "intrinsics": [],
                "serial_numbers": ["a", "b", "c"],
                "fps": 30,
                "WH": [848, 480],
                "frame_num": 1,
                "start_step": 0,
                "end_step": 0,
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "metadata_ext.json").write_text(json.dumps({"streams_present": ["color", "depth"]}), encoding="utf-8")
    (case_dir / "calibrate.pkl").write_bytes(b"calib")
    (case_dir / "notes.txt").write_text("remove me", encoding="utf-8")
    return case_dir


class CleanupDifferentTypesCasesSmokeTest(unittest.TestCase):
    def test_dry_run_reports_deletions_without_mutating_filesystem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "different_types"
            case_dir = _write_case(root, "case_a")
            metadata_before = (case_dir / "metadata.json").read_text(encoding="utf-8")

            result = cleanup_cases(root=root, case_names=None, execute=False)

            self.assertEqual(result["status"], "ready")
            self.assertTrue((case_dir / "metadata_ext.json").exists())
            self.assertTrue((case_dir / "ir_left").exists())
            self.assertEqual((case_dir / "metadata.json").read_text(encoding="utf-8"), metadata_before)
            case_summary = result["cases"][0]
            self.assertIn(str((case_dir / "metadata_ext.json").resolve()), case_summary["delete_paths"])
            self.assertEqual(case_summary["deleted_paths"], [])

    def test_dry_run_reports_missing_color_mp4_generation_without_mutating_filesystem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "different_types"
            case_dir = _write_case(root, "case_a", include_mp4=False)

            result = cleanup_cases(root=root, case_names=None, execute=False)

            case_summary = result["cases"][0]
            expected = [
                str((case_dir / "color" / "0.mp4").resolve()),
                str((case_dir / "color" / "1.mp4").resolve()),
                str((case_dir / "color" / "2.mp4").resolve()),
            ]
            self.assertEqual(case_summary["generate_paths"], expected)
            self.assertFalse((case_dir / "color" / "0.mp4").exists())

    def test_execute_keeps_only_formal_contract_and_preserves_metadata_json_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "different_types"
            case_dir = _write_case(root, "case_a")
            metadata_before = (case_dir / "metadata.json").read_text(encoding="utf-8")

            result = cleanup_cases(root=root, case_names=None, execute=True)

            self.assertEqual(result["status"], "executed")
            remaining = sorted(item.name for item in case_dir.iterdir())
            self.assertEqual(remaining, ["calibrate.pkl", "color", "depth", "metadata.json"])
            self.assertFalse((case_dir / "metadata_ext.json").exists())
            self.assertFalse((case_dir / "ir_left").exists())
            self.assertFalse((case_dir / "depth_ffs_float_m").exists())
            self.assertEqual((case_dir / "metadata.json").read_text(encoding="utf-8"), metadata_before)
            self.assertEqual(sorted(item.name for item in (case_dir / "color").iterdir()), ["0", "0.mp4", "1", "1.mp4", "2", "2.mp4"])
            self.assertEqual(sorted(item.name for item in (case_dir / "depth").iterdir()), ["0", "1", "2"])

    def test_execute_backfills_missing_color_mp4_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "different_types"
            case_dir = _write_case(root, "case_a", include_mp4=False)

            def fake_generate_missing_color_mp4s(*, case_dir: Path, camera_ids: list[str], fps: int, ffmpeg_bin: str) -> list[str]:
                generated = []
                for camera_id in camera_ids:
                    path = case_dir / "color" / f"{camera_id}.mp4"
                    path.write_bytes(b"generated-mp4")
                    generated.append(str(path.resolve()))
                return generated

            with (
                mock.patch("scripts.harness.cleanup_different_types_cases._find_ffmpeg", return_value="fake_ffmpeg"),
                mock.patch(
                    "scripts.harness.cleanup_different_types_cases._generate_missing_color_mp4s",
                    side_effect=fake_generate_missing_color_mp4s,
                ),
            ):
                result = cleanup_cases(root=root, case_names=None, execute=True)

            self.assertEqual(result["status"], "executed")
            case_summary = result["cases"][0]
            self.assertEqual(
                sorted(Path(path).name for path in case_summary["generated_paths"]),
                ["0.mp4", "1.mp4", "2.mp4"],
            )
            self.assertEqual(sorted(item.name for item in (case_dir / "color").iterdir()), ["0", "0.mp4", "1", "1.mp4", "2", "2.mp4"])

    def test_case_filter_only_cleans_selected_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "different_types"
            case_a = _write_case(root, "case_a")
            case_b = _write_case(root, "case_b")

            result = cleanup_cases(root=root, case_names=["case_a"], execute=True)

            self.assertEqual(result["status"], "executed")
            self.assertFalse((case_a / "metadata_ext.json").exists())
            self.assertTrue((case_b / "metadata_ext.json").exists())
            self.assertTrue((case_b / "ir_left").exists())


if __name__ == "__main__":
    unittest.main()
