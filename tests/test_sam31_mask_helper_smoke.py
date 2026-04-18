from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from scripts.harness.sam31_mask_helper import (
    build_mask_output_path,
    default_output_dir,
    discover_color_sources,
    parse_text_prompts,
)


ROOT = Path(__file__).resolve().parents[1]


class Sam31MaskHelperSmokeTest(unittest.TestCase):
    def test_parse_text_prompts_deduplicates_and_normalizes(self) -> None:
        self.assertEqual(
            parse_text_prompts(" sloth . hand ; hand\ncontroller "),
            ["sloth", "hand", "controller"],
        )

    def test_discover_color_sources_prefers_mp4_and_accepts_frame_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_root = Path(tmp_dir) / "case"
            (case_root / "color").mkdir(parents=True, exist_ok=True)
            (case_root / "color" / "0.mp4").write_bytes(b"fake")
            frame_dir = case_root / "color" / "1"
            frame_dir.mkdir(parents=True, exist_ok=True)
            (frame_dir / "12.png").write_bytes(b"fake")
            (frame_dir / "7.png").write_bytes(b"fake")

            sources = discover_color_sources(case_root, camera_ids=[0, 1], source_mode="auto")

            self.assertEqual(sources[0].mode, "mp4")
            self.assertEqual(sources[1].mode, "frames")
            self.assertEqual([item.name for item in sources[1].frame_paths], ["7.png", "12.png"])

    def test_output_contract_paths_are_case_local(self) -> None:
        case_root = Path("C:/tmp/example_case")
        output_dir = default_output_dir(case_root)
        self.assertEqual(output_dir, case_root.resolve() / "sam31_masks")
        mask_path = build_mask_output_path(output_dir, camera_idx=2, obj_id=5, frame_token="1012")
        self.assertEqual(mask_path, output_dir.resolve() / "mask" / "2" / "5" / "1012.png")

    def test_cli_help_does_not_require_sam3_runtime(self) -> None:
        command = [sys.executable, "scripts/harness/generate_sam31_masks.py", "--help"]
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Generate SAM 3.1 object masks", result.stdout)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
