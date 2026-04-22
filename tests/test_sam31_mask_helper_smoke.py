from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from scripts.harness.sam31_mask_helper import (
    _build_sam31_builder_kwargs,
    _call_download_ckpt_from_hf,
    _resolve_sam3_video_predictor_builder,
    build_mask_output_path,
    default_output_dir,
    discover_color_sources,
    parse_text_prompts,
    resolve_sam31_bpe_path,
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

    def test_builder_resolution_supports_legacy_and_current_upstream_names(self) -> None:
        legacy_module = type("LegacyBuilderModule", (), {"build_sam3_predictor": object()})()
        current_module = type("CurrentBuilderModule", (), {"build_sam3_video_predictor": object()})()

        legacy_builder, legacy_name = _resolve_sam3_video_predictor_builder(legacy_module)
        current_builder, current_name = _resolve_sam3_video_predictor_builder(current_module)

        self.assertEqual(legacy_name, "build_sam3_predictor")
        self.assertIs(legacy_builder, legacy_module.build_sam3_predictor)
        self.assertEqual(current_name, "build_sam3_video_predictor")
        self.assertIs(current_builder, current_module.build_sam3_video_predictor)

    def test_builder_kwargs_match_selected_upstream_api(self) -> None:
        self.assertEqual(
            _build_sam31_builder_kwargs(
                "build_sam3_predictor",
                checkpoint_path="/tmp/model.pt",
                bpe_path=None,
                async_loading_frames=True,
                compile_model=True,
                max_num_objects=7,
            ),
            {
                "checkpoint_path": "/tmp/model.pt",
                "version": "sam3.1",
                "compile": True,
                "warm_up": False,
                "max_num_objects": 7,
                "use_fa3": False,
                "async_loading_frames": True,
            },
        )
        self.assertEqual(
            _build_sam31_builder_kwargs(
                "build_sam3_video_predictor",
                checkpoint_path="/tmp/model.pt",
                bpe_path="/tmp/bpe.txt.gz",
                async_loading_frames=True,
                compile_model=True,
                max_num_objects=7,
            ),
            {
                "checkpoint_path": "/tmp/model.pt",
                "bpe_path": "/tmp/bpe.txt.gz",
                "async_loading_frames": True,
            },
        )

    def test_download_ckpt_compat_supports_versioned_and_versionless_functions(self) -> None:
        calls: list[tuple[str, str | None]] = []

        def _versioned(*, version: str) -> str:
            calls.append(("versioned", version))
            return "/tmp/versioned.pt"

        def _versionless() -> str:
            calls.append(("versionless", None))
            return "/tmp/versionless.pt"

        self.assertEqual(_call_download_ckpt_from_hf(_versioned), "/tmp/versioned.pt")
        self.assertEqual(_call_download_ckpt_from_hf(_versionless), "/tmp/versionless.pt")
        self.assertEqual(calls, [("versioned", "sam3.1"), ("versionless", None)])

    def test_resolve_sam31_bpe_path_prefers_checkpoint_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "sam3.1_multiplex.pt"
            bpe_path = Path(tmp_dir) / "bpe_simple_vocab_16e6.txt.gz"
            checkpoint_path.write_bytes(b"fake")
            bpe_path.write_bytes(b"fake")

            self.assertEqual(resolve_sam31_bpe_path(checkpoint_path), str(bpe_path.resolve()))


if __name__ == "__main__":
    raise SystemExit(unittest.main())
