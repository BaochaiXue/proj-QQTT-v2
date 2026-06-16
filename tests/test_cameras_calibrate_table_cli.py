from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from cameras_calibrate_table import build_parser, resolve_output_paths, validate_cli_args


class CamerasCalibrateTableCliTest(unittest.TestCase):
    def test_parser_defaults_to_table_calibration_outputs_and_acceptance_thresholds(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.output, Path("table_calibrate.pkl"))
        self.assertEqual(args.diagnostic_image, Path("table_calibrate_diagnostic.png"))
        self.assertEqual(args.max_reprojection_error_px, 0.20)
        self.assertEqual(args.min_corner_fraction, 0.60)
        self.assertEqual(args.fps, 5)

    def test_resolve_output_paths_uses_output_stem_for_metadata_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_path, sidecar_path, diagnostic_path = resolve_output_paths(
                output=root / "custom.pkl",
                diagnostic_image=root / "custom.png",
            )

            self.assertEqual(output_path, root / "custom.pkl")
            self.assertEqual(sidecar_path, root / "custom_metadata.json")
            self.assertEqual(diagnostic_path, root / "custom.png")

    def test_validate_rejects_legacy_calibrate_pkl_output_name(self) -> None:
        args = build_parser().parse_args(["--output", "calibrate.pkl"])

        with self.assertRaisesRegex(ValueError, "Refusing to overwrite calibrate.pkl"):
            validate_cli_args(args)

    def test_validate_rejects_non_positive_max_reprojection_error(self) -> None:
        args = build_parser().parse_args(["--max-reprojection-error-px", "0"])

        with self.assertRaisesRegex(ValueError, "--max-reprojection-error-px must be > 0"):
            validate_cli_args(args)

    def test_validate_rejects_min_corner_fraction_outside_unit_interval(self) -> None:
        args = build_parser().parse_args(["--min-corner-fraction", "1.5"])

        with self.assertRaisesRegex(ValueError, "--min-corner-fraction must be in"):
            validate_cli_args(args)


if __name__ == "__main__":
    unittest.main()
