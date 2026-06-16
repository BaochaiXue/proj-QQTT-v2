from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
import unittest

import numpy as np

from qqtt.env.camera.calibration_boards import get_calibration_board_config
from qqtt.env.camera.table_calibration import (
    TABLE_CALIBRATE_COMPATIBILITY_CONTRACT,
    TABLE_CALIBRATION_METADATA_SCHEMA_VERSION,
    TABLE_WORLD_FRAME_KIND,
    TableCalibrationLoadError,
    build_table_calibration_metadata,
    load_table_calibration_metadata,
    load_table_calibration_transforms,
    table_calibration_metadata_path_for,
    validate_table_calibration_acceptance,
    write_table_calibration_files,
)


def _sample_metadata(
    *,
    serial_numbers: list[str] | None = None,
    transform_count: int | None = None,
) -> dict:
    serials = ["cam0"] if serial_numbers is None else serial_numbers
    count = len(serials) if transform_count is None else transform_count
    return build_table_calibration_metadata(
        serial_numbers=serials,
        WH=[1280, 720],
        fps=5,
        transform_count=count,
        calibration_board={"name": "calibio-12x9-30mm"},
        max_reprojection_error_px=0.20,
        min_corner_fraction=0.60,
        min_charuco_corners=53,
        per_camera_reprojection_error=[0.10 for _ in range(count)],
        per_camera_corner_count=[60 for _ in range(count)],
        per_camera_corner_fraction=[0.68 for _ in range(count)],
    )


def _write_metadata(path: Path, metadata: dict) -> None:
    table_calibration_metadata_path_for(path).write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def _sample_metadata_kwargs() -> dict:
    return {
        "serial_numbers": ["cam0"],
        "WH": [1280, 720],
        "fps": 5,
        "transform_count": 1,
        "calibration_board": {"name": "calibio-12x9-30mm"},
        "max_reprojection_error_px": 0.20,
        "min_corner_fraction": 0.60,
        "min_charuco_corners": 53,
        "per_camera_reprojection_error": [0.10],
        "per_camera_corner_count": [60],
        "per_camera_corner_fraction": [0.68],
        "distortion_used": True,
        "distortion_model_by_camera": ["inverse_brown_conrady"],
        "distortion_coeffs_by_camera": [[0.0, 0.0, 0.0, 0.0, 0.0]],
    }


class TableCalibrationContractTest(unittest.TestCase):
    def test_metadata_path_uses_output_stem(self) -> None:
        self.assertEqual(
            table_calibration_metadata_path_for(Path("table_calibrate.pkl")),
            Path("table_calibrate_metadata.json"),
        )
        self.assertEqual(
            table_calibration_metadata_path_for(Path("custom.pkl")),
            Path("custom_metadata.json"),
        )

    def test_writer_and_loader_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "table_calibrate.pkl"
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 3] = np.array([0.1, -0.2, 0.3], dtype=np.float32)
            metadata = build_table_calibration_metadata(
                serial_numbers=["239222300781"],
                WH=[1280, 720],
                fps=5,
                transform_count=1,
                calibration_board={"name": "calibio-12x9-30mm"},
                max_reprojection_error_px=0.20,
                min_corner_fraction=0.60,
                min_charuco_corners=53,
                per_camera_reprojection_error=[0.12],
                per_camera_corner_count=[58],
                per_camera_corner_fraction=[0.659],
                distortion_used=True,
                distortion_model_by_camera=["inverse_brown_conrady"],
                distortion_coeffs_by_camera=[[0.0, 0.0, 0.0, 0.0, 0.0]],
                diagnostic_image_path="table_calibrate_diagnostic.png",
            )

            sidecar = write_table_calibration_files(output, [c2w], metadata)

            self.assertTrue(output.is_file())
            self.assertEqual(sidecar, root / "table_calibrate_metadata.json")
            loaded = load_table_calibration_transforms(
                output,
                serial_numbers=["239222300781"],
                table_calibration_reference_serials=["239222300781"],
            )
            np.testing.assert_allclose(loaded[0], c2w, atol=1e-6)
            loaded_metadata = load_table_calibration_metadata(output)
            self.assertEqual(
                loaded_metadata["schema_version"],
                TABLE_CALIBRATION_METADATA_SCHEMA_VERSION,
            )
            self.assertEqual(
                loaded_metadata["compatibility_contract"],
                TABLE_CALIBRATE_COMPATIBILITY_CONTRACT,
            )
            self.assertEqual(loaded_metadata["world_frame_kind"], TABLE_WORLD_FRAME_KIND)
            self.assertEqual(loaded_metadata["diagnostic_image_path"], "table_calibrate_diagnostic.png")

    def test_loader_rejects_missing_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "table_calibrate.pkl"
            with output.open("wb") as handle:
                pickle.dump([np.eye(4, dtype=np.float32)], handle)

            with self.assertRaisesRegex(TableCalibrationLoadError, "Missing table calibration metadata"):
                load_table_calibration_transforms(output)

    def test_loader_rejects_wrong_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "table_calibrate.pkl"
            with output.open("wb") as handle:
                pickle.dump([np.eye(4, dtype=np.float32)], handle)
            table_calibration_metadata_path_for(output).write_text(
                json.dumps({"schema_version": "wrong"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(TableCalibrationLoadError, "Unsupported table calibration metadata schema"):
                load_table_calibration_transforms(output)

    def test_loader_rejects_invalid_transform_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "table_calibrate.pkl"
            metadata = build_table_calibration_metadata(
                serial_numbers=["cam0"],
                WH=[1280, 720],
                fps=5,
                transform_count=1,
                calibration_board={"name": "calibio-12x9-30mm"},
                max_reprojection_error_px=0.20,
                min_corner_fraction=0.60,
                min_charuco_corners=53,
                per_camera_reprojection_error=[0.10],
                per_camera_corner_count=[60],
                per_camera_corner_fraction=[0.68],
            )
            table_calibration_metadata_path_for(output).write_text(json.dumps(metadata), encoding="utf-8")
            with output.open("wb") as handle:
                pickle.dump([np.eye(3, dtype=np.float32)], handle)

            with self.assertRaisesRegex(TableCalibrationLoadError, "Unsupported table calibration transform shape"):
                load_table_calibration_transforms(output)

    def test_acceptance_requires_strict_corner_fraction_and_error(self) -> None:
        board_config = get_calibration_board_config("calibio-12x9-30mm")
        accepted_min = validate_table_calibration_acceptance(
            board_config=board_config,
            corner_count=53,
            reprojection_error_px=0.20,
            max_reprojection_error_px=0.20,
            min_corner_fraction=0.60,
        )
        self.assertEqual(accepted_min["min_charuco_corners"], 53)
        self.assertAlmostEqual(accepted_min["corner_fraction"], 53 / 88)

        with self.assertRaisesRegex(ValueError, "ChArUco corner count"):
            validate_table_calibration_acceptance(
                board_config=board_config,
                corner_count=52,
                reprojection_error_px=0.10,
                max_reprojection_error_px=0.20,
                min_corner_fraction=0.60,
            )

        with self.assertRaisesRegex(ValueError, "reprojection error"):
            validate_table_calibration_acceptance(
                board_config=board_config,
                corner_count=60,
                reprojection_error_px=0.21,
                max_reprojection_error_px=0.20,
                min_corner_fraction=0.60,
            )

    def test_acceptance_enforces_minimum_eleven_charuco_corners(self) -> None:
        board_config = get_calibration_board_config("calibio-12x9-30mm")
        with self.assertRaisesRegex(ValueError, "minimum 11"):
            validate_table_calibration_acceptance(
                board_config=board_config,
                corner_count=10,
                reprojection_error_px=0.10,
                max_reprojection_error_px=0.20,
                min_corner_fraction=0.01,
            )

        accepted = validate_table_calibration_acceptance(
            board_config=board_config,
            corner_count=11,
            reprojection_error_px=0.10,
            max_reprojection_error_px=0.20,
            min_corner_fraction=0.01,
        )
        self.assertEqual(accepted["min_charuco_corners"], 11)
        self.assertAlmostEqual(accepted["corner_fraction"], 11 / 88)

    def test_acceptance_rejects_invalid_numeric_inputs(self) -> None:
        board_config = get_calibration_board_config("calibio-12x9-30mm")
        cases = [
            {
                "kwargs": {"reprojection_error_px": float("nan")},
                "message": "reprojection_error_px must be finite and >= 0",
            },
            {
                "kwargs": {"reprojection_error_px": -0.01},
                "message": "reprojection_error_px must be finite and >= 0",
            },
            {
                "kwargs": {"max_reprojection_error_px": float("inf")},
                "message": "max_reprojection_error_px must be finite and > 0",
            },
            {
                "kwargs": {"max_reprojection_error_px": 0.0},
                "message": "max_reprojection_error_px must be finite and > 0",
            },
            {
                "kwargs": {"min_corner_fraction": float("nan")},
                "message": "min_corner_fraction must be finite and in \\(0, 1\\]",
            },
            {
                "kwargs": {"min_corner_fraction": 0.0},
                "message": "min_corner_fraction must be finite and in \\(0, 1\\]",
            },
            {
                "kwargs": {"min_corner_fraction": 1.01},
                "message": "min_corner_fraction must be finite and in \\(0, 1\\]",
            },
            {
                "kwargs": {"corner_count": -1},
                "message": "corner_count must be finite and >= 0",
            },
            {
                "kwargs": {"corner_count": float("nan")},
                "message": "corner_count must be finite and >= 0",
            },
            {
                "kwargs": {"corner_count": float("inf")},
                "message": "corner_count must be finite and >= 0",
            },
            {
                "kwargs": {"corner_count": 88.5},
                "message": "corner_count must be an integer",
            },
            {
                "kwargs": {"corner_count": 100},
                "message": "corner_count must be <= chessboard_corner_count",
            },
        ]
        defaults = {
            "board_config": board_config,
            "corner_count": 60,
            "reprojection_error_px": 0.10,
            "max_reprojection_error_px": 0.20,
            "min_corner_fraction": 0.60,
        }
        for case in cases:
            with self.subTest(case=case["kwargs"]):
                kwargs = dict(defaults)
                kwargs.update(case["kwargs"])
                with self.assertRaisesRegex(ValueError, case["message"]):
                    validate_table_calibration_acceptance(**kwargs)

    def test_loader_reorders_multi_transform_by_requested_serials(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "table_calibrate.pkl"
            first = np.eye(4, dtype=np.float32)
            first[:3, 3] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            second = np.eye(4, dtype=np.float32)
            second[:3, 3] = np.array([2.0, 0.0, 0.0], dtype=np.float32)
            metadata = _sample_metadata(serial_numbers=["cam_a", "cam_b"])
            write_table_calibration_files(output, [first, second], metadata)

            loaded = load_table_calibration_transforms(
                output,
                serial_numbers=["cam_b", "cam_a"],
            )

            np.testing.assert_allclose(loaded[0], second, atol=1e-6)
            np.testing.assert_allclose(loaded[1], first, atol=1e-6)

    def test_loader_rejects_reference_serial_override_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "table_calibrate.pkl"
            first = np.eye(4, dtype=np.float32)
            first[:3, 3] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            second = np.eye(4, dtype=np.float32)
            second[:3, 3] = np.array([2.0, 0.0, 0.0], dtype=np.float32)
            metadata = _sample_metadata(serial_numbers=["a", "b"])
            write_table_calibration_files(output, [first, second], metadata)

            with self.assertRaisesRegex(
                TableCalibrationLoadError,
                "table_calibration_reference_serials",
            ):
                load_table_calibration_transforms(
                    output,
                    serial_numbers=["b"],
                    table_calibration_reference_serials=["b", "a"],
                )

    def test_loader_rejects_duplicate_and_missing_requested_serials(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "table_calibrate.pkl"
            metadata = _sample_metadata(serial_numbers=["cam_a", "cam_b"])
            write_table_calibration_files(
                output,
                [np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)],
                metadata,
            )

            with self.assertRaisesRegex(TableCalibrationLoadError, "duplicate serials"):
                load_table_calibration_transforms(output, serial_numbers=["cam_a", "cam_a"])
            with self.assertRaisesRegex(TableCalibrationLoadError, "does not cover serials"):
                load_table_calibration_transforms(output, serial_numbers=["cam_c"])

    def test_loader_rejects_transform_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "table_calibrate.pkl"
            _write_metadata(output, _sample_metadata(serial_numbers=["cam_a", "cam_b"]))
            with output.open("wb") as handle:
                pickle.dump([np.eye(4, dtype=np.float32)], handle)

            with self.assertRaisesRegex(TableCalibrationLoadError, "transform count"):
                load_table_calibration_transforms(output)

    def test_loader_rejects_nonfinite_bad_bottom_row_and_singular_transforms(self) -> None:
        cases = [
            ("non-finite", lambda matrix: matrix.__setitem__((0, 0), np.nan)),
            ("invalid homogeneous bottom row", lambda matrix: matrix.__setitem__((3, 3), 2.0)),
            ("singular or degenerate", lambda matrix: matrix.__setitem__((2, 2), 0.0)),
        ]
        for message, mutate in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as tmpdir:
                    output = Path(tmpdir) / "table_calibrate.pkl"
                    _write_metadata(output, _sample_metadata())
                    transform = np.eye(4, dtype=np.float32)
                    mutate(transform)
                    with output.open("wb") as handle:
                        pickle.dump([transform], handle)

                    with self.assertRaisesRegex(TableCalibrationLoadError, message):
                        load_table_calibration_transforms(output)

    def test_writer_and_loader_reject_scaled_or_sheared_transforms(self) -> None:
        cases = [
            ("scaled", lambda matrix: matrix.__setitem__((0, 0), 2.0)),
            ("sheared", lambda matrix: matrix.__setitem__((0, 1), 0.1)),
        ]
        for label, mutate in cases:
            with self.subTest(path="writer", label=label):
                with tempfile.TemporaryDirectory() as tmpdir:
                    output = Path(tmpdir) / "table_calibrate.pkl"
                    transform = np.eye(4, dtype=np.float32)
                    mutate(transform)

                    with self.assertRaisesRegex(TableCalibrationLoadError, "rotation"):
                        write_table_calibration_files(
                            output,
                            [transform],
                            _sample_metadata(),
                        )

            with self.subTest(path="loader", label=label):
                with tempfile.TemporaryDirectory() as tmpdir:
                    output = Path(tmpdir) / "table_calibrate.pkl"
                    _write_metadata(output, _sample_metadata())
                    transform = np.eye(4, dtype=np.float32)
                    mutate(transform)
                    with output.open("wb") as handle:
                        pickle.dump([transform], handle)

                    with self.assertRaisesRegex(TableCalibrationLoadError, "rotation"):
                        load_table_calibration_transforms(output)

    def test_loader_wraps_corrupt_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "table_calibrate.pkl"
            _write_metadata(output, _sample_metadata())
            output.write_bytes(b"not a pickle")

            with self.assertRaisesRegex(TableCalibrationLoadError, "Invalid table calibration pickle"):
                load_table_calibration_transforms(output)

    def test_metadata_loader_rejects_bad_transform_convention(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "table_calibrate.pkl"
            metadata = _sample_metadata()
            metadata["transform_convention"] = "world_to_camera_w2c"
            _write_metadata(output, metadata)

            with self.assertRaisesRegex(TableCalibrationLoadError, "transform_convention"):
                load_table_calibration_metadata(output)

    def test_metadata_loader_rejects_missing_required_fields(self) -> None:
        required_fields = [
            "created_at_utc",
            "serial_numbers",
            "table_calibration_reference_serials",
            "logical_camera_names",
            "WH",
            "fps",
            "transform_count",
            "transform_convention",
            "calibration_board",
            "max_reprojection_error_px",
            "min_corner_fraction",
            "min_charuco_corners",
            "per_camera_reprojection_error",
            "per_camera_corner_count",
            "per_camera_corner_fraction",
        ]
        for field in required_fields:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as tmpdir:
                    output = Path(tmpdir) / "table_calibrate.pkl"
                    metadata = _sample_metadata()
                    del metadata[field]
                    _write_metadata(output, metadata)

                    with self.assertRaisesRegex(TableCalibrationLoadError, field):
                        load_table_calibration_metadata(output)

    def test_metadata_loader_rejects_invalid_core_required_fields(self) -> None:
        cases = [
            ("created_at_utc", ""),
            ("WH", [1280]),
            ("WH", [1280, 0]),
            ("WH", [1280, float("nan")]),
            ("WH", ["1280", 720]),
            ("fps", 0),
            ("fps", 5.5),
            ("fps", float("nan")),
            ("fps", "5"),
            ("transform_count", 0),
            ("transform_count", 1.5),
            ("transform_count", float("inf")),
            ("transform_count", "1"),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                with tempfile.TemporaryDirectory() as tmpdir:
                    output = Path(tmpdir) / "table_calibrate.pkl"
                    metadata = _sample_metadata()
                    metadata[field] = value
                    _write_metadata(output, metadata)

                    with self.assertRaisesRegex(TableCalibrationLoadError, field):
                        load_table_calibration_metadata(output)

    def test_metadata_loader_rejects_nonfinite_numeric_fields(self) -> None:
        cases = [
            ("max_reprojection_error_px", float("nan")),
            ("max_reprojection_error_px", "0.2"),
            ("min_corner_fraction", float("inf")),
            ("min_corner_fraction", "0.6"),
            ("min_charuco_corners", float("nan")),
            ("min_charuco_corners", "53"),
            ("per_camera_reprojection_error", [float("nan")]),
            ("per_camera_reprojection_error", ["0.10"]),
            ("per_camera_corner_count", [float("inf")]),
            ("per_camera_corner_count", ["60"]),
            ("per_camera_corner_fraction", [float("nan")]),
            ("per_camera_corner_fraction", ["0.68"]),
        ]
        for field, value in cases:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as tmpdir:
                    output = Path(tmpdir) / "table_calibrate.pkl"
                    metadata = _sample_metadata()
                    metadata[field] = value
                    _write_metadata(output, metadata)

                    with self.assertRaisesRegex(TableCalibrationLoadError, field):
                        load_table_calibration_metadata(output)

    def test_metadata_loader_rejects_numeric_strings_inside_calibration_board(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "table_calibrate.pkl"
            metadata = _sample_metadata()
            metadata["calibration_board"] = {
                "name": "calibio-12x9-30mm",
                "dictionary_name": "DICT_5X5_250",
                "nested": {"chessboard_corner_count": "88"},
            }
            _write_metadata(output, metadata)

            with self.assertRaisesRegex(TableCalibrationLoadError, "calibration_board"):
                load_table_calibration_metadata(output)

    def test_metadata_loader_rejects_invalid_optional_distortion_fields(self) -> None:
        cases = [
            ("distortion_model_by_camera", [1]),
            ("distortion_model_by_camera", [""]),
            ("distortion_model_by_camera", ["inverse_brown_conrady", "extra"]),
            ("distortion_coeffs_by_camera", [["0.0"]]),
            ("distortion_coeffs_by_camera", [[True]]),
            ("distortion_coeffs_by_camera", [[float("inf")]]),
            ("distortion_coeffs_by_camera", [[0.0], [0.0]]),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                with tempfile.TemporaryDirectory() as tmpdir:
                    output = Path(tmpdir) / "table_calibrate.pkl"
                    metadata = _sample_metadata()
                    metadata[field] = value
                    _write_metadata(output, metadata)

                    with self.assertRaisesRegex(TableCalibrationLoadError, field):
                        load_table_calibration_metadata(output)

    def test_metadata_loader_accepts_valid_optional_distortion_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "table_calibrate.pkl"
            metadata = _sample_metadata(serial_numbers=["cam_a", "cam_b"])
            metadata["distortion_model_by_camera"] = [
                "inverse_brown_conrady",
                None,
            ]
            metadata["distortion_coeffs_by_camera"] = [
                [0.0, 0.1, -0.1],
                None,
            ]
            _write_metadata(output, metadata)

            loaded_metadata = load_table_calibration_metadata(output)

            self.assertEqual(
                loaded_metadata["distortion_coeffs_by_camera"],
                [[0.0, 0.1, -0.1], None],
            )

    def test_metadata_loader_rejects_threshold_failing_metadata(self) -> None:
        cases = [
            (
                "per_camera_reprojection_error",
                lambda metadata: metadata.update(per_camera_reprojection_error=[0.21]),
            ),
            (
                "per_camera_corner_count",
                lambda metadata: metadata.update(per_camera_corner_count=[52]),
            ),
            (
                "per_camera_corner_fraction",
                lambda metadata: metadata.update(per_camera_corner_fraction=[0.59]),
            ),
            (
                "per_camera_corner_count",
                lambda metadata: (
                    metadata["calibration_board"].update(chessboard_corner_count=88),
                    metadata.update(per_camera_corner_count=[89]),
                ),
            ),
            (
                "min_charuco_corners",
                lambda metadata: metadata.update(
                    calibration_board={
                        "name": "small-board",
                        "chessboard_corner_count": 20,
                    },
                    min_corner_fraction=0.60,
                    min_charuco_corners=11,
                    per_camera_corner_count=[12],
                    per_camera_corner_fraction=[0.60],
                ),
            ),
            (
                "required=53",
                lambda metadata: metadata.update(
                    calibration_board={"name": "calibio-12x9-30mm"},
                    min_corner_fraction=0.60,
                    min_charuco_corners=11,
                    per_camera_corner_count=[11],
                    per_camera_corner_fraction=[0.60],
                ),
            ),
        ]
        for field, mutate in cases:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as tmpdir:
                    output = Path(tmpdir) / "table_calibrate.pkl"
                    metadata = _sample_metadata()
                    mutate(metadata)
                    _write_metadata(output, metadata)

                    with self.assertRaisesRegex(TableCalibrationLoadError, field):
                        load_table_calibration_metadata(output)

    def test_builder_rejects_threshold_failing_metadata(self) -> None:
        cases = [
            (
                "per_camera_reprojection_error",
                lambda kwargs: kwargs.update(per_camera_reprojection_error=[0.21]),
            ),
            (
                "per_camera_corner_count",
                lambda kwargs: kwargs.update(per_camera_corner_count=[52]),
            ),
            (
                "per_camera_corner_fraction",
                lambda kwargs: kwargs.update(per_camera_corner_fraction=[0.59]),
            ),
            (
                "per_camera_corner_count",
                lambda kwargs: kwargs.update(
                    calibration_board={
                        "name": "small-board",
                        "chessboard_corner_count": 20,
                    },
                    min_corner_fraction=0.50,
                    min_charuco_corners=11,
                    per_camera_corner_count=[21],
                    per_camera_corner_fraction=[0.50],
                ),
            ),
            (
                "min_charuco_corners",
                lambda kwargs: kwargs.update(
                    calibration_board={
                        "name": "small-board",
                        "chessboard_corner_count": 20,
                    },
                    min_corner_fraction=0.60,
                    min_charuco_corners=11,
                    per_camera_corner_count=[12],
                    per_camera_corner_fraction=[0.60],
                ),
            ),
            (
                "required=53",
                lambda kwargs: kwargs.update(
                    calibration_board={"name": "calibio-12x9-30mm"},
                    min_corner_fraction=0.60,
                    min_charuco_corners=11,
                    per_camera_corner_count=[11],
                    per_camera_corner_fraction=[0.60],
                ),
            ),
        ]
        for field, mutate in cases:
            with self.subTest(field=field):
                kwargs = _sample_metadata_kwargs()
                mutate(kwargs)

                with self.assertRaisesRegex(ValueError, field):
                    build_table_calibration_metadata(**kwargs)

    def test_builder_rejects_string_serial_numbers(self) -> None:
        kwargs = _sample_metadata_kwargs()
        kwargs.update(
            serial_numbers="cam0",
            transform_count=4,
            per_camera_reprojection_error=[0.10, 0.10, 0.10, 0.10],
            per_camera_corner_count=[60, 60, 60, 60],
            per_camera_corner_fraction=[0.68, 0.68, 0.68, 0.68],
            distortion_model_by_camera=[
                "inverse_brown_conrady",
                "inverse_brown_conrady",
                "inverse_brown_conrady",
                "inverse_brown_conrady",
            ],
            distortion_coeffs_by_camera=[
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        )

        with self.assertRaisesRegex(ValueError, "serial_numbers"):
            build_table_calibration_metadata(**kwargs)

    def test_distortion_used_must_be_strict_bool(self) -> None:
        for value in ["false", 1, 0, None]:
            with self.subTest(value=value):
                with tempfile.TemporaryDirectory() as tmpdir:
                    output = Path(tmpdir) / "table_calibrate.pkl"
                    metadata = _sample_metadata()
                    metadata["distortion_used"] = value
                    _write_metadata(output, metadata)

                    with self.assertRaisesRegex(TableCalibrationLoadError, "distortion_used"):
                        load_table_calibration_metadata(output)

        for value in ["false", 1, 0]:
            with self.subTest(builder_value=value):
                kwargs = _sample_metadata_kwargs()
                kwargs["distortion_used"] = value

                with self.assertRaisesRegex(ValueError, "distortion_used"):
                    build_table_calibration_metadata(**kwargs)

    def test_metadata_loader_rejects_contradictory_serial_orders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "table_calibrate.pkl"
            metadata = _sample_metadata(serial_numbers=["cam_a", "cam_b"])
            metadata["serial_numbers"] = ["cam_b", "cam_a"]
            _write_metadata(output, metadata)

            with self.assertRaisesRegex(TableCalibrationLoadError, "serial_numbers"):
                load_table_calibration_metadata(output)

    def test_builder_accepts_finite_numpy_scalar_numeric_inputs(self) -> None:
        metadata = build_table_calibration_metadata(
            serial_numbers=["cam0"],
            WH=[np.int64(1280), np.int32(720)],
            fps=np.int64(5),
            transform_count=np.int64(1),
            calibration_board={
                "name": "calibio-12x9-30mm",
                "chessboard_corner_count": np.int64(88),
            },
            max_reprojection_error_px=np.float32(0.20),
            min_corner_fraction=np.float32(0.60),
            min_charuco_corners=np.int64(53),
            per_camera_reprojection_error=[np.float32(0.10)],
            per_camera_corner_count=[np.int64(60)],
            per_camera_corner_fraction=[np.float64(0.68)],
            distortion_used=True,
            distortion_coeffs_by_camera=[[np.float32(0.0)]],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "table_calibrate.pkl"
            sidecar = write_table_calibration_files(
                output,
                [np.eye(4, dtype=np.float32)],
                metadata,
            )
            loaded = json.loads(sidecar.read_text(encoding="utf-8"))

        self.assertEqual(loaded["WH"], [1280, 720])
        self.assertEqual(loaded["fps"], 5)
        self.assertEqual(loaded["calibration_board"]["chessboard_corner_count"], 88)
        self.assertEqual(loaded["distortion_coeffs_by_camera"], [[0.0]])

    def test_builder_rejects_numeric_string_inputs(self) -> None:
        cases = [
            ("WH", lambda kwargs: kwargs.update(WH=["1280", 720])),
            ("fps", lambda kwargs: kwargs.update(fps="5")),
            ("transform_count", lambda kwargs: kwargs.update(transform_count="1")),
            (
                "calibration_board",
                lambda kwargs: kwargs.update(
                    calibration_board={
                        "name": "calibio-12x9-30mm",
                        "chessboard_corner_count": "88",
                    }
                ),
            ),
            (
                "max_reprojection_error_px",
                lambda kwargs: kwargs.update(max_reprojection_error_px="0.2"),
            ),
            (
                "min_corner_fraction",
                lambda kwargs: kwargs.update(min_corner_fraction="0.6"),
            ),
            (
                "min_charuco_corners",
                lambda kwargs: kwargs.update(min_charuco_corners="53"),
            ),
            (
                "per_camera_reprojection_error",
                lambda kwargs: kwargs.update(per_camera_reprojection_error=["0.10"]),
            ),
            (
                "per_camera_corner_count",
                lambda kwargs: kwargs.update(per_camera_corner_count=["60"]),
            ),
            (
                "per_camera_corner_fraction",
                lambda kwargs: kwargs.update(per_camera_corner_fraction=["0.68"]),
            ),
            (
                "distortion_coeffs_by_camera",
                lambda kwargs: kwargs.update(distortion_coeffs_by_camera=[["0.0"]]),
            ),
        ]
        for field, mutate in cases:
            with self.subTest(field=field):
                kwargs = _sample_metadata_kwargs()
                mutate(kwargs)

                with self.assertRaisesRegex(ValueError, field):
                    build_table_calibration_metadata(**kwargs)

    def test_metadata_loader_wraps_invalid_utf8_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "table_calibrate.pkl"
            table_calibration_metadata_path_for(output).write_bytes(b"\xff\xfe")

            with self.assertRaisesRegex(
                TableCalibrationLoadError,
                "Invalid table calibration metadata JSON",
            ):
                load_table_calibration_metadata(output)

    def test_metadata_loader_rejects_per_camera_and_logical_name_length_mismatch(self) -> None:
        cases = [
            ("per_camera_reprojection_error", [0.10]),
            ("per_camera_corner_count", [60]),
            ("per_camera_corner_fraction", [0.68]),
            ("logical_camera_names", ["cam0"]),
        ]
        for field, value in cases:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as tmpdir:
                    output = Path(tmpdir) / "table_calibrate.pkl"
                    metadata = _sample_metadata(serial_numbers=["cam_a", "cam_b"])
                    metadata[field] = value
                    _write_metadata(output, metadata)

                    with self.assertRaisesRegex(TableCalibrationLoadError, field):
                        load_table_calibration_metadata(output)

    def test_writer_rejects_nan_json_sidecar_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "table_calibrate.pkl"
            metadata = _sample_metadata()
            metadata["diagnostic_image_path"] = float("nan")

            with self.assertRaises(ValueError):
                write_table_calibration_files(output, [np.eye(4, dtype=np.float32)], metadata)


if __name__ == "__main__":
    unittest.main()
