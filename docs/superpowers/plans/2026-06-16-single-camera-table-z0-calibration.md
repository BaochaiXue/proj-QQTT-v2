# Single-Camera Table Z0 Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strict one-shot single-camera ChArUco tabletop calibration tool that writes `table_calibrate.pkl` and `table_calibrate_metadata.json` without changing the existing `calibrate.pkl` default.

**Architecture:** Add a focused table calibration contract module under `qqtt/env/camera/`, then add a root CLI that reuses the existing camera and ChArUco board stack. Downstream tools get explicit table calibration path handling through small loader/copy integrations, while existing behavior remains unchanged when the user does not pass table calibration arguments.

**Tech Stack:** Python, argparse, OpenCV ChArUco APIs, pyrealsense2 via existing `CameraSystem`, pickle, JSON metadata, `unittest`, existing `demo_2_max` conda environment.

---

## Scope And File Structure

Create and modify these files:

- Create `qqtt/env/camera/table_calibration.py`
  - Owns table calibration schema constants, metadata sidecar path resolution, transform validation, metadata build/write/load helpers, and a pure ChArUco estimate helper.
- Create `cameras_calibrate_table.py`
  - Root CLI for strict one-shot tabletop calibration.
- Modify `record_data.py`
  - Add `--table-calibrate` and copy validated table calibration artifacts into raw cases.
- Modify `data_process/record_data_align.py`
  - Preserve validated table calibration artifacts from raw case to aligned case.
- Modify `qqtt/demo/single_demo_v3_runtime.py`
  - Add `--table-calibrate` validation and dry-run/delegate contract fields so demos can explicitly load the table frame contract.
- Create `tests/test_table_calibration_contract.py`
  - Unit tests for metadata, writer, loader, validation, and ChArUco threshold behavior.
- Create `tests/test_cameras_calibrate_table_cli.py`
  - Parser and no-hardware runtime tests for the new CLI.
- Create `tests/test_record_data_table_calibration.py`
  - Recording copy/metadata tests.
- Create `tests/test_record_data_align_table_calibration.py`
  - Aligned case table calibration preservation tests.
- Modify `tests/test_single_demo_v3_runtime.py`
  - Demo 3.x dry-run contract tests for explicit table calibration loading.
- Modify `docs/WORKFLOWS.md`
  - Add the operator command and usage notes.
- Modify `docs/HARDWARE_VALIDATION.md`
  - Add manual checklist for table Z0 calibration.
- Modify `docs/ARCHITECTURE.md`
  - Document table calibration as a separate runtime artifact.

Keep this implementation out of experiment-only visualization packages.

## Task 1: Table Calibration Contract Tests

**Files:**
- Create: `tests/test_table_calibration_contract.py`
- Create later in Task 2: `qqtt/env/camera/table_calibration.py`

- [ ] **Step 1: Write failing tests for table calibration metadata, writer, and loader**

Create `tests/test_table_calibration_contract.py` with this content:

```python
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
                min_charuco_corners=52,
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
            root = Path(tmpdir)
            output = root / "table_calibrate.pkl"
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
                min_charuco_corners=52,
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
        self.assertEqual(accepted_min["min_charuco_corners"], 52)
        self.assertAlmostEqual(accepted_min["corner_fraction"], 53 / 88)

        with self.assertRaisesRegex(ValueError, "ChArUco corner count"):
            validate_table_calibration_acceptance(
                board_config=board_config,
                corner_count=51,
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the contract tests and verify they fail on missing module symbols**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest tests.test_table_calibration_contract
```

Expected: FAIL with an import error for `qqtt.env.camera.table_calibration`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_table_calibration_contract.py
git commit -m "test: cover table calibration contract"
```

## Task 2: Table Calibration Contract Module

**Files:**
- Create: `qqtt/env/camera/table_calibration.py`
- Test: `tests/test_table_calibration_contract.py`

- [ ] **Step 1: Implement the contract module**

Create `qqtt/env/camera/table_calibration.py` with these public constants and functions:

```python
from __future__ import annotations

from datetime import datetime, timezone
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


TABLE_CALIBRATION_METADATA_SCHEMA_VERSION = "qqtt_table_calibration_v1"
TABLE_CALIBRATE_COMPATIBILITY_CONTRACT = "qqtt_table_calibrate_c2w_v1"
TABLE_WORLD_FRAME_KIND = "table_world_z0"


class TableCalibrationLoadError(RuntimeError):
    pass


def table_calibration_metadata_path_for(table_calibrate_path: str | Path) -> Path:
    path = Path(table_calibrate_path)
    return path.with_name(f"{path.stem}_metadata.json")


def _validate_serials(name: str, serials: Any) -> list[str]:
    if not isinstance(serials, list) or not serials:
        raise TableCalibrationLoadError(f"{name} must be a non-empty serial list.")
    if not all(isinstance(item, str) and item for item in serials):
        raise TableCalibrationLoadError(f"{name} must contain non-empty string serials.")
    duplicates = sorted({item for item in serials if serials.count(item) > 1})
    if duplicates:
        raise TableCalibrationLoadError(f"{name} contains duplicate serials: {duplicates}")
    return list(serials)


def _metadata_float_list(values: list[float] | None) -> list[float] | None:
    if values is None:
        return None
    return [float(item) for item in values]


def build_table_calibration_metadata(
    *,
    serial_numbers: list[str],
    WH,
    fps: int,
    transform_count: int,
    calibration_board: dict[str, Any],
    max_reprojection_error_px: float,
    min_corner_fraction: float,
    min_charuco_corners: int,
    per_camera_reprojection_error: list[float],
    per_camera_corner_count: list[int],
    per_camera_corner_fraction: list[float],
    distortion_used: bool | None = None,
    distortion_model_by_camera: list[str | None] | None = None,
    distortion_coeffs_by_camera: list[list[float] | None] | None = None,
    diagnostic_image_path: str | None = None,
) -> dict[str, Any]:
    serials = _validate_serials("serial_numbers", list(serial_numbers))
    if int(transform_count) != len(serials):
        raise ValueError("transform_count must match serial_numbers length")
    metadata: dict[str, Any] = {
        "schema_version": TABLE_CALIBRATION_METADATA_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "serial_numbers": serials,
        "table_calibration_reference_serials": serials,
        "logical_camera_names": [f"cam{i}" for i in range(len(serials))],
        "WH": list(WH),
        "fps": int(fps),
        "transform_count": int(transform_count),
        "transform_convention": "camera_to_world_c2w",
        "world_frame_kind": TABLE_WORLD_FRAME_KIND,
        "compatibility_contract": TABLE_CALIBRATE_COMPATIBILITY_CONTRACT,
        "calibration_board": dict(calibration_board),
        "max_reprojection_error_px": float(max_reprojection_error_px),
        "min_corner_fraction": float(min_corner_fraction),
        "min_charuco_corners": int(min_charuco_corners),
        "per_camera_reprojection_error": [float(item) for item in per_camera_reprojection_error],
        "per_camera_corner_count": [int(item) for item in per_camera_corner_count],
        "per_camera_corner_fraction": [float(item) for item in per_camera_corner_fraction],
    }
    if distortion_used is not None:
        metadata["distortion_used"] = bool(distortion_used)
    if distortion_model_by_camera is not None:
        metadata["distortion_model_by_camera"] = list(distortion_model_by_camera)
    if distortion_coeffs_by_camera is not None:
        metadata["distortion_coeffs_by_camera"] = [
            None if item is None else [float(value) for value in item]
            for item in distortion_coeffs_by_camera
        ]
    if diagnostic_image_path is not None:
        metadata["diagnostic_image_path"] = str(diagnostic_image_path)
    return metadata


def _validate_transform_matrix(matrix: np.ndarray, *, index: int) -> np.ndarray:
    item = np.asarray(matrix, dtype=np.float32)
    if item.shape != (4, 4):
        raise TableCalibrationLoadError(f"Unsupported table calibration transform shape at index {index}: {item.shape}")
    if not np.all(np.isfinite(item)):
        raise TableCalibrationLoadError(f"Table calibration transform at index {index} contains non-finite values.")
    expected_bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if not np.allclose(item[3], expected_bottom, atol=1e-4):
        raise TableCalibrationLoadError(f"Table calibration transform at index {index} has invalid homogeneous bottom row.")
    if abs(float(np.linalg.det(item[:3, :3]))) <= 1e-6:
        raise TableCalibrationLoadError(f"Table calibration transform at index {index} is singular or degenerate.")
    return item


def _coerce_transform_list(raw: Any) -> list[np.ndarray]:
    if isinstance(raw, np.ndarray):
        if raw.ndim == 3 and raw.shape[1:] == (4, 4):
            return [_validate_transform_matrix(item, index=idx) for idx, item in enumerate(raw)]
        raise TableCalibrationLoadError(f"Unsupported table calibration ndarray shape: {raw.shape}")
    if isinstance(raw, (list, tuple)):
        transforms = [_validate_transform_matrix(item, index=idx) for idx, item in enumerate(raw)]
        if not transforms:
            raise TableCalibrationLoadError("Table calibration transform list is empty.")
        return transforms
    raise TableCalibrationLoadError(f"Unsupported table calibration object type: {type(raw).__name__}")


def write_table_calibration_files(
    table_calibrate_path: str | Path,
    c2w_list: list[np.ndarray],
    metadata: dict[str, Any],
) -> Path:
    output_path = Path(table_calibrate_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transforms = [_validate_transform_matrix(item, index=idx) for idx, item in enumerate(c2w_list)]
    if int(metadata.get("transform_count", -1)) != len(transforms):
        raise ValueError("table calibration metadata transform_count does not match transforms")
    with output_path.open("wb") as handle:
        pickle.dump(transforms, handle)
    sidecar_path = table_calibration_metadata_path_for(output_path)
    sidecar_path.write_text(json.dumps(metadata), encoding="utf-8")
    return sidecar_path


def load_table_calibration_metadata(table_calibrate_path: str | Path) -> dict[str, Any]:
    sidecar_path = table_calibration_metadata_path_for(table_calibrate_path)
    if not sidecar_path.is_file():
        raise TableCalibrationLoadError(f"Missing table calibration metadata: {sidecar_path}")
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != TABLE_CALIBRATION_METADATA_SCHEMA_VERSION:
        raise TableCalibrationLoadError(
            f"Unsupported table calibration metadata schema in {sidecar_path}: {metadata.get('schema_version')!r}"
        )
    if metadata.get("compatibility_contract") != TABLE_CALIBRATE_COMPATIBILITY_CONTRACT:
        raise TableCalibrationLoadError(
            f"Unsupported table calibration compatibility contract: {metadata.get('compatibility_contract')!r}"
        )
    if metadata.get("world_frame_kind") != TABLE_WORLD_FRAME_KIND:
        raise TableCalibrationLoadError(f"Unsupported table world frame kind: {metadata.get('world_frame_kind')!r}")
    serials = _validate_serials("table calibration serial_numbers", metadata.get("serial_numbers"))
    reference_serials = _validate_serials(
        "table calibration reference serials",
        metadata.get("table_calibration_reference_serials", serials),
    )
    if int(metadata.get("transform_count", len(reference_serials))) != len(reference_serials):
        raise TableCalibrationLoadError("table calibration transform_count does not match reference serials")
    return metadata


def load_table_calibration_transforms(
    table_calibrate_path: str | Path,
    *,
    serial_numbers: list[str] | None = None,
    table_calibration_reference_serials: list[str] | None = None,
) -> list[np.ndarray]:
    path = Path(table_calibrate_path)
    if not path.is_file():
        raise TableCalibrationLoadError(f"Missing table calibration file: {path}")
    metadata = load_table_calibration_metadata(path)
    reference_serials = list(table_calibration_reference_serials or metadata["table_calibration_reference_serials"])
    with path.open("rb") as handle:
        transforms = _coerce_transform_list(pickle.load(handle))
    if len(transforms) != len(reference_serials):
        raise TableCalibrationLoadError("table calibration transform count does not match reference serials")
    if serial_numbers is None:
        return transforms
    _validate_serials("serial_numbers", list(serial_numbers))
    index_by_serial = {serial: idx for idx, serial in enumerate(reference_serials)}
    missing = [serial for serial in serial_numbers if serial not in index_by_serial]
    if missing:
        raise TableCalibrationLoadError(f"Table calibration does not cover serials: {missing}")
    return [transforms[index_by_serial[serial]] for serial in serial_numbers]


def validate_table_calibration_acceptance(
    *,
    board_config,
    corner_count: int,
    reprojection_error_px: float,
    max_reprojection_error_px: float,
    min_corner_fraction: float,
) -> dict[str, float | int]:
    board_corners = int(board_config.chessboard_corner_count)
    min_by_fraction = int(np.ceil(float(min_corner_fraction) * float(board_corners)))
    min_charuco_corners = max(11, min_by_fraction)
    corner_fraction = float(corner_count) / float(board_corners) if board_corners else 0.0
    if int(corner_count) < min_charuco_corners:
        raise ValueError(
            f"ChArUco corner count {int(corner_count)} is below strict table calibration minimum {min_charuco_corners}."
        )
    if float(reprojection_error_px) > float(max_reprojection_error_px):
        raise ValueError(
            f"ChArUco reprojection error {float(reprojection_error_px):.6f}px exceeds strict table calibration maximum "
            f"{float(max_reprojection_error_px):.6f}px."
        )
    return {
        "min_charuco_corners": int(min_charuco_corners),
        "corner_fraction": float(corner_fraction),
    }
```

- [ ] **Step 2: Run the contract tests**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest tests.test_table_calibration_contract
```

Expected: PASS.

- [ ] **Step 3: Commit the passing contract module**

```bash
git add qqtt/env/camera/table_calibration.py tests/test_table_calibration_contract.py
git commit -m "feat: add table calibration contract"
```

## Task 3: One-Shot Table Calibration CLI Tests

**Files:**
- Create: `tests/test_cameras_calibrate_table_cli.py`
- Create later in Task 4: `cameras_calibrate_table.py`

- [ ] **Step 1: Write parser and output safety tests**

Create `tests/test_cameras_calibrate_table_cli.py` with this content:

```python
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from cameras_calibrate_table import build_parser, resolve_output_paths, validate_cli_args


class CamerasCalibrateTableCliTest(unittest.TestCase):
    def test_defaults_write_table_calibrate_in_repo_root(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.output, Path("table_calibrate.pkl"))
        self.assertEqual(args.diagnostic_image, Path("table_calibrate_diagnostic.png"))
        self.assertEqual(args.max_reprojection_error_px, 0.20)
        self.assertEqual(args.min_corner_fraction, 0.60)
        self.assertEqual(args.fps, 5)

    def test_output_paths_resolve_sidecar_from_output_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output, sidecar, diagnostic = resolve_output_paths(
                output=root / "custom.pkl",
                diagnostic_image=root / "custom.png",
            )
            self.assertEqual(output, root / "custom.pkl")
            self.assertEqual(sidecar, root / "custom_metadata.json")
            self.assertEqual(diagnostic, root / "custom.png")

    def test_validate_rejects_calibrate_pkl_output(self) -> None:
        args = build_parser().parse_args(["--output", "calibrate.pkl"])
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite calibrate.pkl"):
            validate_cli_args(args)

    def test_validate_rejects_non_positive_thresholds(self) -> None:
        args = build_parser().parse_args(["--max-reprojection-error-px", "0"])
        with self.assertRaisesRegex(ValueError, "--max-reprojection-error-px must be > 0"):
            validate_cli_args(args)

        args = build_parser().parse_args(["--min-corner-fraction", "1.5"])
        with self.assertRaisesRegex(ValueError, "--min-corner-fraction must be in"):
            validate_cli_args(args)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the CLI tests and verify they fail on missing entrypoint**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest tests.test_cameras_calibrate_table_cli
```

Expected: FAIL with `ModuleNotFoundError: No module named 'cameras_calibrate_table'`.

- [ ] **Step 3: Commit the failing CLI tests**

```bash
git add tests/test_cameras_calibrate_table_cli.py
git commit -m "test: cover table calibration cli"
```

## Task 4: One-Shot Table Calibration CLI Implementation

**Files:**
- Create: `cameras_calibrate_table.py`
- Modify: `qqtt/env/camera/table_calibration.py`
- Test: `tests/test_cameras_calibrate_table_cli.py`
- Test: `tests/test_table_calibration_contract.py`

- [ ] **Step 1: Add ChArUco estimate helper to `table_calibration.py`**

Append these helpers to `qqtt/env/camera/table_calibration.py`:

```python
def _dist_coeffs_to_metadata(coeffs) -> list[float] | None:
    if coeffs is None:
        return None
    return [float(value) for value in np.asarray(coeffs, dtype=np.float64).reshape(-1)]


def estimate_table_c2w_from_charuco_image(
    *,
    color_bgr: np.ndarray,
    intrinsic: np.ndarray,
    dist_coeffs,
    board_config,
    dictionary,
    board,
    max_reprojection_error_px: float,
    min_corner_fraction: float,
) -> dict[str, Any]:
    import cv2

    corners, ids, _ = cv2.aruco.detectMarkers(
        image=color_bgr,
        dictionary=dictionary,
        parameters=None,
    )
    if ids is None or len(corners) == 0:
        raise ValueError("No ArUco markers detected for table calibration.")
    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=color_bgr,
        board=board,
        cameraMatrix=intrinsic,
        distCoeffs=dist_coeffs,
    )
    if charuco_corners is None or charuco_ids is None or len(charuco_corners) == 0:
        raise ValueError("No ChArUco corners detected for table calibration.")
    rvec = None
    tvec = None
    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners,
        charuco_ids,
        board,
        intrinsic,
        dist_coeffs,
        rvec=rvec,
        tvec=tvec,
    )
    if (not retval) or (rvec is None) or (tvec is None):
        raise ValueError("Failed to estimate ChArUco table pose.")

    from qqtt.env.camera.calibration_boards import get_charuco_chessboard_corners

    charuco_id_values = charuco_ids.reshape(-1)
    reprojected, _ = cv2.projectPoints(
        get_charuco_chessboard_corners(board)[charuco_id_values, :],
        rvec,
        tvec,
        intrinsic,
        dist_coeffs,
    )
    reprojected = reprojected.reshape(-1, 2)
    observed = charuco_corners.reshape(-1, 2)
    reprojection_error = float(np.sqrt(np.sum((reprojected - observed) ** 2, axis=1)).mean())
    acceptance = validate_table_calibration_acceptance(
        board_config=board_config,
        corner_count=int(len(observed)),
        reprojection_error_px=reprojection_error,
        max_reprojection_error_px=float(max_reprojection_error_px),
        min_corner_fraction=float(min_corner_fraction),
    )
    R_board2cam = cv2.Rodrigues(rvec)[0]
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R_board2cam.astype(np.float32)
    w2c[:3, 3] = tvec[:, 0].astype(np.float32)
    c2w = np.linalg.inv(w2c).astype(np.float32)

    diagnostic = color_bgr.copy()
    cv2.aruco.drawDetectedMarkers(diagnostic, corners, ids)
    cv2.aruco.drawDetectedCornersCharuco(
        image=diagnostic,
        charucoCorners=observed.reshape(-1, 1, 2),
        charucoIds=charuco_id_values.reshape(-1, 1),
    )
    cv2.drawFrameAxes(diagnostic, intrinsic, dist_coeffs, rvec, tvec, 0.1)

    return {
        "c2w": c2w,
        "reprojection_error_px": reprojection_error,
        "corner_count": int(len(observed)),
        "corner_fraction": float(acceptance["corner_fraction"]),
        "min_charuco_corners": int(acceptance["min_charuco_corners"]),
        "diagnostic_bgr": diagnostic,
    }
```

- [ ] **Step 2: Implement `cameras_calibrate_table.py`**

Create `cameras_calibrate_table.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from qqtt.env.camera.calibration_boards import (
    DEFAULT_CALIBRATION_BOARD,
    available_calibration_boards,
    charuco_board_config_to_metadata,
    create_charuco_board,
    get_calibration_board_config,
)
from qqtt.env.camera.defaults import DEFAULT_EXPOSURE, DEFAULT_GAIN
from qqtt.env.camera.table_calibration import (
    build_table_calibration_metadata,
    estimate_table_c2w_from_charuco_image,
    table_calibration_metadata_path_for,
    write_table_calibration_files,
)


TABLE_CALIBRATE_DEFAULT_WIDTH = 1280
TABLE_CALIBRATE_DEFAULT_HEIGHT = 720
TABLE_CALIBRATE_DEFAULT_FPS = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict one-shot single-camera table Z=0 calibration with a flat ChArUco board.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--width", type=int, default=TABLE_CALIBRATE_DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=TABLE_CALIBRATE_DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=TABLE_CALIBRATE_DEFAULT_FPS)
    parser.add_argument("--serial", default=None, help="Single RealSense serial to calibrate.")
    parser.add_argument("--exposure", type=float, default=DEFAULT_EXPOSURE)
    parser.add_argument("--gain", type=float, default=DEFAULT_GAIN)
    parser.add_argument("--calibration-board", choices=available_calibration_boards(), default=DEFAULT_CALIBRATION_BOARD)
    parser.add_argument("--board-squares-x", type=int, default=None)
    parser.add_argument("--board-squares-y", type=int, default=None)
    parser.add_argument("--board-square-size-mm", type=float, default=None)
    parser.add_argument("--board-marker-size-mm", type=float, default=None)
    parser.add_argument("--board-dictionary", default=None)
    parser.add_argument("--max-reprojection-error-px", type=float, default=0.20)
    parser.add_argument("--min-corner-fraction", type=float, default=0.60)
    parser.add_argument("--output", type=Path, default=Path("table_calibrate.pkl"))
    parser.add_argument("--diagnostic-image", type=Path, default=Path("table_calibrate_diagnostic.png"))
    parser.add_argument("--disable-keyboard-listener", action="store_true", default=True)
    return parser


def resolve_board_config_from_args(args: argparse.Namespace):
    return get_calibration_board_config(args.calibration_board).with_overrides(
        squares_x=args.board_squares_x,
        squares_y=args.board_squares_y,
        square_size_mm=args.board_square_size_mm,
        marker_size_mm=args.board_marker_size_mm,
        dictionary_name=args.board_dictionary,
    )


def resolve_output_paths(*, output: Path, diagnostic_image: Path | None) -> tuple[Path, Path, Path | None]:
    output_path = Path(output).resolve()
    sidecar_path = table_calibration_metadata_path_for(output_path)
    diagnostic_path = None if diagnostic_image is None else Path(diagnostic_image).resolve()
    return output_path, sidecar_path, diagnostic_path


def validate_cli_args(args: argparse.Namespace) -> None:
    if Path(args.output).name == "calibrate.pkl":
        raise ValueError("Refusing to overwrite calibrate.pkl with table calibration output.")
    if float(args.max_reprojection_error_px) <= 0.0:
        raise ValueError("--max-reprojection-error-px must be > 0")
    if not (0.0 < float(args.min_corner_fraction) <= 1.0):
        raise ValueError("--min-corner-fraction must be in (0, 1]")
    if int(args.width) <= 0 or int(args.height) <= 0:
        raise ValueError("--width and --height must be positive")
    if int(args.fps) <= 0:
        raise ValueError("--fps must be positive")


def _dist_coeffs_from_metadata(metadata: dict, key: str):
    import numpy as np

    coeffs = metadata.get(key)
    if coeffs is None:
        return None
    arr = np.asarray(coeffs, dtype=np.float64).reshape(-1, 1)
    return None if arr.size == 0 else arr


def _dist_coeffs_to_metadata(coeffs):
    import numpy as np

    if coeffs is None:
        return None
    return [float(value) for value in np.asarray(coeffs, dtype=np.float64).reshape(-1)]


def run_table_calibration(args: argparse.Namespace) -> tuple[Path, Path]:
    validate_cli_args(args)
    from qqtt.env import CameraSystem

    board_config = resolve_board_config_from_args(args)
    dictionary, board = create_charuco_board(board_config)
    output_path, sidecar_path, diagnostic_path = resolve_output_paths(
        output=args.output,
        diagnostic_image=args.diagnostic_image,
    )

    camera_system = CameraSystem(
        WH=[args.width, args.height],
        fps=args.fps,
        num_cam=1,
        serial_numbers=[args.serial] if args.serial else None,
        capture_mode="color",
        exposure=args.exposure,
        gain=args.gain,
        enable_keyboard_listener=False,
    )
    try:
        intrinsics = camera_system.realsense.get_intrinsics()
        obs = camera_system.get_observation()
        color_bgr = obs[0]["color"]
        stream_metadata = list(camera_system.stream_metadata)
        dist_coeffs = _dist_coeffs_from_metadata(stream_metadata[0], "color_distortion_coeffs")
        estimate = estimate_table_c2w_from_charuco_image(
            color_bgr=color_bgr,
            intrinsic=intrinsics[0],
            dist_coeffs=dist_coeffs,
            board_config=board_config,
            dictionary=dictionary,
            board=board,
            max_reprojection_error_px=args.max_reprojection_error_px,
            min_corner_fraction=args.min_corner_fraction,
        )
        if diagnostic_path is not None:
            import cv2

            diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(diagnostic_path), estimate["diagnostic_bgr"])
        metadata = build_table_calibration_metadata(
            serial_numbers=camera_system.serial_numbers,
            WH=camera_system.WH,
            fps=int(camera_system.fps),
            transform_count=1,
            calibration_board=charuco_board_config_to_metadata(board_config),
            max_reprojection_error_px=float(args.max_reprojection_error_px),
            min_corner_fraction=float(args.min_corner_fraction),
            min_charuco_corners=int(estimate["min_charuco_corners"]),
            per_camera_reprojection_error=[float(estimate["reprojection_error_px"])],
            per_camera_corner_count=[int(estimate["corner_count"])],
            per_camera_corner_fraction=[float(estimate["corner_fraction"])],
            distortion_used=dist_coeffs is not None,
            distortion_model_by_camera=[stream_metadata[0].get("color_distortion_model")],
            distortion_coeffs_by_camera=[_dist_coeffs_to_metadata(dist_coeffs)],
            diagnostic_image_path=None if diagnostic_path is None else str(diagnostic_path),
        )
        write_table_calibration_files(output_path, [estimate["c2w"]], metadata)
        print(f"[table-calibrate] wrote {output_path} and {sidecar_path}")
        if diagnostic_path is not None:
            print(f"[table-calibrate] wrote {diagnostic_path}")
        return output_path, sidecar_path
    finally:
        camera_system.stop()


def main() -> int:
    args = build_parser().parse_args()
    run_table_calibration(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run the table calibration CLI and contract tests**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest tests.test_table_calibration_contract tests.test_cameras_calibrate_table_cli
```

Expected: PASS.

- [ ] **Step 4: Run CLI help to verify the entrypoint imports without hardware startup**

Run:

```bash
conda run -n demo_2_max --no-capture-output python cameras_calibrate_table.py --help
```

Expected: exits 0 and prints `Strict one-shot single-camera table Z=0 calibration`.

- [ ] **Step 5: Commit the CLI implementation**

```bash
git add cameras_calibrate_table.py qqtt/env/camera/table_calibration.py tests/test_cameras_calibrate_table_cli.py tests/test_table_calibration_contract.py
git commit -m "feat: add single-camera table calibration cli"
```

## Task 5: Raw Recording Table Calibration Copy

**Files:**
- Modify: `record_data.py`
- Create: `tests/test_record_data_table_calibration.py`

- [ ] **Step 1: Write failing tests for raw recording table calibration copy helpers**

Create `tests/test_record_data_table_calibration.py`:

```python
from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from qqtt.env.camera.table_calibration import build_table_calibration_metadata, write_table_calibration_files
from record_data import copy_table_calibration_into_case


def _write_sample_table_calibration(path: Path) -> None:
    metadata = build_table_calibration_metadata(
        serial_numbers=["cam0"],
        WH=[1280, 720],
        fps=5,
        transform_count=1,
        calibration_board={"name": "calibio-12x9-30mm"},
        max_reprojection_error_px=0.20,
        min_corner_fraction=0.60,
        min_charuco_corners=52,
        per_camera_reprojection_error=[0.10],
        per_camera_corner_count=[60],
        per_camera_corner_fraction=[0.68],
    )
    write_table_calibration_files(path, [np.eye(4, dtype=np.float32)], metadata)


class RecordDataTableCalibrationTest(unittest.TestCase):
    def test_copy_table_calibration_into_case_updates_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table_path = root / "source_table.pkl"
            case_dir = root / "case"
            case_dir.mkdir()
            (case_dir / "metadata.json").write_text(
                json.dumps({"schema_version": "qqtt_recording_v2", "serial_numbers": ["cam0"]}),
                encoding="utf-8",
            )
            _write_sample_table_calibration(table_path)

            copy_table_calibration_into_case(
                table_calibrate_path=table_path,
                output_path=case_dir,
                serial_numbers=["cam0"],
            )

            self.assertTrue((case_dir / "table_calibrate.pkl").is_file())
            self.assertTrue((case_dir / "table_calibrate_metadata.json").is_file())
            metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["table_calibration_path"], "table_calibrate.pkl")
            self.assertEqual(metadata["table_calibration_metadata_path"], "table_calibrate_metadata.json")
            self.assertEqual(metadata["table_world_frame_kind"], "table_world_z0")
            self.assertEqual(metadata["table_calibration_reference_serials"], ["cam0"])

    def test_copy_table_calibration_rejects_missing_serial(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table_path = root / "source_table.pkl"
            case_dir = root / "case"
            case_dir.mkdir()
            (case_dir / "metadata.json").write_text(json.dumps({"serial_numbers": ["other"]}), encoding="utf-8")
            _write_sample_table_calibration(table_path)

            with self.assertRaisesRegex(Exception, "does not cover serials"):
                copy_table_calibration_into_case(
                    table_calibrate_path=table_path,
                    output_path=case_dir,
                    serial_numbers=["other"],
                )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the recording tests and verify they fail on missing helper**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest tests.test_record_data_table_calibration
```

Expected: FAIL with an import error for `copy_table_calibration_into_case`.

- [ ] **Step 3: Implement recording copy support**

Modify `record_data.py`:

```python
import json
```

Add imports:

```python
from qqtt.env.camera.table_calibration import (
    TABLE_WORLD_FRAME_KIND,
    load_table_calibration_metadata,
    load_table_calibration_transforms,
    table_calibration_metadata_path_for,
)
```

Add parser option after `--calibrate_path`:

```python
    parser.add_argument(
        "--table-calibrate",
        type=str,
        default=None,
        help="Optional table Z=0 calibration file to copy into the recorded case.",
    )
```

Add helper:

```python
def _update_case_metadata(case_metadata_path: Path, updates: dict[str, object]) -> None:
    metadata = json.loads(case_metadata_path.read_text(encoding="utf-8"))
    metadata.update(updates)
    case_metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def copy_table_calibration_into_case(
    *,
    table_calibrate_path: Path,
    output_path: Path,
    serial_numbers: list[str],
) -> None:
    table_path = Path(table_calibrate_path).resolve()
    load_table_calibration_transforms(table_path, serial_numbers=list(serial_numbers))
    table_metadata = load_table_calibration_metadata(table_path)
    output_table_path = output_path / "table_calibrate.pkl"
    output_sidecar_path = output_path / "table_calibrate_metadata.json"
    copy2(table_path, output_table_path)
    sidecar_path = table_calibration_metadata_path_for(table_path)
    copy2(sidecar_path, output_sidecar_path)
    _update_case_metadata(
        output_path / "metadata.json",
        {
            "table_calibration_path": output_table_path.name,
            "table_calibration_metadata_path": output_sidecar_path.name,
            "table_world_frame_kind": TABLE_WORLD_FRAME_KIND,
            "table_calibration_reference_serials": list(table_metadata["table_calibration_reference_serials"]),
        },
    )
```

After normal `calibrate.pkl` copy in `main()`, add:

```python
    if args.table_calibrate is not None:
        copy_table_calibration_into_case(
            table_calibrate_path=Path(args.table_calibrate),
            output_path=output_path,
            serial_numbers=list(effective_serials),
        )
```

- [ ] **Step 4: Run recording table calibration tests**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest tests.test_record_data_table_calibration
```

Expected: PASS.

- [ ] **Step 5: Commit recording support**

```bash
git add record_data.py tests/test_record_data_table_calibration.py
git commit -m "feat: copy table calibration into recordings"
```

## Task 6: Aligned Case Table Calibration Preservation

**Files:**
- Modify: `data_process/record_data_align.py`
- Create: `tests/test_record_data_align_table_calibration.py`

- [ ] **Step 1: Write failing tests for aligned table calibration preservation**

Create `tests/test_record_data_align_table_calibration.py`:

```python
from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from data_process.record_data_align import write_aligned_table_calibration_file
from qqtt.env.camera.table_calibration import (
    build_table_calibration_metadata,
    load_table_calibration_transforms,
    write_table_calibration_files,
)


class RecordDataAlignTableCalibrationTest(unittest.TestCase):
    def test_write_aligned_table_calibration_file_reorders_to_case_serials(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "raw"
            output_dir = root / "aligned"
            case_dir.mkdir()
            output_dir.mkdir()
            first = np.eye(4, dtype=np.float32)
            first[:3, 3] = [1.0, 0.0, 0.0]
            second = np.eye(4, dtype=np.float32)
            second[:3, 3] = [2.0, 0.0, 0.0]
            metadata = build_table_calibration_metadata(
                serial_numbers=["a", "b"],
                WH=[1280, 720],
                fps=5,
                transform_count=2,
                calibration_board={"name": "calibio-12x9-30mm"},
                max_reprojection_error_px=0.20,
                min_corner_fraction=0.60,
                min_charuco_corners=52,
                per_camera_reprojection_error=[0.10, 0.11],
                per_camera_corner_count=[60, 61],
                per_camera_corner_fraction=[0.68, 0.69],
            )
            write_table_calibration_files(case_dir / "table_calibrate.pkl", [first, second], metadata)

            write_aligned_table_calibration_file(
                case_dir=case_dir,
                output_case_dir=output_dir,
                metadata={"serial_numbers": ["b"], "table_calibration_reference_serials": ["a", "b"]},
            )

            loaded = load_table_calibration_transforms(
                output_dir / "table_calibrate.pkl",
                serial_numbers=["b"],
                table_calibration_reference_serials=["b"],
            )
            np.testing.assert_allclose(loaded[0], second)
            aligned_metadata = json.loads((output_dir / "table_calibrate_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(aligned_metadata["serial_numbers"], ["b"])
            self.assertEqual(aligned_metadata["table_calibration_reference_serials"], ["b"])

    def test_write_aligned_table_calibration_file_is_noop_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "raw"
            output_dir = root / "aligned"
            case_dir.mkdir()
            output_dir.mkdir()

            self.assertFalse(
                write_aligned_table_calibration_file(
                    case_dir=case_dir,
                    output_case_dir=output_dir,
                    metadata={"serial_numbers": ["cam0"]},
                )
            )
            self.assertFalse((output_dir / "table_calibrate.pkl").exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the aligned table tests and verify they fail on missing helper**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest tests.test_record_data_align_table_calibration
```

Expected: FAIL with an import error for `write_aligned_table_calibration_file`.

- [ ] **Step 3: Implement aligned table calibration copy**

Modify imports in `data_process/record_data_align.py`:

```python
from qqtt.env.camera.table_calibration import (
    TABLE_WORLD_FRAME_KIND,
    load_table_calibration_metadata,
    load_table_calibration_transforms,
    table_calibration_metadata_path_for,
    write_table_calibration_files,
)
```

Add helper near `write_aligned_calibration_file`:

```python
def write_aligned_table_calibration_file(*, case_dir: Path, output_case_dir: Path, metadata: dict[str, Any]) -> bool:
    table_path = case_dir / "table_calibrate.pkl"
    if not table_path.is_file():
        return False
    source_metadata = load_table_calibration_metadata(table_path)
    transforms = load_table_calibration_transforms(
        table_path,
        serial_numbers=list(metadata["serial_numbers"]),
        table_calibration_reference_serials=metadata.get(
            "table_calibration_reference_serials",
            source_metadata.get("table_calibration_reference_serials", source_metadata["serial_numbers"]),
        ),
    )
    aligned_metadata = dict(source_metadata)
    aligned_metadata["serial_numbers"] = list(metadata["serial_numbers"])
    aligned_metadata["table_calibration_reference_serials"] = list(metadata["serial_numbers"])
    aligned_metadata["logical_camera_names"] = metadata.get(
        "logical_camera_names",
        [f"cam{i}" for i in range(len(metadata["serial_numbers"]))],
    )
    aligned_metadata["transform_count"] = len(transforms)
    write_table_calibration_files(output_case_dir / "table_calibrate.pkl", transforms, aligned_metadata)
    return True
```

Call it after `write_aligned_calibration_file(...)`:

```python
    has_table_calibration = write_aligned_table_calibration_file(
        case_dir=case_dir,
        output_case_dir=output_case_dir,
        metadata=metadata,
    )
```

Add fields to `aligned_metadata`:

```python
        "table_calibration_path": "table_calibrate.pkl" if has_table_calibration else None,
        "table_calibration_metadata_path": "table_calibrate_metadata.json" if has_table_calibration else None,
        "table_world_frame_kind": TABLE_WORLD_FRAME_KIND if has_table_calibration else None,
```

- [ ] **Step 4: Run aligned table calibration tests**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest tests.test_record_data_align_table_calibration
```

Expected: PASS.

- [ ] **Step 5: Commit alignment support**

```bash
git add data_process/record_data_align.py tests/test_record_data_align_table_calibration.py
git commit -m "feat: preserve table calibration in aligned cases"
```

## Task 7: Demo 3.x Explicit Table Calibration Contract

**Files:**
- Modify: `qqtt/demo/single_demo_v3_runtime.py`
- Modify: `tests/test_single_demo_v3_runtime.py`

- [ ] **Step 1: Add failing runtime tests**

Append these tests to the relevant test class in `tests/test_single_demo_v3_runtime.py`:

```python
    def test_demo32_contract_includes_table_calibration_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table_path = root / "table_calibrate.pkl"
            metadata = build_table_calibration_metadata(
                serial_numbers=["cam0"],
                WH=[1280, 720],
                fps=5,
                transform_count=1,
                calibration_board={"name": "calibio-12x9-30mm"},
                max_reprojection_error_px=0.20,
                min_corner_fraction=0.60,
                min_charuco_corners=52,
                per_camera_reprojection_error=[0.10],
                per_camera_corner_count=[60],
                per_camera_corner_fraction=[0.68],
            )
            write_table_calibration_files(table_path, [np.eye(4, dtype=np.float32)], metadata)
            parser = runtime.build_arg_parser(demo_version=runtime.DEMO_VERSION_3_2)
            args = parser.parse_args([
                "--dry-run",
                "--input-source",
                "fake-live",
                "--table-calibrate",
                str(table_path),
            ])
            args = runtime.apply_preset_defaults(args, explicit_options={"--dry-run", "--input-source", "--table-calibrate"})
            runtime.validate_args(args)
            contract = runtime.build_contract(args)
            self.assertEqual(contract["table_world_frame_kind"], "table_world_z0")
            self.assertEqual(contract["table_calibration_path"], str(table_path))

    def test_demo32_rejects_missing_table_calibration_path(self) -> None:
        parser = runtime.build_arg_parser(demo_version=runtime.DEMO_VERSION_3_2)
        args = parser.parse_args([
            "--dry-run",
            "--input-source",
            "fake-live",
            "--table-calibrate",
            "missing_table_calibrate.pkl",
        ])
        args = runtime.apply_preset_defaults(args, explicit_options={"--dry-run", "--input-source", "--table-calibrate"})
        with self.assertRaisesRegex(ValueError, "Missing table calibration file"):
            runtime.validate_args(args)
```

Add imports if missing:

```python
import tempfile
from pathlib import Path
import numpy as np
from qqtt.env.camera.table_calibration import build_table_calibration_metadata, write_table_calibration_files
```

- [ ] **Step 2: Run the runtime tests and verify they fail**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest tests.test_single_demo_v3_runtime
```

Expected: FAIL because `--table-calibrate` is unknown.

- [ ] **Step 3: Implement runtime contract support**

Modify `qqtt/demo/single_demo_v3_runtime.py`.

Add import:

```python
from qqtt.env.camera.table_calibration import TABLE_WORLD_FRAME_KIND, load_table_calibration_transforms
```

Add parser argument near recording inputs:

```python
    parser.add_argument(
        "--table-calibrate",
        type=Path,
        default=None,
        help="Optional table Z=0 calibration file to validate and expose in the demo contract.",
    )
```

In `validate_args`, after replay validation:

```python
    if args.table_calibrate is not None:
        table_path = Path(args.table_calibrate).expanduser()
        if not table_path.is_absolute():
            table_path = REPO_ROOT / table_path
        load_table_calibration_transforms(table_path)
        args.table_calibrate = table_path
```

In `build_contract`, add:

```python
        "table_calibration_path": None if args.table_calibrate is None else str(args.table_calibrate),
        "table_world_frame_kind": None if args.table_calibrate is None else TABLE_WORLD_FRAME_KIND,
```

In `build_live_delegate_argv`, do not forward `--table-calibrate` until the shared delegate has table-world rendering support. This contract-level load validates the artifact and makes the choice visible without changing current render coordinates.

- [ ] **Step 4: Run runtime tests**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest tests.test_single_demo_v3_runtime
```

Expected: PASS.

- [ ] **Step 5: Commit demo contract support**

```bash
git add qqtt/demo/single_demo_v3_runtime.py tests/test_single_demo_v3_runtime.py
git commit -m "feat: expose table calibration in demo contract"
```

## Task 8: Documentation Updates

**Files:**
- Modify: `docs/WORKFLOWS.md`
- Modify: `docs/HARDWARE_VALIDATION.md`
- Modify: `docs/ARCHITECTURE.md`

- [ ] **Step 1: Update workflow docs**

In `docs/WORKFLOWS.md`, after section `## 2. Calibrate`, add:

````markdown
## 2a. Table Z0 Calibration

Place the current lab ChArUco board flat on the table surface, with the printed
board plane touching the tabletop. Then run:

```bash
conda run -n demo_2_max --no-capture-output python cameras_calibrate_table.py
```

This writes `table_calibrate.pkl`, `table_calibrate_metadata.json`, and
`table_calibrate_diagnostic.png` in the repo root when the strict one-shot
check passes. The table calibration is separate from `calibrate.pkl`; it is used
only by commands that explicitly pass `--table-calibrate table_calibrate.pkl`.

Rerun table calibration after moving the camera, moving the table, changing the
camera mount, or changing the tabletop surface used as `Z=0`.
````

- [ ] **Step 2: Update hardware validation docs**

In `docs/HARDWARE_VALIDATION.md`, after `## Calibration Checklist`, add:

```markdown
## Table Z0 Calibration Checklist

- exactly one D455 is connected, or `--serial` selects the intended camera
- the ChArUco board is flat on the tabletop that should define `Z=0`
- `conda run -n demo_2_max --no-capture-output python cameras_calibrate_table.py` exits 0
- `table_calibrate.pkl` exists
- `table_calibrate_metadata.json` exists and reports `world_frame_kind = table_world_z0`
- `table_calibrate_diagnostic.png` shows the board corners and pose axes on the board
- if the strict corner count or reprojection check fails, adjust lighting/board visibility and rerun
```

- [ ] **Step 3: Update architecture docs**

In `docs/ARCHITECTURE.md`, in the Camera Identity Contract section after the `calibrate_metadata.json` paragraph, add:

```markdown
`table_calibrate.pkl` is a separate optional single-camera table-world artifact.
It uses the same list-of-4x4 `camera_to_world_c2w` physical shape as
`calibrate.pkl`, but its metadata declares `world_frame_kind = table_world_z0`
and compatibility contract `qqtt_table_calibrate_c2w_v1`. It is never used by
default; recording, alignment, and demo commands must receive it explicitly via
`--table-calibrate`.
```

- [ ] **Step 4: Run docs grep checks**

Run:

```bash
rg -n "table_calibrate|table_world_z0|cameras_calibrate_table" docs/WORKFLOWS.md docs/HARDWARE_VALIDATION.md docs/ARCHITECTURE.md
```

Expected: each file has at least one match.

- [ ] **Step 5: Commit docs**

```bash
git add docs/WORKFLOWS.md docs/HARDWARE_VALIDATION.md docs/ARCHITECTURE.md
git commit -m "docs: document table z0 calibration"
```

## Task 9: Final Validation

**Files:**
- No source edits unless validation reveals a defect.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest \
  tests.test_table_calibration_contract \
  tests.test_cameras_calibrate_table_cli \
  tests.test_record_data_table_calibration \
  tests.test_record_data_align_table_calibration \
  tests.test_single_demo_v3_runtime
```

Expected: PASS.

- [ ] **Step 2: Run entrypoint help commands**

Run:

```bash
conda run -n demo_2_max --no-capture-output python cameras_calibrate_table.py --help
conda run -n demo_2_max --no-capture-output python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py --dry-run --input-source fake-live --table-calibrate table_calibrate.pkl
```

Expected:

- the first command prints CLI help and exits 0
- the second command exits 0 only when `table_calibrate.pkl` already exists; if the file is not present, it should fail clearly with `Missing table calibration file`

- [ ] **Step 3: Run smoke validation**

Run:

```bash
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke
```

Expected: PASS.

- [ ] **Step 4: Run whitespace validation**

Run:

```bash
git diff --check
```

Expected: no output and exit 0.

- [ ] **Step 5: Review final diff**

Run:

```bash
git status --short --untracked-files=all
git log --oneline -8
```

Expected:

- only intended table calibration changes remain
- unrelated pre-existing dirty files are either untouched or explicitly excluded from commits

- [ ] **Step 6: Push after user-approved implementation**

After implementation is complete and all validation passes, push the branch:

```bash
git push origin single-camera
```

Expected: remote `single-camera` receives the implementation commits.
