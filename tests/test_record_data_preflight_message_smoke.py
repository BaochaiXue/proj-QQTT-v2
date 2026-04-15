from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import qqtt.env
import record_data
from qqtt.env.camera.preflight import CapturePreflightDecision


def _decision(
    *,
    capture_mode: str,
    operator_status: str,
    allowed_to_record: bool,
    serials: list[str] | None,
    reason: str,
) -> CapturePreflightDecision:
    serial_list = [] if serials is None else list(serials)
    return CapturePreflightDecision(
        capture_mode=capture_mode,
        serials=serial_list,
        width=848,
        height=480,
        fps=30,
        emitter="on",
        topology_type=None if serials is None else ("three_camera" if len(serial_list) == 3 else "single"),
        stream_set={"both_eval": "rgbd_ir_pair", "stereo_ir": "rgb_ir_pair"}.get(capture_mode),
        probe_support=None if operator_status in {"pending_serial_resolution", "unknown"} else allowed_to_record,
        policy_label="test_policy",
        unsupported_behavior="warn" if capture_mode in {"both_eval", "stereo_ir"} else "allow",
        operator_status=operator_status,
        allowed_to_record=allowed_to_record,
        requires_probe=True,
        reason=reason,
        probe_results_json="C:/probe.json",
        probe_results_md="C:/probe.md",
    )


class _FakeRealsense:
    def __init__(self) -> None:
        self.stop_called = False

    def stop(self) -> None:
        self.stop_called = True


class _FakeCameraSystem:
    last_instance = None

    def __init__(self, *args, serial_numbers=None, **kwargs) -> None:
        type(self).last_instance = self
        self.serial_numbers = list(serial_numbers) if serial_numbers is not None else ["a", "b", "c"]
        self.realsense = _FakeRealsense()
        self.record_calls: list[tuple[str, int | None]] = []

    def record(self, output_path, max_frames=None) -> None:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        self.record_calls.append((str(output_path), max_frames))


class RecordDataPreflightMessageSmokeTest(unittest.TestCase):
    def test_explicit_serials_allow_both_eval_with_warning(self) -> None:
        _FakeCameraSystem.last_instance = None
        with tempfile.TemporaryDirectory() as tmp_dir:
            argv = [
                "record_data.py",
                "--case_name",
                "warning_case",
                "--output_dir",
                tmp_dir,
                "--capture_mode",
                "both_eval",
                "--emitter",
                "on",
                "--serials",
                "a",
                "b",
                "c",
                "--disable-keyboard-listener",
            ]
            stdout = io.StringIO()
            with patch.object(qqtt.env, "CameraSystem", _FakeCameraSystem), patch.object(
                record_data,
                "evaluate_capture_preflight",
                side_effect=[
                    _decision(
                        capture_mode="both_eval",
                        operator_status="experimental_warning",
                        allowed_to_record=True,
                        serials=["a", "b", "c"],
                        reason="warning before startup",
                    ),
                    _decision(
                        capture_mode="both_eval",
                        operator_status="experimental_warning",
                        allowed_to_record=True,
                        serials=["a", "b", "c"],
                        reason="warning after discovery",
                    ),
                ],
            ), patch("sys.argv", argv), redirect_stdout(stdout):
                self.assertEqual(record_data.main(), 0)
            self.assertIsNotNone(_FakeCameraSystem.last_instance)
            self.assertEqual(len(_FakeCameraSystem.last_instance.record_calls), 1)
            output = stdout.getvalue()
            self.assertIn("stage: before camera startup", output)
            self.assertIn("stage: after camera discovery", output)
            self.assertIn("operator_status: experimental_warning", output)
            self.assertIn("recording will still be attempted", output)

    def test_discovered_serials_allow_both_eval_with_warning_after_camera_discovery(self) -> None:
        _FakeCameraSystem.last_instance = None
        with tempfile.TemporaryDirectory() as tmp_dir:
            argv = [
                "record_data.py",
                "--case_name",
                "warning_after_discovery",
                "--output_dir",
                tmp_dir,
                "--capture_mode",
                "both_eval",
                "--emitter",
                "on",
                "--disable-keyboard-listener",
            ]
            stdout = io.StringIO()
            with patch.object(qqtt.env, "CameraSystem", _FakeCameraSystem), patch.object(
                record_data,
                "evaluate_capture_preflight",
                side_effect=[
                    _decision(
                        capture_mode="both_eval",
                        operator_status="pending_serial_resolution",
                        allowed_to_record=True,
                        serials=None,
                        reason="pending serial discovery",
                    ),
                    _decision(
                        capture_mode="both_eval",
                        operator_status="experimental_warning",
                        allowed_to_record=True,
                        serials=["a", "b", "c"],
                        reason="warning after discovery",
                    ),
                ],
            ), patch("sys.argv", argv), redirect_stdout(stdout):
                self.assertEqual(record_data.main(), 0)
            self.assertIsNotNone(_FakeCameraSystem.last_instance)
            self.assertFalse(_FakeCameraSystem.last_instance.realsense.stop_called)
            self.assertEqual(len(_FakeCameraSystem.last_instance.record_calls), 1)
            output = stdout.getvalue()
            self.assertIn("stage: before camera discovery", output)
            self.assertIn("stage: after camera discovery", output)
            self.assertIn("operator_status: experimental_warning", output)
            self.assertIn("recording will still be attempted", output)

    def test_discovered_serials_warn_for_stereo_ir(self) -> None:
        _FakeCameraSystem.last_instance = None
        with tempfile.TemporaryDirectory() as tmp_dir:
            argv = [
                "record_data.py",
                "--case_name",
                "warn_case",
                "--output_dir",
                tmp_dir,
                "--capture_mode",
                "stereo_ir",
                "--emitter",
                "on",
                "--disable-keyboard-listener",
            ]
            stdout = io.StringIO()
            with patch.object(qqtt.env, "CameraSystem", _FakeCameraSystem), patch.object(
                record_data,
                "evaluate_capture_preflight",
                side_effect=[
                    _decision(
                        capture_mode="stereo_ir",
                        operator_status="pending_serial_resolution",
                        allowed_to_record=True,
                        serials=None,
                        reason="pending serial discovery",
                    ),
                    _decision(
                        capture_mode="stereo_ir",
                        operator_status="experimental_warning",
                        allowed_to_record=True,
                        serials=["a", "b", "c"],
                        reason="warning after discovery",
                    ),
                ],
            ), patch("sys.argv", argv), redirect_stdout(stdout):
                self.assertEqual(record_data.main(), 0)
            self.assertEqual(len(_FakeCameraSystem.last_instance.record_calls), 1)
            output = stdout.getvalue()
            self.assertIn("operator_status: experimental_warning", output)
            self.assertIn("recording will still be attempted", output)

    def test_unknown_support_state_message_is_printed(self) -> None:
        _FakeCameraSystem.last_instance = None
        with tempfile.TemporaryDirectory() as tmp_dir:
            argv = [
                "record_data.py",
                "--case_name",
                "unknown_case",
                "--output_dir",
                tmp_dir,
                "--capture_mode",
                "stereo_ir",
                "--emitter",
                "on",
                "--disable-keyboard-listener",
            ]
            stdout = io.StringIO()
            with patch.object(qqtt.env, "CameraSystem", _FakeCameraSystem), patch.object(
                record_data,
                "evaluate_capture_preflight",
                side_effect=[
                    _decision(
                        capture_mode="stereo_ir",
                        operator_status="pending_serial_resolution",
                        allowed_to_record=True,
                        serials=None,
                        reason="pending serial discovery",
                    ),
                    _decision(
                        capture_mode="stereo_ir",
                        operator_status="unknown",
                        allowed_to_record=True,
                        serials=["a", "b", "c"],
                        reason="probe file missing in test",
                    ),
                ],
            ), patch("sys.argv", argv), redirect_stdout(stdout):
                self.assertEqual(record_data.main(), 0)
            output = stdout.getvalue()
            self.assertIn("operator_status: unknown", output)
            self.assertIn("preflight support is unknown", output)


if __name__ == "__main__":
    unittest.main()
