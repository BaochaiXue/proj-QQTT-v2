"""Camera session state: source, runtime, calibration, and the capture sink.

Everything warm-up populates and ``stop()`` tears down lives here, so the
pipeline stages can hold one session reference instead of sharing mutable
attributes on a common ``self``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qqtt.env.camera.table_calibration import (
    TABLE_WORLD_FRAME_KIND,
    TableCalibrationLoadError,
    load_table_calibration_transforms,
)

from demo_v6_2.mdp.capture_source import (
    RecordedRgbdFrameSource,
    _start_realsense_pipeline,
)
from demo_v6_2.mdp.cli import RunMode
from demo_v6_2.mdp.constants import (
    DEFAULT_PROFILE,
    FAKE_LIVE_FRAME_SELECTION_POLICY,
)
from demo_v6_2.mdp.headless_writer import HeadlessCaptureWriter
from demo_v6_2.mdp.packets import RealtimeCameraRuntime
from demo_v6_2.utils.camera import parse_profile
from demo_v6_2.utils.ffs_align import FfsDepthEngine


class CameraSession:
    """Frame source, camera runtime, calibration, and headless capture sink."""

    def __init__(self) -> None:
        """Initialize CameraSession."""
        self.width, self.height = parse_profile(DEFAULT_PROFILE)
        self.camera_runtime: RealtimeCameraRuntime | None = None
        self.recording_source: RecordedRgbdFrameSource | None = None
        self.table_c2w: np.ndarray | None = None
        self.table_calibration_path: Path | None = None
        self.headless_capture_writer: HeadlessCaptureWriter | None = None
        # Built at run start when --depth-source ffs (TensorRT engines load
        # in the runner's constructor, so this cannot be an __init__ service).
        self.depth_engine: FfsDepthEngine | None = None

    def prepare_source(self, args: argparse.Namespace, mode: RunMode) -> None:
        """Open the frame source: fake-live recorded case or live RealSense."""
        if mode.fake_live_input:
            # The only replay source is the fake-live recorded case.
            self.recording_source = RecordedRgbdFrameSource(
                args.recording_case,
                replay_fps=float(args.replay_fps),
                depth_source=str(args.depth_source),
            )
            self.width = self.recording_source.width
            self.height = self.recording_source.height
            self.camera_runtime = self.recording_source.make_runtime()
            print(
                "[fake-live] "
                f"case={self.recording_source.case_path} "
                f"frames={self.recording_source.frame_count} "
                f"replay_fps={self.recording_source.effective_fps:g} "
                f"recording_fps={self.recording_source.recording_fps:g} "
                f"first_step={self.recording_source.steps[0]} "
                f"serial={self.recording_source.serial} "
                f"depth_source={self.recording_source.depth_source} "
                f"ir_stereo={str(self.recording_source.has_ir_stereo).lower()} "
                f"frame_selection={FAKE_LIVE_FRAME_SELECTION_POLICY}",
                flush=True,
            )
        else:
            self.camera_runtime = _start_realsense_pipeline(args)

    def initialize_table_calibration(self, args: argparse.Namespace) -> None:
        """Load and validate the camera-to-world table calibration."""
        if args.table_calibrate is None:
            raise RuntimeError(
                "formal runtime requires camera-to-world table calibration"
            )
        if self.camera_runtime is None:
            raise RuntimeError("camera runtime is not initialized")
        path = Path(args.table_calibrate)
        try:
            transforms = load_table_calibration_transforms(
                path, serial_numbers=[str(self.camera_runtime.serial)]
            )
        except TableCalibrationLoadError as exc:
            raise RuntimeError(
                f"Invalid table calibration for active camera "
                f"{self.camera_runtime.serial}: {exc}"
            ) from exc
        self.table_c2w = np.ascontiguousarray(transforms[0], dtype=np.float32)
        self.table_calibration_path = path
        print(
            "[table-calibrate] "
            f"path={path} serial={self.camera_runtime.serial} "
            f"pcd_coordinate_frame={TABLE_WORLD_FRAME_KIND}",
            flush=True,
        )

    def release_camera(self) -> None:
        """Stop the camera pipeline and drop the source/runtime references.

        The headless writer is cleared separately by the composition root's
        ``stop()`` — the incomplete-timeline check still needs it after the
        camera is gone.
        """
        if self.camera_runtime is not None and self.camera_runtime.pipeline is not None:
            try:
                self.camera_runtime.pipeline.stop()
            except Exception:
                pass
        self.camera_runtime = None
        self.recording_source = None


__all__ = ["CameraSession"]
