from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

from qqtt.tracking.base import BackendAvailability, BackendUnavailableError, TrackingResult


DEFAULT_EXTERNAL_ROOT = Path("/home/zhangxinjie/external_tracking_backends")


class ExternalRepoProbeBackend:
    name = "external_probe"
    repo_dir_name = ""
    module_names: tuple[str, ...] = ()
    checkpoint_env_vars: tuple[str, ...] = ()
    install_hint = ""
    integration_ready = False

    def __init__(self, *, external_root: str | Path | None = None, device: str = "cuda") -> None:
        self.external_root = Path(external_root or os.environ.get("DEMO3_TRACKING_EXTERNAL_ROOT", DEFAULT_EXTERNAL_ROOT))
        self.device = str(device)

    def _repo_path(self) -> Path:
        return self.external_root / self.repo_dir_name

    def _module_available(self) -> tuple[bool, str]:
        last = "no module probe configured"
        for module_name in self.module_names:
            try:
                __import__(module_name)
                return True, f"module {module_name} importable"
            except Exception as exc:
                last = f"module {module_name} import failed: {exc}"
        return False, last

    def _checkpoint_available(self) -> tuple[bool, str]:
        if not self.checkpoint_env_vars:
            return False, "no checkpoint environment variable configured"
        checked: list[str] = []
        for env_var in self.checkpoint_env_vars:
            raw_path = os.environ.get(env_var, "")
            if not raw_path:
                checked.append(f"{env_var}=<unset>")
                continue
            path = Path(raw_path).expanduser()
            if path.exists():
                return True, f"{env_var}={path}"
            checked.append(f"{env_var}={path} (missing)")
        return False, "checkpoint not configured; " + ", ".join(checked)

    def availability(self) -> BackendAvailability:
        module_ok, module_reason = self._module_available()
        checkpoint_ok, checkpoint_reason = self._checkpoint_available()
        if module_ok and checkpoint_ok and self.integration_ready:
            return BackendAvailability(self.name, True, f"{module_reason}; {checkpoint_reason}")
        if module_ok:
            return BackendAvailability(
                self.name,
                False,
                f"{module_reason}; {checkpoint_reason}; PyTorch runtime wrapper is not implemented in this dependency-gated probe",
            )
        if self._repo_path().exists():
            return BackendAvailability(self.name, False, f"{self._repo_path()} exists but runtime module/checkpoint is not configured; {module_reason}")
        return BackendAvailability(self.name, False, f"{self.repo_dir_name} repo/runtime not found. {self.install_hint}")

    def is_available(self) -> bool:
        return self.availability().available

    def availability_reason(self) -> str:
        return self.availability().reason

    def initialize(self, frames: Sequence[np.ndarray], query_points_yx: np.ndarray, masks: Sequence[np.ndarray] | None = None) -> None:
        _ = frames, query_points_yx, masks
        raise BackendUnavailableError(self.availability_reason())

    def track_sequence(
        self,
        frames: Sequence[np.ndarray] | None = None,
        query_points_yx: np.ndarray | None = None,
        *,
        frames_rgb: Sequence[np.ndarray] | None = None,
        camera_idx: int | None = None,
        output_shape_hw: tuple[int, int] | None = None,
    ) -> TrackingResult:
        _ = frames, query_points_yx, frames_rgb, camera_idx, output_shape_hw
        raise BackendUnavailableError(
            f"{self.name} PyTorch integration is a dependency-gated probe in this slice: {self.availability_reason()}"
        )

    def update(self, frame: np.ndarray) -> TrackingResult:
        _ = frame
        raise BackendUnavailableError(self.availability_reason())


class LocoTrackBackend(ExternalRepoProbeBackend):
    name = "locotrack"
    repo_dir_name = "locotrack"
    module_names = ("locotrack",)
    checkpoint_env_vars = ("DEMO3_LOCOTRACK_CHECKPOINT", "LOCOTRACK_CHECKPOINT")
    install_hint = "Clone https://github.com/cvlab-kaist/locotrack and configure weights."


class TapNextBackend(ExternalRepoProbeBackend):
    name = "tapnext"
    repo_dir_name = "tapnet"
    module_names = ("tapnet",)
    checkpoint_env_vars = ("DEMO3_TAPNEXT_CHECKPOINT", "TAPNEXT_CHECKPOINT")
    install_hint = "Clone https://github.com/google-deepmind/tapnet and configure TAPNext/TAPNext++ checkpoints."


class TapirBackend(ExternalRepoProbeBackend):
    name = "tapir"
    repo_dir_name = "tapnet"
    module_names = ("tapnet",)
    checkpoint_env_vars = ("DEMO3_TAPIR_CHECKPOINT", "TAPIR_CHECKPOINT")
    install_hint = "Clone https://github.com/google-deepmind/tapnet and configure TAPIR/BootsTAPIR checkpoints."
