from __future__ import annotations

from pathlib import Path

from qqtt.tracking.backends.point_tracker_adapter import (
    TRACKER_BACKEND_TRACKON2,
    UnavailableExternalPointTrackerAdapter,
    tracker_backend_spec,
)


class TrackOn2Adapter(UnavailableExternalPointTrackerAdapter):
    """Track-On2 adapter shell for Demo 3.1 external-backend routing.

    The repository and checkpoints are intentionally external to QQTT. This
    adapter makes the Demo 3.1 child-process route, CLI, and profiling contract
    ready without importing Track-On2 at module-import time.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        camera_idx: int | None = None,
        checkpoint: str | None = None,
        config_path: str | None = None,
        repo_dir: str | None = None,
    ) -> None:
        missing: list[str] = []
        if not checkpoint:
            missing.append("--trackon2-checkpoint")
        elif not Path(checkpoint).expanduser().exists():
            missing.append(f"--trackon2-checkpoint {checkpoint!r} does not exist")
        if config_path and not Path(config_path).expanduser().exists():
            missing.append(f"--trackon2-config {config_path!r} does not exist")
        if repo_dir and not Path(repo_dir).expanduser().exists():
            missing.append(f"--trackon2-repo-dir {repo_dir!r} does not exist")
        detail = "; ".join(missing) if missing else "Track-On2 Python runtime wrapper is not installed yet"
        reason = (
            "Track-On2 backend is configured as an external tracker. "
            f"{detail}. Install/checkout the Track-On2 repo in the demo_3_1_max env "
            "and wire its model builder into TrackOn2Adapter before running live tracking."
        )
        super().__init__(
            spec=tracker_backend_spec(TRACKER_BACKEND_TRACKON2),
            device=device,
            camera_idx=camera_idx,
            reason=reason,
        )
        self.checkpoint = checkpoint
        self.config_path = config_path
        self.repo_dir = repo_dir


__all__ = ["TrackOn2Adapter"]
