from __future__ import annotations

from pathlib import Path

from qqtt.tracking.backends.point_tracker_adapter import (
    TRACKER_BACKEND_LITETRACKER,
    UnavailableExternalPointTrackerAdapter,
    tracker_backend_spec,
)


class LiteTrackerAdapter(UnavailableExternalPointTrackerAdapter):
    """LiteTracker adapter shell for Demo 3.1 external-backend routing."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        camera_idx: int | None = None,
        weights: str | None = None,
        repo_dir: str | None = None,
    ) -> None:
        missing: list[str] = []
        if not weights:
            missing.append("--litetracker-weights")
        elif not Path(weights).expanduser().exists():
            missing.append(f"--litetracker-weights {weights!r} does not exist")
        if repo_dir and not Path(repo_dir).expanduser().exists():
            missing.append(f"--litetracker-repo-dir {repo_dir!r} does not exist")
        detail = "; ".join(missing) if missing else "LiteTracker Python runtime wrapper is not installed yet"
        reason = (
            "LiteTracker backend is configured as an external tracker. "
            f"{detail}. Install/checkout the LiteTracker repo in the demo_3_1_max env "
            "and wire its model builder into LiteTrackerAdapter before running live tracking."
        )
        super().__init__(
            spec=tracker_backend_spec(TRACKER_BACKEND_LITETRACKER),
            device=device,
            camera_idx=camera_idx,
            reason=reason,
        )
        self.weights = weights
        self.repo_dir = repo_dir


__all__ = ["LiteTrackerAdapter"]
