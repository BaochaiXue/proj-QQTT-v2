#!/usr/bin/env python3
"""Demo v6.1 main data processing runtime."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_repo_root() -> Path:
    """Resolve repo root."""
    candidates: list[Path] = []
    candidates.extend([Path(__file__).resolve().parents[1], Path.cwd()])
    env_root = os.environ.get("QQTT_REPO_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (
            (root / "data_process").is_dir()
            and (root / "demo_v6_2").is_dir()
            and (root / "qqtt").is_dir()
        ):
            return root
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _resolve_repo_root()
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR in sys.path:
    sys.path.remove(REPO_ROOT_STR)
sys.path.insert(0, REPO_ROOT_STR)


from demo_v6_2.mdp_constants import *  # noqa: E402,F401,F403
from demo_v6_2.mdp_packets import *  # noqa: E402,F401,F403
from demo_v6_2.mdp_capture_source import *  # noqa: E402,F401,F403
from demo_v6_2.mdp_headless_writer import *  # noqa: E402,F401,F403
from demo_v6_2.mdp_pipeline_plumbing import *  # noqa: E402,F401,F403
from demo_v6_2.mdp_cli import *  # noqa: E402,F401,F403
from demo_v6_2.mdp_pcd_depth import *  # noqa: E402,F401,F403
from demo_v6_2.mdp_segmentation import *  # noqa: E402,F401,F403

from demo_v6_2.mdp_demo_lifecycle import _LifecycleMixin  # noqa: E402
from demo_v6_2.mdp_demo_capture import _CaptureMixin  # noqa: E402
from demo_v6_2.mdp_demo_segwarmup import _SegWarmupMixin  # noqa: E402
from demo_v6_2.mdp_demo_tracker import _TrackerMixin  # noqa: E402
from demo_v6_2.mdp_demo_pcd import _PcdMixin  # noqa: E402
from demo_v6_2.mdp_demo_pairpublish import _PairPublishMixin  # noqa: E402


class MainDataProcessingDemo(
    _LifecycleMixin,
    _CaptureMixin,
    _SegWarmupMixin,
    _TrackerMixin,
    _PcdMixin,
    _PairPublishMixin,
):
    """Camera -> segmentation -> tracker/pcd -> filter -> pairing -> headless capture."""


def main(argv: list[str] | None = None) -> int:
    """Run the command-line entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        apply_demo_preset(args)
        validate_args(args)
        return MainDataProcessingDemo(args).run()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        # Startup errors (camera/device selection, arg validation) never reach
        # the worker-thread fatal hook, so surface them on the live status band
        # too (design question 23). Best-effort: a None capture dir is a no-op.
        from demo_v6_2.pipeline_status import STAGE_FATAL, PipelineStatusWriter  # noqa: PLC0415

        capture_dir = args.headless_capture_dir
        PipelineStatusWriter(
            Path(capture_dir).parent if capture_dir is not None else None,
            "camera",
        ).emit(STAGE_FATAL, f"startup: {exc}", ok=False, exc_type=type(exc).__name__)
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
