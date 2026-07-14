#!/usr/bin/env python3
"""Demo v6.2 main data processing runtime (camera subprocess entry)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR in sys.path:
    sys.path.remove(REPO_ROOT_STR)
sys.path.insert(0, REPO_ROOT_STR)


from demo_v6_2.mdp.cli import build_parser, validate_and_normalize_args  # noqa: E402
from demo_v6_2.mdp.runtime import MainDataProcessingDemo  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Run the command-line entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_and_normalize_args(args)
        return MainDataProcessingDemo(args).run()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        # Startup errors (camera/device selection, arg validation) never reach
        # the worker-thread fatal hook, so surface them on the live status band
        # too (design question 25). Best-effort: a None capture dir is a no-op.
        from demo_v6_2.pipeline_status import STAGE_FATAL, PipelineStatusWriter  # noqa: PLC0415

        capture_dir = args.headless_capture_dir
        PipelineStatusWriter(
            Path(capture_dir).parent if capture_dir is not None else None,
            "camera",
        ).emit(STAGE_FATAL, f"startup: {exc}", ok=False, exc_type=type(exc).__name__)
        parser.exit(2, f"{parser.prog}: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
