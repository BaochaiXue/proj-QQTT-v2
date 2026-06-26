"""Legacy compatibility wrapper for :mod:`demo_v5.visualize_track`."""
from __future__ import annotations

from demo_v5.visualize_track import *  # noqa: F401,F403
from demo_v5.visualize_track import main


if __name__ == "__main__":
    raise SystemExit(main())
