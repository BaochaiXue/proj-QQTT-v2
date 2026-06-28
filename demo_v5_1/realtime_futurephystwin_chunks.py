"""Legacy compatibility wrapper for :mod:`demo_v5_1.realtime_data_process_sam3d`."""
from __future__ import annotations

from demo_v5_1.realtime_data_process_sam3d import *  # noqa: F401,F403
from demo_v5_1.realtime_data_process_sam3d import main


if __name__ == "__main__":
    raise SystemExit(main())
