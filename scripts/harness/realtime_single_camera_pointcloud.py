from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_v2 import realtime_single_camera_pointcloud as _impl


globals().update({name: getattr(_impl, name) for name in dir(_impl) if not name.startswith("__")})
main = _impl.main


if __name__ == "__main__":
    raise SystemExit(main())
