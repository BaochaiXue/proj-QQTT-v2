from __future__ import annotations

from pathlib import Path
import sys


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "qqtt").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"failed to locate repo root from {start}")


ROOT = _find_repo_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qqtt.demo import realtime_single_camera_pointcloud as _impl


globals().update({name: getattr(_impl, name) for name in dir(_impl) if not name.startswith("__")})
main = _impl.main


if __name__ == "__main__":
    raise SystemExit(main())
