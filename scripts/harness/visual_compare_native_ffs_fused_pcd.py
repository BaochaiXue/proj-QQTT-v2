from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.harness.experiments.visual_compare_native_ffs_fused_pcd import main


if __name__ == "__main__":
    raise SystemExit(main())
