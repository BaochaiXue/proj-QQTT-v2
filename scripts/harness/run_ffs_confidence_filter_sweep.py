from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.harness.experiments.run_ffs_confidence_filter_sweep import main


if __name__ == "__main__":
    raise SystemExit(main())
