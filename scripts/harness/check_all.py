from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str]) -> None:
    print(f"[check] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> int:
    python = sys.executable
    run([python, "cameras_viewer.py", "--help"])
    run([python, "cameras_calibrate.py", "--help"])
    run([python, "record_data.py", "--help"])
    run([python, "data_process/record_data_align.py", "--help"])
    run([python, "-m", "scripts.harness.check_scope"])
    run([python, "-m", "unittest", "-v", "tests.test_record_data_align_smoke"])
    print("[check] all deterministic checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
