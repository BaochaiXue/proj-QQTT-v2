from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


REQUIRED_FILES = (
    "AGENTS.md",
    "docs/HARNESS_ENGINEERING.md",
    "scripts/harness/README.md",
    "scripts/harness/_catalog.py",
    "scripts/harness/check_all.py",
    "scripts/harness/check_scope.py",
    "scripts/harness/summarize_demo23_failure_packet.py",
)

REQUIRED_TEXT = {
    "AGENTS.md": (
        "scripts/harness/README.md",
        "docs/HARNESS_ENGINEERING.md",
        "Single-Camera Branch Policy",
        "git push origin single-camera",
    ),
    "docs/HARNESS_ENGINEERING.md": (
        "Demo 2.3 Failure Packet",
        "scripts/harness/summarize_demo23_failure_packet.py",
        "docs/generated/",
        "stuffed animal",
        "towel",
    ),
    "scripts/harness/README.md": (
        "docs/HARNESS_ENGINEERING.md",
        "summarize_demo23_failure_packet.py",
        "Single-Camera Branch Safety",
        "git push origin single-camera",
    ),
    "scripts/harness/_catalog.py": (
        "scripts/harness/check_harness_engineering.py",
        "scripts/harness/summarize_demo23_failure_packet.py",
    ),
    "scripts/harness/check_all.py": (
        "scripts/harness/check_harness_engineering.py",
        "tests.test_demo23_harness_engineering_smoke",
    ),
    "scripts/harness/check_scope.py": (
        "BRANCH_POLICY_REQUIRED_TEXT",
        "Single-Camera Branch Policy",
        "Single-Camera Branch Safety",
    ),
}


def _failures() -> list[str]:
    failures: list[str] = []
    for relpath in REQUIRED_FILES:
        path = ROOT / relpath
        if not path.is_file():
            failures.append(f"missing required harness engineering file: {relpath}")
    for relpath, needles in REQUIRED_TEXT.items():
        path = ROOT / relpath
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                failures.append(f"{relpath} is missing required text: {needle}")
    return failures


def main() -> int:
    failures = _failures()
    if failures:
        for failure in failures:
            print(f"[harness-engineering] FAIL {failure}", file=sys.stderr)
        return 1
    print("[harness-engineering] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
