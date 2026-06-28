from __future__ import annotations

from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "qqtt").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"failed to locate repo root from {start}")


ROOT = _find_repo_root(Path(__file__).resolve())
DEMO_ROOTS = (ROOT / "demo_v5", ROOT / "demo_v5_1")
FORBIDDEN_MARKER = "Legacy compatibility wrapper"
FORBIDDEN_SOURCE_TOKENS = {
    Path("demo_v5_1/realtime_dense_track.py"): (
        "from qqtt.demo import realtime_masked_edgetam_pcd",
        "import qqtt.demo.realtime_masked_edgetam_pcd",
        "from demo_v5 import realtime_dense_track",
        "import demo_v5.realtime_dense_track",
        "masked_pcd.main",
        "thin wrapper",
    ),
}


def collect_violations() -> list[str]:
    violations: list[str] = []
    for root in DEMO_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.py")):
            relative_path = path.relative_to(ROOT)
            source = path.read_text(encoding="utf-8")
            if FORBIDDEN_MARKER in source:
                violations.append(
                    f"{relative_path} still declares a legacy compatibility wrapper."
                )
            for token in FORBIDDEN_SOURCE_TOKENS.get(relative_path, ()):
                if token in source:
                    violations.append(
                        f"{relative_path} still contains wrapper token: {token}"
                    )
    return violations


def main() -> int:
    violations = collect_violations()
    if violations:
        for item in violations:
            print(f"[demo-v5-compat] {item}")
        return 1
    print("[demo-v5-compat] no Demo v5 compatibility wrappers found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
