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


def collect_violations() -> list[str]:
    violations: list[str] = []
    for root in DEMO_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.py")):
            if FORBIDDEN_MARKER in path.read_text(encoding="utf-8"):
                violations.append(f"{path.relative_to(ROOT)} still declares a legacy compatibility wrapper.")
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
