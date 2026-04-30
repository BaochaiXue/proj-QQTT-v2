from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.harness._catalog import CATALOG


HARNESS_ROOT = ROOT / "scripts" / "harness"
PRIVATE_PYTHON_FILES = {
    HARNESS_ROOT / "__init__.py",
    HARNESS_ROOT / "_catalog.py",
    HARNESS_ROOT / "experiments" / "__init__.py",
}
KNOWN_CATEGORIES = {
    "checks",
    "hardware_external",
    "mask_support",
    "formal_cleanup",
    "current_compare",
    "experiments",
    "focused_diagnostics",
}


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def collect_violations() -> list[str]:
    violations: list[str] = []
    entry_paths = [ROOT / entry.path for entry in CATALOG]
    unique_paths = set(entry_paths)

    if len(unique_paths) != len(entry_paths):
        seen: set[Path] = set()
        for path in entry_paths:
            if path in seen:
                violations.append(f"Duplicate catalog entry: {path.relative_to(ROOT)}")
            seen.add(path)

    for entry in CATALOG:
        path = ROOT / entry.path
        if entry.category not in KNOWN_CATEGORIES:
            violations.append(f"Unknown category for {entry.path}: {entry.category}")
        if not path.exists():
            violations.append(f"Catalog path does not exist: {entry.path}")
        if entry.help_profile is not None and path.suffix != ".py":
            violations.append(f"Non-Python path cannot have help coverage: {entry.path}")
        is_experiment_path = _is_under(path, HARNESS_ROOT / "experiments")
        if entry.category == "experiments" and not is_experiment_path:
            violations.append(f"Experiment entry is outside experiments/: {entry.path}")
        if is_experiment_path and entry.category != "experiments":
            violations.append(f"Experiment path has non-experiment category: {entry.path}")

    cataloged_python = {path for path in unique_paths if path.suffix == ".py"}
    for path in sorted(HARNESS_ROOT.rglob("*.py")):
        if path in PRIVATE_PYTHON_FILES:
            continue
        if path not in cataloged_python:
            violations.append(f"Uncataloged harness Python file: {path.relative_to(ROOT)}")

    return violations


def main() -> int:
    violations = collect_violations()
    if violations:
        for item in violations:
            print(f"[harness-catalog] {item}")
        return 1
    print("[harness-catalog] catalog checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
