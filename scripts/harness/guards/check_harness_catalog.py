from __future__ import annotations

from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "qqtt").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"failed to locate repo root from {start}")


ROOT = _find_repo_root(Path(__file__).resolve())
ROOT_STR = str(ROOT)
if ROOT_STR in sys.path:
    sys.path.remove(ROOT_STR)
sys.path.insert(0, ROOT_STR)

from scripts.harness._catalog import CATALOG, LIFECYCLES, VALIDATION_PROFILES


HARNESS_ROOT = ROOT / "scripts" / "harness"
PRIVATE_PYTHON_FILES = {
    HARNESS_ROOT / "__init__.py",
    HARNESS_ROOT / "_catalog.py",
}
PACKAGE_INIT_FILES = {path for path in HARNESS_ROOT.rglob("__init__.py")}


def _tracked_harness_cache_artifacts() -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "scripts/harness/**/__pycache__/*",
            "scripts/harness/__pycache__/*",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


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

    for path in _tracked_harness_cache_artifacts():
        violations.append(f"Committed harness cache artifact: {path}")

    if len(unique_paths) != len(entry_paths):
        seen: set[Path] = set()
        for path in entry_paths:
            if path in seen:
                violations.append(f"Duplicate catalog entry: {path.relative_to(ROOT)}")
            seen.add(path)

    for entry in CATALOG:
        path = ROOT / entry.path
        if entry.lifecycle not in LIFECYCLES:
            violations.append(f"Unknown lifecycle for {entry.path}: {entry.lifecycle}")
        if entry.validation_profile is not None and entry.validation_profile not in VALIDATION_PROFILES:
            violations.append(f"Unknown validation profile for {entry.path}: {entry.validation_profile}")
        if not path.exists():
            violations.append(f"Catalog path does not exist: {entry.path}")
        if entry.help and path.suffix != ".py":
            violations.append(f"Non-Python path cannot have help coverage: {entry.path}")
        expected_lifecycle_root = HARNESS_ROOT / entry.lifecycle
        if entry.lifecycle in LIFECYCLES and not _is_under(path, expected_lifecycle_root):
            violations.append(f"Lifecycle path mismatch for {entry.path}: expected scripts/harness/{entry.lifecycle}/")
        is_experiment_path = _is_under(path, HARNESS_ROOT / "experiments")
        if entry.lifecycle == "experiments" and not is_experiment_path:
            violations.append(f"Experiment entry is outside experiments/: {entry.path}")
        if is_experiment_path and entry.lifecycle != "experiments":
            violations.append(f"Experiment path has non-experiment lifecycle: {entry.path}")
        if entry.validation_profile == "hardware" and entry.automatic:
            violations.append(f"Hardware validation entry must be manual: {entry.path}")

    cataloged_python = {path for path in unique_paths if path.suffix == ".py"}
    for path in sorted(HARNESS_ROOT.rglob("*.py")):
        if path in PRIVATE_PYTHON_FILES or path in PACKAGE_INIT_FILES:
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
