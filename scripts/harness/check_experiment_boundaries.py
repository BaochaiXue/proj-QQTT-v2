from __future__ import annotations

import ast
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
VIS_ROOT = ROOT / "data_process" / "visualization"
HARNESS_ROOT = ROOT / "scripts" / "harness"

EXPERIMENT_WORKFLOW_SHIMS = {
    ROOT / "data_process" / "visualization" / "workflows" / "ffs_confidence_filter_pcd_compare.py",
    ROOT / "data_process" / "visualization" / "workflows" / "ffs_confidence_panels.py",
    ROOT / "data_process" / "visualization" / "workflows" / "ffs_confidence_pcd_panels.py",
    ROOT / "data_process" / "visualization" / "workflows" / "ffs_confidence_threshold_sweep_pcd_compare.py",
    ROOT / "data_process" / "visualization" / "workflows" / "ffs_mask_erode_sweep_pcd_compare.py",
    ROOT / "data_process" / "visualization" / "workflows" / "native_ffs_fused_pcd_compare.py",
}

EXPERIMENT_CLI_SHIMS = {
    ROOT / "scripts" / "harness" / "run_ffs_confidence_filter_sweep.py",
    ROOT / "scripts" / "harness" / "visual_compare_ffs_confidence_filter_pcd.py",
    ROOT / "scripts" / "harness" / "visual_compare_ffs_confidence_threshold_sweep_pcd.py",
    ROOT / "scripts" / "harness" / "visual_compare_ffs_mask_erode_sweep_pcd.py",
    ROOT / "scripts" / "harness" / "visual_compare_native_ffs_fused_pcd.py",
    ROOT / "scripts" / "harness" / "visualize_ffs_static_confidence_panels.py",
    ROOT / "scripts" / "harness" / "visualize_ffs_static_confidence_pcd_panels.py",
}

FORMAL_CODE_ROOTS = (
    ROOT / "qqtt",
    ROOT / "data_process" / "depth_backends",
)

FORMAL_CODE_FILES = {
    ROOT / "cameras_viewer.py",
    ROOT / "cameras_viewer_FFS.py",
    ROOT / "cameras_calibrate.py",
    ROOT / "record_data.py",
    ROOT / "record_data_realtime_align.py",
    ROOT / "data_process" / "record_data_align.py",
    ROOT / "data_process" / "aligned_case_metadata.py",
}

EXPERIMENT_IMPORT_PREFIXES = (
    "data_process.visualization.experiments",
    "scripts.harness.experiments",
)


def _module_name_for_path(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    return ".".join(rel.parts)


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    module_name = _module_name_for_path(path)
    module_parts = module_name.split(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    imports.append(node.module)
                continue
            base_parts = module_parts[:-node.level]
            if node.module:
                target = ".".join(base_parts + node.module.split("."))
            else:
                target = ".".join(base_parts)
            if target:
                imports.append(target)
    return imports


def _imports_experiment(imports: list[str]) -> bool:
    return any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in imports
        for prefix in EXPERIMENT_IMPORT_PREFIXES
    )


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _formal_paths() -> list[Path]:
    paths = [path for path in FORMAL_CODE_FILES if path.exists()]
    for root in FORMAL_CODE_ROOTS:
        if root.exists():
            paths.extend(root.rglob("*.py"))
    return sorted(set(paths))


def collect_violations() -> list[str]:
    violations: list[str] = []

    for path in _formal_paths():
        imports = _imported_modules(path)
        if _imports_experiment(imports):
            violations.append(f"{path.relative_to(ROOT)} imports experiment-only code.")

    for path in sorted(VIS_ROOT.rglob("*.py")):
        if _is_under(path, VIS_ROOT / "experiments") or path in EXPERIMENT_WORKFLOW_SHIMS:
            continue
        imports = _imported_modules(path)
        if any(
            module == "data_process.visualization.experiments"
            or module.startswith("data_process.visualization.experiments.")
            for module in imports
        ):
            violations.append(f"{path.relative_to(ROOT)} imports visualization experiments.")

    for path in sorted(HARNESS_ROOT.rglob("*.py")):
        if _is_under(path, HARNESS_ROOT / "experiments") or path in EXPERIMENT_CLI_SHIMS:
            continue
        imports = _imported_modules(path)
        if any(module == "scripts.harness.experiments" or module.startswith("scripts.harness.experiments.") for module in imports):
            violations.append(f"{path.relative_to(ROOT)} imports harness experiments.")

    for path in sorted(EXPERIMENT_WORKFLOW_SHIMS | EXPERIMENT_CLI_SHIMS):
        if not path.exists():
            violations.append(f"Missing experiment compatibility shim: {path.relative_to(ROOT)}")
            continue
        nonblank_lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        max_nonblank_lines = 12 if path in EXPERIMENT_CLI_SHIMS else 7
        if len(nonblank_lines) > max_nonblank_lines:
            violations.append(f"Experiment compatibility shim is too large: {path.relative_to(ROOT)}")

    return violations


def main() -> int:
    violations = collect_violations()
    if violations:
        for item in violations:
            print(f"[experiment-boundary] {item}")
        return 1
    print("[experiment-boundary] experiment boundary checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
