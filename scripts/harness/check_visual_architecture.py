from __future__ import annotations

import ast
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
VIS_ROOT = ROOT / "data_process" / "visualization"
SCRIPTS_ROOT = ROOT / "scripts" / "harness"


LOW_LEVEL_MODULES = {
    "data_process.visualization.calibration_io",
    "data_process.visualization.camera_frusta",
    "data_process.visualization.depth_colormap",
    "data_process.visualization.io_artifacts",
    "data_process.visualization.io_case",
    "data_process.visualization.layouts",
    "data_process.visualization.object_compare",
    "data_process.visualization.object_roi",
    "data_process.visualization.renderers.__init__",
    "data_process.visualization.renderers.fallback",
    "data_process.visualization.roi",
    "data_process.visualization.source_compare",
    "data_process.visualization.support_compare",
    "data_process.visualization.types",
    "data_process.visualization.views",
}


FILE_LENGTH_LIMITS = {
    ROOT / "data_process" / "visualization" / "pointcloud_compare.py": 700,
    ROOT / "data_process" / "visualization" / "turntable_compare.py": 2450,
    ROOT / "scripts" / "harness" / "visual_compare_depth_panels.py": 220,
    ROOT / "scripts" / "harness" / "visual_compare_reprojection.py": 220,
    ROOT / "scripts" / "harness" / "visual_compare_depth_video.py": 220,
    ROOT / "scripts" / "harness" / "visual_compare_turntable.py": 260,
    ROOT / "scripts" / "harness" / "visual_make_professor_triptych.py": 260,
    ROOT / "scripts" / "harness" / "visual_make_match_board.py": 260,
    ROOT / "scripts" / "harness" / "visual_compare_stereo_order_pcd.py": 260,
    ROOT / "scripts" / "harness" / "audit_ffs_left_right.py": 220,
    ROOT / "scripts" / "harness" / "compare_face_smoothness.py": 220,
}


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
            for alias in node.names:
                imports.append(alias.name)
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


def collect_violations() -> list[str]:
    violations: list[str] = []
    visual_paths = list(VIS_ROOT.rglob("*.py"))
    for path in visual_paths:
        module_name = _module_name_for_path(path)
        imports = _imported_modules(path)
        if any(name.startswith("scripts.harness") for name in imports):
            violations.append(f"{module_name} imports scripts.harness, which violates layering.")
        if module_name in LOW_LEVEL_MODULES and any(name.startswith("data_process.visualization.workflows") for name in imports):
            violations.append(f"{module_name} imports workflow modules, which violates low-level layering.")
        if module_name.startswith("data_process.visualization.renderers") and any(name == "argparse" or name.startswith("scripts.harness") for name in imports):
            violations.append(f"{module_name} imports CLI-only modules.")

    for path, max_lines in FILE_LENGTH_LIMITS.items():
        if not path.exists():
            violations.append(f"Missing expected path for architecture check: {path}")
            continue
        with path.open("r", encoding="utf-8") as handle:
            line_count = sum(1 for _ in handle)
        if line_count > max_lines:
            violations.append(f"{path.relative_to(ROOT)} is {line_count} lines, exceeds limit {max_lines}.")
    return violations


def main() -> int:
    violations = collect_violations()
    if violations:
        for item in violations:
            print(f"[visual-arch] {item}")
        return 1
    print("[visual-arch] architecture checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
