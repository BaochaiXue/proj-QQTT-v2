#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.harness._catalog import CATALOG


FORBIDDEN_DEMO_IMPORT_PREFIXES = ("demo_v1", "demo_v2", "demo_v2_1", "demo_v2_1_5")
DEMO22_FILES = (
    ROOT / "demo_v2_2" / "realtime_three_view_async_filtered_fused_pcd.py",
    ROOT / "demo_v2_2" / "runtime.py",
    ROOT / "demo_v2_2" / "render_fastpath.py",
    ROOT / "qqtt" / "demo" / "demo22_runtime.py",
)
REQUIRED_HARNESS_ENTRY = "scripts/harness/benchmark_demo22_render_replay.py"


def _import_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        return ",".join(alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return node.module or ""
    return None


def collect_violations() -> list[str]:
    violations: list[str] = []
    for path in DEMO22_FILES:
        if not path.exists():
            violations.append(f"Missing Demo 2.2 boundary file: {path.relative_to(ROOT)}")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            import_name = _import_name(node)
            if import_name is None:
                continue
            for prefix in FORBIDDEN_DEMO_IMPORT_PREFIXES:
                if import_name == prefix or import_name.startswith(f"{prefix}."):
                    violations.append(
                        f"{path.relative_to(ROOT)} imports {import_name}; use qqtt.demo shared runtime instead"
                    )

    catalog_entries = {entry.path: entry for entry in CATALOG}
    entry = catalog_entries.get(REQUIRED_HARNESS_ENTRY)
    if entry is None:
        violations.append(f"Missing harness catalog entry: {REQUIRED_HARNESS_ENTRY}")
    elif entry.category != "focused_diagnostics":
        violations.append(f"{REQUIRED_HARNESS_ENTRY} must be focused_diagnostics, got {entry.category}")
    return violations


def main() -> int:
    violations = collect_violations()
    if violations:
        for item in violations:
            print(f"[demo22-boundary] {item}")
        return 1
    print("[demo22-boundary] Demo 2.2 dependency and harness boundary checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
