"""Demo v6.2 must be self-contained: no services/data_process, and no demo_v6_1.

demo_v6_2 is the refactor target; demo_v6_1 is the frozen known-good reference.
The refactor rewired every demo_v6_1 import/launch to demo_v6_2, so the standalone
guard here additionally forbids importing demo_v6_1 (a leak would mean the
self-containment rewire is incomplete). The vendored depth-backend helpers
(geometry, ffs_defaults) keep demo_v6_2 free of the repo-level data_process
package, mirroring the demo_v6_1 guard.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

DEMO_ROOT = Path(__file__).resolve().parents[1] / "demo_v6_2"
# Repo-level / sibling packages demo_v6_2 must not depend on for standalone
# release. demo_v6_1 is included so an incomplete self-containment rewire fails
# loudly. others/ holds experimental scripts and is excluded from the scan.
FORBIDDEN_TOP_LEVEL = {"services", "data_process", "demo_v6_1"}


def _imported_top_level_packages(tree: ast.AST) -> set[str]:
    packages: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                packages.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import (never services/data_process).
            if node.level == 0 and node.module:
                packages.add(node.module.split(".")[0])
    return packages


class StandaloneImportGuardTests(unittest.TestCase):
    def test_no_forbidden_imports(self) -> None:
        offenders: list[str] = []
        for path in sorted(DEMO_ROOT.rglob("*.py")):
            if "others" in path.relative_to(DEMO_ROOT).parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            hits = _imported_top_level_packages(tree) & FORBIDDEN_TOP_LEVEL
            if hits:
                rel = path.relative_to(DEMO_ROOT.parent)
                offenders.append(f"{rel}: {sorted(hits)}")
        self.assertEqual(
            offenders,
            [],
            "demo_v6_2 must not import services/data_process/demo_v6_1; offenders:\n"
            + "\n".join(offenders),
        )

    def test_core_modules_import_without_forbidden_packages(self) -> None:
        probe = (
            "import demo_v6_2.main_warmup, demo_v6_2.main_data_processing, "
            "demo_v6_2.chunk_data_stream, demo_v6_2.main, sys; "
            "leaked = sorted(n for n in sys.modules "
            "if n.split('.')[0] in {'services', 'data_process', 'demo_v6_1'}); "
            "assert not leaked, leaked; print('clean')"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(DEMO_ROOT.parent),
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"standalone import probe failed:\nstdout={result.stdout}\n"
            f"stderr={result.stderr[-2000:]}",
        )
        self.assertIn("clean", result.stdout)


class VendoredGeometryAndDefaultsTests(unittest.TestCase):
    def test_transform_points_matches_manual_math(self) -> None:
        from demo_v6_2.utils.depth_geometry import transform_points

        rng = np.random.default_rng(2)
        points = rng.random((7, 3)).astype(np.float32)
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        transform[:3, :3] = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        expected = (points @ transform[:3, :3].T) + transform[:3, 3]
        np.testing.assert_allclose(
            transform_points(points, transform), expected, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_array_equal(
            transform_points(np.empty((0, 3), dtype=np.float32), transform),
            np.empty((0, 3), dtype=np.float32),
        )

    def test_ffs_defaults_repo_root_resolves(self) -> None:
        from demo_v6_2.utils import ffs_defaults

        self.assertEqual(ffs_defaults.REPO_ROOT.name, "single_proj_qqtt")
        self.assertEqual(ffs_defaults.DEFAULT_FFS_MAX_DISP, 192)
        self.assertEqual(ffs_defaults.DEFAULT_FFS_TRT_ENGINE_SIZE, (480, 864))


if __name__ == "__main__":
    unittest.main()
