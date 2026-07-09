"""Demo v6.1 must not import the repo-level services/data_process packages.

The data_process depth-backend helpers (geometry, ffs_defaults) are vendored
under demo_v6_1/utils so demo_v6_1 can be published standalone. These tests
guard that decoupling (static import scan) and that the vendored modules are
functionally faithful to the originals.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

DEMO_ROOT = Path(__file__).resolve().parents[1] / "demo_v6_1"
# Repo-level packages demo_v6_1 must not depend on for standalone release.
FORBIDDEN_TOP_LEVEL = {"services", "data_process"}


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
    def test_no_services_or_data_process_imports(self) -> None:
        offenders: list[str] = []
        for path in sorted(DEMO_ROOT.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            hits = _imported_top_level_packages(tree) & FORBIDDEN_TOP_LEVEL
            if hits:
                rel = path.relative_to(DEMO_ROOT.parent)
                offenders.append(f"{rel}: {sorted(hits)}")
        self.assertEqual(
            offenders,
            [],
            "demo_v6_1 must not import services/data_process; offenders:\n"
            + "\n".join(offenders),
        )

    def test_flagged_modules_import_without_services_or_data_process(self) -> None:
        # The two modules the report flagged as unconditionally importing
        # services.ffs_remote must now import cleanly without dragging in the
        # repo-level packages. Run in a fresh interpreter so other tests that
        # legitimately import services/data_process do not pollute sys.modules.
        probe = (
            "import demo_v6_1.main_warmup, demo_v6_1.main_data_processing, sys; "
            "leaked = sorted(n for n in sys.modules "
            "if n.split('.')[0] in {'services', 'data_process'}); "
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
        from demo_v6_1.utils.depth_geometry import transform_points

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
        from demo_v6_1.utils import ffs_defaults

        self.assertEqual(ffs_defaults.REPO_ROOT.name, "single_proj_qqtt")
        self.assertEqual(ffs_defaults.DEFAULT_FFS_MAX_DISP, 192)
        self.assertEqual(ffs_defaults.DEFAULT_FFS_TRT_ENGINE_SIZE, (480, 864))


if __name__ == "__main__":
    unittest.main()
