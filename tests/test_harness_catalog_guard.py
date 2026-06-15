from __future__ import annotations

import unittest
import shutil
import sys

sys.dont_write_bytecode = True

from scripts.harness.guards import check_harness_catalog


class HarnessCatalogGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        for path in check_harness_catalog.HARNESS_ROOT.rglob("__pycache__"):
            shutil.rmtree(path)

    def test_current_catalog_has_no_violations(self) -> None:
        self.assertEqual(check_harness_catalog.collect_violations(), [])

    def test_no_harness_pycache_directories_are_committed(self) -> None:
        runtime_cache = check_harness_catalog.HARNESS_ROOT / "__pycache__"
        runtime_cache.mkdir()

        violations = check_harness_catalog.collect_violations()

        self.assertFalse(
            any("harness cache" in violation for violation in violations),
            violations,
        )

    def test_root_has_no_public_python_or_shell_entrypoints(self) -> None:
        allowed = {
            check_harness_catalog.HARNESS_ROOT / "__init__.py",
            check_harness_catalog.HARNESS_ROOT / "_catalog.py",
        }
        root_public = {
            path
            for path in check_harness_catalog.HARNESS_ROOT.iterdir()
            if path.is_file() and path.suffix in {".py", ".sh"}
        }
        self.assertEqual(root_public, allowed)


if __name__ == "__main__":
    unittest.main()
