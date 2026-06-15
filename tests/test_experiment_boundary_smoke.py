from __future__ import annotations

import unittest

from scripts.harness.guards import check_experiment_boundaries


class ExperimentBoundarySmokeTest(unittest.TestCase):
    def test_experiment_boundary_has_no_violations(self) -> None:
        self.assertEqual(check_experiment_boundaries.collect_violations(), [])

    def test_formal_entrypoints_are_guarded(self) -> None:
        formal_paths = {path.name for path in check_experiment_boundaries._formal_paths()}
        self.assertIn("record_data_align.py", formal_paths)
        self.assertIn("aligned_case_metadata.py", formal_paths)

    def test_shared_demo_runtime_is_not_formal_code(self) -> None:
        formal_paths = {path.as_posix() for path in check_experiment_boundaries._formal_paths()}

        self.assertFalse(any("/qqtt/demo/" in path for path in formal_paths))


if __name__ == "__main__":
    unittest.main()
