from __future__ import annotations

import unittest

from scripts.harness import check_experiment_boundaries


class ExperimentBoundarySmokeTest(unittest.TestCase):
    def test_experiment_boundary_has_no_violations(self) -> None:
        self.assertEqual(check_experiment_boundaries.collect_violations(), [])

    def test_formal_entrypoints_are_guarded(self) -> None:
        formal_paths = {path.name for path in check_experiment_boundaries._formal_paths()}
        self.assertIn("record_data_align.py", formal_paths)
        self.assertIn("aligned_case_metadata.py", formal_paths)


if __name__ == "__main__":
    unittest.main()
