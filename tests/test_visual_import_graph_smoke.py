from __future__ import annotations

import unittest

from scripts.harness.check_visual_architecture import collect_violations


class VisualImportGraphSmokeTest(unittest.TestCase):
    def test_visual_architecture_has_no_layering_violations(self) -> None:
        self.assertEqual(collect_violations(), [])
