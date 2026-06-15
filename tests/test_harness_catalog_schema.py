from __future__ import annotations

import unittest

from scripts.harness import _catalog


class HarnessCatalogSchemaTest(unittest.TestCase):
    def test_validation_profiles_are_new_names(self) -> None:
        self.assertEqual(
            _catalog.VALIDATION_PROFILES,
            ("smoke", "deterministic", "hardware", "exhaustive"),
        )

    def test_lifecycles_are_new_directory_names(self) -> None:
        self.assertEqual(
            _catalog.LIFECYCLES,
            ("guards", "validation", "diagnostics", "benchmarks", "experiments", "support"),
        )

    def test_entries_expose_validation_metadata(self) -> None:
        entry = _catalog.HarnessEntry(
            path="scripts/harness/guards/check_scope.py",
            lifecycle="guards",
            domain="scope",
            summary="Repo scope guard.",
            validation_profile="smoke",
            help=False,
            automatic=True,
            requires=(),
        )
        self.assertEqual(entry.path, "scripts/harness/guards/check_scope.py")
        self.assertEqual(entry.lifecycle, "guards")
        self.assertEqual(entry.domain, "scope")
        self.assertEqual(entry.summary, "Repo scope guard.")
        self.assertEqual(entry.validation_profile, "smoke")
        self.assertFalse(entry.help)
        self.assertTrue(entry.automatic)
        self.assertEqual(entry.requires, ())

    def test_help_scripts_uses_validation_profiles(self) -> None:
        smoke_scripts = _catalog.help_scripts("smoke")
        deterministic_scripts = _catalog.help_scripts("deterministic")
        self.assertIsInstance(smoke_scripts, tuple)
        self.assertIsInstance(deterministic_scripts, tuple)
        self.assertIn("scripts/harness/visual_compare_depth_panels.py", smoke_scripts)
        self.assertIn("scripts/harness/visual_compare_depth_triplet_ply.py", deterministic_scripts)
        self.assertTrue(set(smoke_scripts).issubset(set(deterministic_scripts)))

    def test_entries_by_lifecycle_replaces_entries_by_category(self) -> None:
        grouped = _catalog.entries_by_lifecycle()
        self.assertIn("guards", grouped)
        self.assertIn("validation", grouped)
        self.assertIn("scripts/harness/check_scope.py", [entry.path for entry in grouped["guards"]])
        self.assertIn("scripts/harness/validation/run.py", [entry.path for entry in grouped["validation"]])
        self.assertNotIn("checks", grouped)
        self.assertFalse(hasattr(_catalog, "entries_by_category"))


if __name__ == "__main__":
    unittest.main()
