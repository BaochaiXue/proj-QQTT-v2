from __future__ import annotations

import unittest
from pathlib import Path

from scripts.harness.validation import run as validation_run


def _module_file(module: str) -> Path:
    return validation_run.ROOT.joinpath(*module.split(".")).with_suffix(".py")


def _top_level_test_module(path: Path) -> str:
    return ".".join(path.relative_to(validation_run.ROOT).with_suffix("").parts)


def _profile_modules() -> set[str]:
    return {
        module
        for profile in ("smoke", "deterministic", "exhaustive")
        for batch in validation_run._profile_unittest_batches(profile)
        for module in batch
    }


class ValidationSmokeManifestTests(unittest.TestCase):
    def test_unittest_modules_exist_in_current_checkout(self) -> None:
        missing = [
            module
            for profile in ("smoke", "deterministic", "exhaustive")
            for batch in validation_run._profile_unittest_batches(profile)
            for module in batch
            if not _module_file(module).is_file()
        ]

        self.assertEqual([], missing)

    def test_top_level_tests_are_listed_in_validation_profiles(self) -> None:
        configured_modules = _profile_modules()
        missing = [
            _top_level_test_module(path)
            for path in sorted((validation_run.ROOT / "tests").glob("test*.py"))
            if _top_level_test_module(path) not in configured_modules
        ]

        self.assertEqual([], missing)

    def test_formal_help_scripts_exist_in_current_checkout(self) -> None:
        missing = [
            script
            for profile in ("smoke", "deterministic", "exhaustive")
            for script in validation_run._formal_scripts_for_profile(profile)
            if not (validation_run.ROOT / script).is_file()
        ]

        self.assertEqual([], missing)


if __name__ == "__main__":
    unittest.main()
