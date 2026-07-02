from __future__ import annotations

import unittest
from pathlib import Path

from scripts.harness.validation import run as validation_run


def _module_file(module: str) -> Path:
    return validation_run.ROOT.joinpath(*module.split(".")).with_suffix(".py")


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

    def test_formal_help_scripts_exist_in_current_checkout(self) -> None:
        missing = [
            script
            for profile in ("smoke", "deterministic", "exhaustive")
            for script in validation_run._formal_scripts_for_profile(profile)
            if not (validation_run.ROOT / script).is_file()
        ]

        self.assertEqual([], missing)

    def test_pytest_batch_files_exist_in_current_checkout(self) -> None:
        missing = [
            test_path
            for batch in validation_run.PYTEST_BATCHES
            for test_path in batch
            if not (validation_run.ROOT / test_path).is_file()
        ]

        self.assertEqual([], missing)


if __name__ == "__main__":
    unittest.main()
