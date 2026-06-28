from __future__ import annotations

import unittest
from pathlib import Path

from scripts.harness.validation import run as validation_run


def _module_file(module: str) -> Path:
    return validation_run.ROOT.joinpath(*module.split(".")).with_suffix(".py")


class ValidationSmokeManifestTests(unittest.TestCase):
    def test_smoke_unittest_modules_exist_in_current_checkout(self) -> None:
        missing = [
            module
            for module in validation_run.SMOKE_UNITTEST_MODULES
            if not _module_file(module).is_file()
        ]

        self.assertEqual([], missing)


if __name__ == "__main__":
    unittest.main()
