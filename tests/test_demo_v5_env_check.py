from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = REPO_ROOT / "demo_v5" / "env" / "check_demo_v5_env.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("demo_v5_env_checker", CHECKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load checker module from {CHECKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DemoV5EnvCheckTest(unittest.TestCase):
    def test_nvcc_toolchain_reports_cuda_home_without_nvcc(self) -> None:
        checker = _load_checker()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cuda_home = tmp_path / "cuda-home"
            cuda_home.mkdir()
            path_dir = tmp_path / "path-bin"
            path_dir.mkdir()
            path_nvcc = path_dir / "nvcc"
            path_nvcc.write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
            path_nvcc.chmod(0o755)

            with mock.patch.dict(
                os.environ,
                {
                    "CUDA_HOME": str(cuda_home),
                    "PATH": str(path_dir),
                },
                clear=True,
            ):
                errors = checker._check_nvcc_toolchain()

        self.assertTrue(
            any("CUDA_HOME/bin/nvcc" in error for error in errors),
            errors,
        )

    def test_shape_prior_require_cuda_runs_gsplat_runtime_smoke(self) -> None:
        checker = _load_checker()

        with (
            mock.patch.object(checker, "_check_imports", return_value=[]),
            mock.patch.object(checker, "_check_paths", return_value=[]),
            mock.patch.object(checker, "_check_shape_prior_source_import", return_value=[]),
            mock.patch.object(checker, "_check_cuda", return_value=[]),
            mock.patch.object(checker, "_check_nvcc_toolchain", return_value=[]),
            mock.patch.object(checker, "_check_gsplat_runtime_smoke", return_value=[]) as smoke,
        ):
            result = checker.main(["--role", "shape-prior", "--require-cuda"])

        self.assertEqual(result, 0)
        smoke.assert_called_once_with()

    def test_shape_prior_without_require_cuda_skips_gsplat_runtime_smoke(self) -> None:
        checker = _load_checker()

        with (
            mock.patch.object(checker, "_check_imports", return_value=[]),
            mock.patch.object(checker, "_check_paths", return_value=[]),
            mock.patch.object(checker, "_check_shape_prior_source_import", return_value=[]),
            mock.patch.object(checker, "_check_cuda", return_value=[]),
            mock.patch.object(checker, "_check_nvcc_toolchain", return_value=[]),
            mock.patch.object(checker, "_check_gsplat_runtime_smoke", return_value=[]) as smoke,
        ):
            result = checker.main(["--role", "shape-prior"])

        self.assertEqual(result, 0)
        smoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
