from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "qqtt").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"failed to locate repo root from {start}")


ROOT = _find_repo_root(Path(__file__).resolve())
ROOT_STR = str(ROOT)
if ROOT_STR in sys.path:
    sys.path.remove(ROOT_STR)
sys.path.insert(0, ROOT_STR)

from scripts.harness._catalog import help_scripts


BASE_FORMAL_HELP_SCRIPTS: tuple[str, ...] = (
    "cameras_viewer.py",
    "cameras_calibrate.py",
    "record_data.py",
    "record_data_realtime_align.py",
    "data_process/record_data_align.py",
)

FULL_ONLY_FORMAL_HELP_SCRIPTS: tuple[str, ...] = ("cameras_viewer_FFS.py",)

DEMO_HELP_SCRIPTS: tuple[str, ...] = (
    "demo_v5_1/main.py",
    "demo_v5_1/main_data_processing.py",
    "demo_v5_1/visualize_track.py",
    "demo_v5_1/env/check_demo_v5_env.py",
)


def _unique(items: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return tuple(unique)


QUICK_UNITTEST_MODULES: tuple[str, ...] = (
    "tests.test_demo_v5_legacy_key_cleanup",
    "tests.test_demo_v5_1_default_config",
    "tests.test_demo_v5_1_shape_prior_simplification",
    "tests.test_demo_v5_1_chunk_data",
    "tests.test_demo_v5_1_tools_io",
    "tests.test_demo_v5_1_tracking",
    "tests.test_demo_v5_1_visualize_track",
    "tests.test_demo_v6_asap",
    "tests.test_realsense_extrinsics_matrix",
    "tests.test_single_view_shape_align",
    "tests.test_validation_smoke_manifest",
)

FULL_ONLY_UNITTEST_MODULES: tuple[str, ...] = (
    "tests.test_data_process_origin_sam3d_pipeline",
)

SMOKE_UNITTEST_MODULES: tuple[str, ...] = QUICK_UNITTEST_MODULES
DETERMINISTIC_ONLY_UNITTEST_MODULES: tuple[str, ...] = FULL_ONLY_UNITTEST_MODULES
EXHAUSTIVE_ONLY_UNITTEST_MODULES: tuple[str, ...] = ()

CHECK_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("scripts/harness/guards/check_harness_catalog.py",),
    ("scripts/harness/guards/check_demo_v5_no_compat_wrappers.py",),
    ("scripts/harness/guards/check_experiment_boundaries.py",),
    ("scripts/harness/guards/check_visual_architecture.py",),
    ("-m", "scripts.harness.guards.check_scope"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run repo validation profiles. Default is smoke; hardware profile lists "
            "manual commands unless --run-hardware is passed."
        )
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "deterministic", "hardware", "exhaustive"),
        default="smoke",
        help="Validation profile to run. Defaults to smoke.",
    )
    parser.add_argument(
        "--run-hardware",
        action="store_true",
        help="Run hardware profile commands instead of listing them for manual execution.",
    )
    return parser.parse_args(argv)


def run(cmd: list[str]) -> None:
    print(f"[validation] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def _help_commands(*, python: str, scripts: tuple[str, ...]) -> list[list[str]]:
    return [[python, script, "--help"] for script in scripts]


def _check_commands(*, python: str) -> list[list[str]]:
    commands: list[list[str]] = []
    for command in CHECK_COMMANDS:
        if command[0] == "-m":
            commands.append([python, *command])
        else:
            commands.append([python, command[0]])
    return commands


def _unittest_commands(
    *, python: str, module_batches: tuple[tuple[str, ...], ...]
) -> list[list[str]]:
    return [[python, "-m", "unittest", "-v", *modules] for modules in module_batches]


def _profile_unittest_batches(profile: str) -> tuple[tuple[str, ...], ...]:
    if profile == "smoke":
        return (SMOKE_UNITTEST_MODULES,)
    if profile == "deterministic":
        modules = _unique(
            (*SMOKE_UNITTEST_MODULES, *DETERMINISTIC_ONLY_UNITTEST_MODULES)
        )
        return tuple((module,) for module in modules)
    if profile == "exhaustive":
        modules = _unique(
            (
                *SMOKE_UNITTEST_MODULES,
                *DETERMINISTIC_ONLY_UNITTEST_MODULES,
                *EXHAUSTIVE_ONLY_UNITTEST_MODULES,
            )
        )
        return tuple((module,) for module in modules)
    if profile == "hardware":
        return ()
    raise ValueError(f"Unsupported profile: {profile}")


def _formal_scripts_for_profile(profile: str) -> tuple[str, ...]:
    if profile == "smoke":
        return BASE_FORMAL_HELP_SCRIPTS
    if profile in {"deterministic", "exhaustive"}:
        return _unique(
            (
                *BASE_FORMAL_HELP_SCRIPTS,
                *FULL_ONLY_FORMAL_HELP_SCRIPTS,
                *DEMO_HELP_SCRIPTS,
            )
        )
    if profile == "hardware":
        return ()
    raise ValueError(f"Unsupported profile: {profile}")


def _catalog_help_commands(
    *, python: str, profile: str, include_manual: bool = False
) -> list[list[str]]:
    return [
        [python, script, "--help"]
        for script in help_scripts(profile, include_manual=include_manual)
    ]


def build_commands(
    *, python: str, profile: str, run_hardware: bool = False
) -> list[list[str]]:
    if profile == "hardware" and not run_hardware:
        return []
    if profile == "hardware":
        return _catalog_help_commands(
            python=python, profile=profile, include_manual=True
        )
    if profile in {"smoke", "deterministic", "exhaustive"}:
        commands = [
            *_help_commands(
                python=python, scripts=_formal_scripts_for_profile(profile)
            ),
            *_catalog_help_commands(python=python, profile=profile),
            *_check_commands(python=python),
            *_unittest_commands(
                python=python, module_batches=_profile_unittest_batches(profile)
            ),
        ]
        return commands
    raise ValueError(f"Unsupported profile: {profile}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    python = sys.executable
    print(f"[validation] profile={args.profile}")
    if args.profile == "hardware" and not args.run_hardware:
        print(
            "[validation] hardware profile is manual; pass --run-hardware to run these commands:"
        )
        for cmd in _catalog_help_commands(
            python=python, profile=args.profile, include_manual=True
        ):
            print(f"[validation] {' '.join(cmd)}")
        return 0
    for cmd in build_commands(
        python=python, profile=args.profile, run_hardware=args.run_hardware
    ):
        run(cmd)
    print(f"[validation] {args.profile} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
