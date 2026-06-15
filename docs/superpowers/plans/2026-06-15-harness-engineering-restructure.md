# Harness Engineering Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `scripts/harness/` into a catalog-driven validation subsystem with lifecycle directories, explicit validation profiles, updated guards, and no old root-level public harness scripts.

**Architecture:** Keep `_catalog.py` as the machine-readable source of truth, move the validation runner to `scripts/harness/validation/run.py`, and make guards consume catalog metadata instead of legacy categories. File moves are mechanical `git mv` operations; behavior changes are isolated to catalog schema, validation profile expansion, guard checks, docs, and tests.

**Tech Stack:** Python standard library, `unittest`, `pytest` for existing D455 probe tests, argparse CLIs, git path migration.

---

## File Structure

Create or modify these files:

- Create: `scripts/harness/guards/__init__.py`
- Create: `scripts/harness/validation/__init__.py`
- Create: `scripts/harness/validation/run.py`
- Create: `scripts/harness/diagnostics/__init__.py`
- Create: `scripts/harness/diagnostics/demo/__init__.py`
- Create: `scripts/harness/diagnostics/depth/__init__.py`
- Create: `scripts/harness/diagnostics/visualization/__init__.py`
- Create: `scripts/harness/diagnostics/hardware/__init__.py`
- Create: `scripts/harness/benchmarks/__init__.py`
- Create: `scripts/harness/benchmarks/ffs/__init__.py`
- Create: `scripts/harness/benchmarks/sam/__init__.py`
- Create: `scripts/harness/support/__init__.py`
- Modify: `scripts/harness/_catalog.py`
- Move and modify: `scripts/harness/check_all.py` -> `scripts/harness/validation/run.py`
- Move: `scripts/harness/check_experiment_boundaries.py` -> `scripts/harness/guards/check_experiment_boundaries.py`
- Move: `scripts/harness/check_harness_catalog.py` -> `scripts/harness/guards/check_harness_catalog.py`
- Move: `scripts/harness/check_scope.py` -> `scripts/harness/guards/check_scope.py`
- Move: `scripts/harness/check_visual_architecture.py` -> `scripts/harness/guards/check_visual_architecture.py`
- Move public harness scripts according to the migration table in Task 4.
- Modify: `scripts/harness/README.md`
- Modify: `AGENTS.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/SCOPE.md`
- Modify: `docs/WORKFLOWS.md`
- Modify: `docs/HARDWARE_VALIDATION.md`
- Modify: `docs/envs.md`
- Modify: `docs/external-deps.md`
- Modify: `tests/test_check_all_smoke.py` -> content becomes validation-runner tests.
- Modify: `tests/test_experiment_boundary_smoke.py`
- Modify: `tests/test_agents_scope_contract_smoke.py`
- Modify: `tests/test_sam31_still_object_benchmark_smoke.py`
- Modify: `tests/test_sam31_mask_helper_smoke.py`
- Modify: `tests/test_realtime_single_camera_pointcloud_smoke.py`
- Create: `tests/test_harness_catalog_schema.py`
- Create: `tests/test_harness_catalog_guard.py`

Do not change formal recording, alignment, camera runtime, demo runtime, or visualization behavior.

## Migration Table

Use these exact destination paths when executing Task 4:

| Current path | New path |
| --- | --- |
| `scripts/harness/check_all.py` | `scripts/harness/validation/run.py` |
| `scripts/harness/check_experiment_boundaries.py` | `scripts/harness/guards/check_experiment_boundaries.py` |
| `scripts/harness/check_harness_catalog.py` | `scripts/harness/guards/check_harness_catalog.py` |
| `scripts/harness/check_scope.py` | `scripts/harness/guards/check_scope.py` |
| `scripts/harness/check_visual_architecture.py` | `scripts/harness/guards/check_visual_architecture.py` |
| `scripts/harness/realtime_single_camera_pointcloud.py` | `scripts/harness/diagnostics/demo/realtime_single_camera_pointcloud.py` |
| `scripts/harness/render_demo32_headless_capture.py` | `scripts/harness/diagnostics/demo/render_demo32_headless_capture.py` |
| `scripts/harness/probe_d455_ir_pair.py` | `scripts/harness/diagnostics/hardware/probe_d455_ir_pair.py` |
| `scripts/harness/probe_d455_stream_capability.py` | `scripts/harness/diagnostics/hardware/probe_d455_stream_capability.py` |
| `scripts/harness/render_d455_stream_probe_report.py` | `scripts/harness/diagnostics/hardware/render_d455_stream_probe_report.py` |
| `scripts/harness/run_wslg_open3d.sh` | `scripts/harness/diagnostics/hardware/run_wslg_open3d.sh` |
| `scripts/harness/audit_ffs_left_right.py` | `scripts/harness/diagnostics/depth/audit_ffs_left_right.py` |
| `scripts/harness/compare_face_smoothness.py` | `scripts/harness/diagnostics/depth/compare_face_smoothness.py` |
| `scripts/harness/diagnose_floating_point_sources.py` | `scripts/harness/diagnostics/depth/diagnose_floating_point_sources.py` |
| `scripts/harness/reproject_ffs_to_color.py` | `scripts/harness/diagnostics/depth/reproject_ffs_to_color.py` |
| `scripts/harness/visual_compare_depth_panels.py` | `scripts/harness/diagnostics/depth/visual_compare_depth_panels.py` |
| `scripts/harness/visual_compare_depth_triplet_ply.py` | `scripts/harness/diagnostics/depth/visual_compare_depth_triplet_ply.py` |
| `scripts/harness/visual_compare_depth_triplet_video.py` | `scripts/harness/diagnostics/depth/visual_compare_depth_triplet_video.py` |
| `scripts/harness/visual_compare_depth_video.py` | `scripts/harness/diagnostics/depth/visual_compare_depth_video.py` |
| `scripts/harness/visual_compare_reprojection.py` | `scripts/harness/diagnostics/depth/visual_compare_reprojection.py` |
| `scripts/harness/visual_compare_masked_camera_views.py` | `scripts/harness/diagnostics/visualization/visual_compare_masked_camera_views.py` |
| `scripts/harness/visual_compare_masked_pointcloud.py` | `scripts/harness/diagnostics/visualization/visual_compare_masked_pointcloud.py` |
| `scripts/harness/visual_compare_rerun.py` | `scripts/harness/diagnostics/visualization/visual_compare_rerun.py` |
| `scripts/harness/visual_compare_stereo_order_pcd.py` | `scripts/harness/diagnostics/visualization/visual_compare_stereo_order_pcd.py` |
| `scripts/harness/visual_compare_turntable.py` | `scripts/harness/diagnostics/visualization/visual_compare_turntable.py` |
| `scripts/harness/visual_make_match_board.py` | `scripts/harness/diagnostics/visualization/visual_make_match_board.py` |
| `scripts/harness/visual_make_professor_triptych.py` | `scripts/harness/diagnostics/visualization/visual_make_professor_triptych.py` |
| `scripts/harness/benchmark_ffs_configs.py` | `scripts/harness/benchmarks/ffs/benchmark_ffs_configs.py` |
| `scripts/harness/run_ffs_on_saved_pair.py` | `scripts/harness/benchmarks/ffs/run_ffs_on_saved_pair.py` |
| `scripts/harness/run_ffs_static_replay_matrix.py` | `scripts/harness/benchmarks/ffs/run_ffs_static_replay_matrix.py` |
| `scripts/harness/verify_ffs_demo.py` | `scripts/harness/benchmarks/ffs/verify_ffs_demo.py` |
| `scripts/harness/verify_ffs_single_engine_tensorrt_wsl.py` | `scripts/harness/benchmarks/ffs/verify_ffs_single_engine_tensorrt_wsl.py` |
| `scripts/harness/verify_ffs_tensorrt_windows.py` | `scripts/harness/benchmarks/ffs/verify_ffs_tensorrt_windows.py` |
| `scripts/harness/verify_ffs_tensorrt_wsl.py` | `scripts/harness/benchmarks/ffs/verify_ffs_tensorrt_wsl.py` |
| `scripts/harness/benchmark_sam31_still_object_views.py` | `scripts/harness/benchmarks/sam/benchmark_sam31_still_object_views.py` |
| `scripts/harness/generate_sam31_masks.py` | `scripts/harness/diagnostics/visualization/generate_sam31_masks.py` |
| `scripts/harness/object_case_registry.py` | `scripts/harness/support/object_case_registry.py` |
| `scripts/harness/sam31_mask_helper.py` | `scripts/harness/support/sam31_mask_helper.py` |
| `scripts/harness/cleanup_different_types_cases.py` | `scripts/harness/support/cleanup_different_types_cases.py` |
| `scripts/harness/experiments/benchmark_edgetam_trt_components.py` | `scripts/harness/experiments/edgetam/benchmark_edgetam_trt_components.py` |
| `scripts/harness/experiments/inspect_edgetam_onnx.py` | `scripts/harness/experiments/edgetam/inspect_edgetam_onnx.py` |
| `scripts/harness/experiments/probe_edgetam_video_trt_compile.py` | `scripts/harness/experiments/edgetam/probe_edgetam_video_trt_compile.py` |
| `scripts/harness/experiments/run_edgetam_video_masks.py` | `scripts/harness/experiments/edgetam/run_edgetam_video_masks.py` |
| `scripts/harness/experiments/run_edgetam_vs_sam21_compile_ablation.py` | `scripts/harness/experiments/edgetam/run_edgetam_vs_sam21_compile_ablation.py` |
| `scripts/harness/experiments/run_hf_edgetam_streaming_realcase.py` | `scripts/harness/experiments/edgetam/run_hf_edgetam_streaming_realcase.py` |
| `scripts/harness/experiments/run_sloth_base_motion_fused_pcd_overlay_2x3_gif.py` | `scripts/harness/experiments/edgetam/run_sloth_base_motion_fused_pcd_overlay_2x3_gif.py` |
| `scripts/harness/experiments/run_sloth_base_motion_mask_overlay_3x3_gif.py` | `scripts/harness/experiments/edgetam/run_sloth_base_motion_mask_overlay_3x3_gif.py` |
| `scripts/harness/experiments/run_sloth_set2_hf_edgetam_hand_object_pcd_gif.py` | `scripts/harness/experiments/edgetam/run_sloth_set2_hf_edgetam_hand_object_pcd_gif.py` |
| `scripts/harness/experiments/run_sloth_set2_hf_edgetam_streaming_pcd_xor_gif.py` | `scripts/harness/experiments/edgetam/run_sloth_set2_hf_edgetam_streaming_pcd_xor_gif.py` |
| `scripts/harness/experiments/run_ffs_confidence_filter_sweep.py` | `scripts/harness/experiments/ffs/run_ffs_confidence_filter_sweep.py` |
| `scripts/harness/experiments/visual_compare_ffs_confidence_filter_pcd.py` | `scripts/harness/experiments/ffs/visual_compare_ffs_confidence_filter_pcd.py` |
| `scripts/harness/experiments/visual_compare_ffs_confidence_threshold_sweep_pcd.py` | `scripts/harness/experiments/ffs/visual_compare_ffs_confidence_threshold_sweep_pcd.py` |
| `scripts/harness/experiments/visual_compare_ffs_mask_erode_multipage_sweep_pcd.py` | `scripts/harness/experiments/ffs/visual_compare_ffs_mask_erode_multipage_sweep_pcd.py` |
| `scripts/harness/experiments/visual_compare_ffs_mask_erode_sweep_pcd.py` | `scripts/harness/experiments/ffs/visual_compare_ffs_mask_erode_sweep_pcd.py` |
| `scripts/harness/experiments/visual_compare_native_ffs_fused_pcd.py` | `scripts/harness/experiments/ffs/visual_compare_native_ffs_fused_pcd.py` |
| `scripts/harness/experiments/visualize_ffs_static_confidence_panels.py` | `scripts/harness/experiments/ffs/visualize_ffs_static_confidence_panels.py` |
| `scripts/harness/experiments/visualize_ffs_static_confidence_pcd_panels.py` | `scripts/harness/experiments/ffs/visualize_ffs_static_confidence_pcd_panels.py` |
| `scripts/harness/experiments/visualize_still_object_orbit_gif.py` | `scripts/harness/experiments/visualization/visualize_still_object_orbit_gif.py` |
| `scripts/harness/experiments/visualize_still_object_rope_6x2_orbit_erode_sweep_gif.py` | `scripts/harness/experiments/visualization/visualize_still_object_rope_6x2_orbit_erode_sweep_gif.py` |
| `scripts/harness/experiments/visualize_still_object_rope_6x2_orbit_gif.py` | `scripts/harness/experiments/visualization/visualize_still_object_rope_6x2_orbit_gif.py` |
| `scripts/harness/experiments/run_still_object_round1_projection_panel.py` | `scripts/harness/experiments/visualization/run_still_object_round1_projection_panel.py` |
| `scripts/harness/experiments/run_sam21_checkpoint_ladder_3x5_gifs.py` | `scripts/harness/experiments/sam/run_sam21_checkpoint_ladder_3x5_gifs.py` |
| `scripts/harness/experiments/visualize_sam21_edgetam_mask_overlay_3x3_gif.py` | `scripts/harness/experiments/sam/visualize_sam21_edgetam_mask_overlay_3x3_gif.py` |

## Task 1: Add failing tests for the new catalog schema

**Files:**
- Create: `tests/test_harness_catalog_schema.py`
- Modify later: `scripts/harness/_catalog.py`

- [ ] **Step 1: Write the failing schema tests**

Create `tests/test_harness_catalog_schema.py` with:

```python
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
        self.assertEqual(entry.validation_profile, "smoke")
        self.assertFalse(entry.help)
        self.assertTrue(entry.automatic)
        self.assertEqual(entry.requires, ())

    def test_help_scripts_uses_validation_profiles(self) -> None:
        smoke_scripts = _catalog.help_scripts("smoke")
        deterministic_scripts = _catalog.help_scripts("deterministic")
        self.assertIsInstance(smoke_scripts, tuple)
        self.assertIsInstance(deterministic_scripts, tuple)
        self.assertTrue(set(smoke_scripts).issubset(set(deterministic_scripts)))

    def test_entries_by_lifecycle_replaces_entries_by_category(self) -> None:
        grouped = _catalog.entries_by_lifecycle()
        self.assertIn("guards", grouped)
        self.assertIn("validation", grouped)
        self.assertNotIn("checks", grouped)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the schema tests and verify they fail**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_harness_catalog_schema
```

Expected: FAIL because `_catalog.VALIDATION_PROFILES`, `_catalog.LIFECYCLES`, and `entries_by_lifecycle` do not exist yet.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_harness_catalog_schema.py
git commit -m "test: specify harness catalog schema"
```

## Task 2: Implement the catalog schema and compatibility-free helpers

**Files:**
- Modify: `scripts/harness/_catalog.py`
- Test: `tests/test_harness_catalog_schema.py`

- [ ] **Step 1: Replace the schema header in `_catalog.py`**

Replace the current `HelpProfile` and `HarnessEntry` definitions with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ValidationProfile = Literal["smoke", "deterministic", "hardware", "exhaustive"]
Lifecycle = Literal["guards", "validation", "diagnostics", "benchmarks", "experiments", "support"]

VALIDATION_PROFILES: tuple[str, ...] = ("smoke", "deterministic", "hardware", "exhaustive")
LIFECYCLES: tuple[str, ...] = ("guards", "validation", "diagnostics", "benchmarks", "experiments", "support")


@dataclass(frozen=True)
class HarnessEntry:
    path: str
    lifecycle: Lifecycle
    domain: str
    summary: str
    validation_profile: ValidationProfile | None = None
    help: bool = False
    automatic: bool = True
    requires: tuple[str, ...] = ()
```

- [ ] **Step 2: Replace catalog helper functions**

Replace `entries_by_category` and `help_scripts` with:

```python
def entries_by_lifecycle() -> dict[str, tuple[HarnessEntry, ...]]:
    grouped: dict[str, list[HarnessEntry]] = {}
    for entry in CATALOG:
        grouped.setdefault(entry.lifecycle, []).append(entry)
    return {lifecycle: tuple(entries) for lifecycle, entries in grouped.items()}


def entries_for_profile(profile: ValidationProfile, *, include_manual: bool = False) -> tuple[HarnessEntry, ...]:
    if profile == "smoke":
        allowed = {"smoke"}
    elif profile == "deterministic":
        allowed = {"smoke", "deterministic"}
    elif profile == "exhaustive":
        allowed = {"smoke", "deterministic", "exhaustive"}
    elif profile == "hardware":
        allowed = {"hardware"}
    else:
        raise ValueError(f"Unsupported profile: {profile}")

    entries = []
    for entry in CATALOG:
        if entry.validation_profile not in allowed:
            continue
        if not include_manual and not entry.automatic:
            continue
        entries.append(entry)
    return tuple(entries)


def help_scripts(profile: ValidationProfile, *, include_manual: bool = False) -> tuple[str, ...]:
    return tuple(
        entry.path
        for entry in entries_for_profile(profile, include_manual=include_manual)
        if entry.help
    )
```

- [ ] **Step 3: Temporarily convert existing catalog entries in place**

Before file moves, convert each old `HarnessEntry` call to the new signature while keeping current paths. Use this profile mapping:

```text
checks -> lifecycle="guards", domain="repo", validation_profile="smoke", help=False
hardware_external D455/probe/report/WSLg/realtime scripts -> lifecycle="diagnostics", domain="hardware" or "demo", validation_profile="hardware", automatic=False, requires=("camera",) for D455 probes, requires=("camera", "gpu", "gui") for realtime/Open3D entries
hardware_external FFS/TensorRT proof scripts -> lifecycle="benchmarks", domain="ffs", validation_profile="hardware", automatic=False, requires=("gpu", "tensorrt", "external_repo")
hardware_external SAM still-object benchmark -> lifecycle="benchmarks", domain="sam", validation_profile="exhaustive", automatic=True, requires=("gpu", "external_repo")
mask_support -> lifecycle="support" for imported helpers and lifecycle="diagnostics" for public CLIs, validation_profile="deterministic" when help was old full
formal_cleanup -> lifecycle="support", domain="data", validation_profile="deterministic", help=True
current_compare -> lifecycle="diagnostics", domain equal to the diagnostics subdirectory in the migration table, validation_profile="smoke" for old quick and "deterministic" for old full, help=True
experiments -> lifecycle="experiments", domain equal to the experiment subdirectory in the migration table, validation_profile="exhaustive" when old full, help=True if old help_profile was full
focused_diagnostics -> lifecycle="diagnostics", domain="depth", validation_profile="deterministic", help=True
```

Add the validation runner entry:

```python
HarnessEntry(
    "scripts/harness/validation/run.py",
    "validation",
    "runner",
    "Catalog-driven validation profile runner.",
)
```

- [ ] **Step 4: Run the schema tests and verify they pass**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_harness_catalog_schema
```

Expected: PASS.

- [ ] **Step 5: Commit the catalog schema**

```bash
git add scripts/harness/_catalog.py tests/test_harness_catalog_schema.py
git commit -m "feat: add harness catalog validation schema"
```

## Task 3: Create the validation runner with new profiles

**Files:**
- Create: `scripts/harness/validation/__init__.py`
- Move and modify: `scripts/harness/check_all.py` -> `scripts/harness/validation/run.py`
- Modify: `tests/test_check_all_smoke.py`

- [ ] **Step 1: Move the old runner into the validation package**

Run:

```bash
mkdir -p scripts/harness/validation
touch scripts/harness/validation/__init__.py
git mv scripts/harness/check_all.py scripts/harness/validation/run.py
```

- [ ] **Step 2: Replace `tests/test_check_all_smoke.py` imports and class name**

Edit the top of `tests/test_check_all_smoke.py` to:

```python
from __future__ import annotations

from pathlib import Path
import unittest

from scripts.harness.validation import run as validation_run
```

Rename `CheckAllSmokeTest` to `ValidationRunnerSmokeTest`, and replace every `check_all.` reference with `validation_run.`.

- [ ] **Step 3: Replace old quick/full argument tests**

Replace the first two tests with:

```python
    def test_parse_args_defaults_to_smoke_profile(self) -> None:
        args = validation_run.parse_args([])
        self.assertEqual(args.profile, "smoke")
        self.assertFalse(args.run_hardware)

    def test_parse_args_accepts_new_profiles(self) -> None:
        for profile in ("smoke", "deterministic", "hardware", "exhaustive"):
            with self.subTest(profile=profile):
                args = validation_run.parse_args(["--profile", profile])
                self.assertEqual(args.profile, profile)
```

- [ ] **Step 4: Replace `parse_args` in `validation/run.py`**

Use this implementation:

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run catalog-driven single-camera validation profiles."
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
        help="Run hardware-profile help/proof commands instead of only listing them.",
    )
    return parser.parse_args(argv)
```

- [ ] **Step 5: Replace command generation in `validation/run.py`**

Keep the existing formal help script constants, but rename profile constants and build commands with:

```python
SMOKE_UNITTEST_MODULES: tuple[str, ...] = QUICK_UNITTEST_MODULES
DETERMINISTIC_ONLY_UNITTEST_MODULES: tuple[str, ...] = FULL_ONLY_UNITTEST_MODULES
EXHAUSTIVE_ONLY_UNITTEST_MODULES: tuple[str, ...] = ()


def _profile_unittest_batches(profile: str) -> tuple[tuple[str, ...], ...]:
    if profile == "smoke":
        return (SMOKE_UNITTEST_MODULES,)
    if profile == "deterministic":
        modules = _unique((*SMOKE_UNITTEST_MODULES, *DETERMINISTIC_ONLY_UNITTEST_MODULES))
        return tuple((module,) for module in modules)
    if profile == "exhaustive":
        modules = _unique((*SMOKE_UNITTEST_MODULES, *DETERMINISTIC_ONLY_UNITTEST_MODULES, *EXHAUSTIVE_ONLY_UNITTEST_MODULES))
        return tuple((module,) for module in modules)
    if profile == "hardware":
        return ()
    raise ValueError(f"Unsupported profile: {profile}")


def _guard_commands(*, python: str) -> list[list[str]]:
    return [
        [python, "scripts/harness/guards/check_harness_catalog.py"],
        [python, "scripts/harness/guards/check_experiment_boundaries.py"],
        [python, "scripts/harness/guards/check_visual_architecture.py"],
        [python, "-m", "scripts.harness.guards.check_scope"],
    ]


def _catalog_help_commands(*, python: str, profile: str, include_manual: bool = False) -> list[list[str]]:
    return [[python, script, "--help"] for script in help_scripts(profile, include_manual=include_manual)]


def _formal_scripts_for_profile(profile: str) -> tuple[str, ...]:
    if profile == "smoke":
        return BASE_FORMAL_HELP_SCRIPTS
    if profile in {"deterministic", "exhaustive"}:
        return _unique((*BASE_FORMAL_HELP_SCRIPTS, *FULL_ONLY_FORMAL_HELP_SCRIPTS, *DEMO_HELP_SCRIPTS))
    if profile == "hardware":
        return ()
    raise ValueError(f"Unsupported profile: {profile}")


def build_commands(*, python: str, profile: str, run_hardware: bool = False) -> list[list[str]]:
    if profile == "hardware" and not run_hardware:
        return []
    if profile == "hardware":
        return _catalog_help_commands(python=python, profile=profile, include_manual=True)
    if profile in {"smoke", "deterministic", "exhaustive"}:
        commands = [
            *_help_commands(python=python, scripts=_formal_scripts_for_profile(profile)),
            *_catalog_help_commands(python=python, profile=profile),
            *_guard_commands(python=python),
            *_unittest_commands(python=python, module_batches=_profile_unittest_batches(profile)),
        ]
        if profile == "exhaustive":
            commands.extend(_pytest_commands(python=python))
        return commands
    raise ValueError(f"Unsupported profile: {profile}")
```

- [ ] **Step 6: Update `main` in `validation/run.py`**

Use:

```python
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    python = sys.executable
    print(f"[validation] profile={args.profile}")
    if args.profile == "hardware" and not args.run_hardware:
        for cmd in _catalog_help_commands(python=python, profile="hardware", include_manual=True):
            print(f"[validation] hardware manual: {' '.join(cmd)}")
        print("[validation] hardware profile listed only; pass --run-hardware to run listed help/proof commands")
        return 0
    for cmd in build_commands(python=python, profile=args.profile, run_hardware=args.run_hardware):
        run(cmd)
    print(f"[validation] {args.profile} checks passed")
    return 0
```

- [ ] **Step 7: Update the runner tests for new command paths**

In `tests/test_check_all_smoke.py`, assert these commands:

```python
self.assertIn(["python", "scripts/harness/diagnostics/demo/render_demo32_headless_capture.py", "--help"], commands)
self.assertIn(["python", "scripts/harness/diagnostics/depth/visual_compare_depth_panels.py", "--help"], commands)
self.assertIn(["python", "scripts/harness/diagnostics/depth/visual_compare_reprojection.py", "--help"], commands)
self.assertIn(["python", "scripts/harness/diagnostics/visualization/visual_compare_turntable.py", "--help"], commands)
self.assertIn(["python", "scripts/harness/guards/check_harness_catalog.py"], commands)
self.assertIn(["python", "scripts/harness/guards/check_experiment_boundaries.py"], commands)
self.assertIn(["python", "scripts/harness/guards/check_visual_architecture.py"], commands)
```

Also assert old quick/full are gone:

```python
with self.assertRaises(SystemExit):
    validation_run.parse_args(["--full"])
with self.assertRaises(SystemExit):
    validation_run.parse_args(["--profile", "quick"])
```

- [ ] **Step 8: Run the runner tests and verify expected failures**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_check_all_smoke
```

Expected before Task 4: FAIL on missing moved script paths. That failure is expected because paths are not moved yet.

- [ ] **Step 9: Commit the validation runner skeleton**

```bash
git add scripts/harness/validation tests/test_check_all_smoke.py
git commit -m "feat: introduce harness validation runner"
```

## Task 4: Move harness files into lifecycle directories

**Files:**
- Move all paths listed in the migration table.
- Create package `__init__.py` files listed in File Structure.
- Remove harness `__pycache__` directories.

- [ ] **Step 1: Create lifecycle directories**

Run:

```bash
mkdir -p \
  scripts/harness/guards \
  scripts/harness/validation \
  scripts/harness/diagnostics/demo \
  scripts/harness/diagnostics/depth \
  scripts/harness/diagnostics/visualization \
  scripts/harness/diagnostics/hardware \
  scripts/harness/benchmarks/ffs \
  scripts/harness/benchmarks/sam \
  scripts/harness/support \
  scripts/harness/experiments/edgetam \
  scripts/harness/experiments/ffs \
  scripts/harness/experiments/sam \
  scripts/harness/experiments/visualization
touch \
  scripts/harness/guards/__init__.py \
  scripts/harness/diagnostics/__init__.py \
  scripts/harness/diagnostics/demo/__init__.py \
  scripts/harness/diagnostics/depth/__init__.py \
  scripts/harness/diagnostics/visualization/__init__.py \
  scripts/harness/diagnostics/hardware/__init__.py \
  scripts/harness/benchmarks/__init__.py \
  scripts/harness/benchmarks/ffs/__init__.py \
  scripts/harness/benchmarks/sam/__init__.py \
  scripts/harness/support/__init__.py \
  scripts/harness/experiments/edgetam/__init__.py \
  scripts/harness/experiments/ffs/__init__.py \
  scripts/harness/experiments/sam/__init__.py \
  scripts/harness/experiments/visualization/__init__.py
```

- [ ] **Step 2: Move files using the migration table**

Run one `git mv` per row in the migration table. Example:

```bash
git mv scripts/harness/check_experiment_boundaries.py scripts/harness/guards/check_experiment_boundaries.py
git mv scripts/harness/render_demo32_headless_capture.py scripts/harness/diagnostics/demo/render_demo32_headless_capture.py
git mv scripts/harness/visual_compare_depth_panels.py scripts/harness/diagnostics/depth/visual_compare_depth_panels.py
```

Continue until every table row has been moved.

- [ ] **Step 3: Remove committed harness caches**

Run:

```bash
find scripts/harness -type d -name __pycache__ -prune -exec git rm -r {} +
```

Expected: all tracked `scripts/harness/**/__pycache__` files are staged for deletion.

- [ ] **Step 4: Verify no root-level public harness scripts remain**

Run:

```bash
find scripts/harness -maxdepth 1 -type f \( -name '*.py' -o -name '*.sh' \) | sort
```

Expected:

```text
scripts/harness/README.md
scripts/harness/__init__.py
scripts/harness/_catalog.py
```

If `find` omits `README.md` because the command only matches Python and shell files, the acceptable output is:

```text
scripts/harness/__init__.py
scripts/harness/_catalog.py
```

- [ ] **Step 5: Commit the file migration**

```bash
git add scripts/harness
git commit -m "refactor: move harness files into lifecycle directories"
```

## Task 5: Rebuild catalog entries with final paths and metadata

**Files:**
- Modify: `scripts/harness/_catalog.py`
- Test: `tests/test_harness_catalog_schema.py`

- [ ] **Step 1: Update `CATALOG` paths**

For every moved file, update `entry.path` to the new path from the migration table. The catalog must not contain any of these old path prefixes:

```text
scripts/harness/check_all.py
scripts/harness/check_
scripts/harness/visual_
scripts/harness/probe_
scripts/harness/verify_
scripts/harness/run_ffs
scripts/harness/benchmark_
scripts/harness/experiments/run_
scripts/harness/experiments/visual_
```

The only allowed root-level harness paths in catalog are:

```text
scripts/harness/_catalog.py
scripts/harness/README.md
```

Neither root-level file needs a `HarnessEntry`.

- [ ] **Step 2: Assign final metadata**

Use these profile decisions:

```text
smoke:
  scripts/harness/guards/check_harness_catalog.py
  scripts/harness/guards/check_experiment_boundaries.py
  scripts/harness/guards/check_visual_architecture.py
  scripts/harness/guards/check_scope.py
  scripts/harness/diagnostics/demo/render_demo32_headless_capture.py
  scripts/harness/diagnostics/depth/visual_compare_depth_panels.py
  scripts/harness/diagnostics/depth/visual_compare_reprojection.py
  scripts/harness/diagnostics/visualization/visual_compare_turntable.py

deterministic:
  all old current_compare full entries
  all focused_diagnostics entries
  generate_sam31_masks.py
  reproject_ffs_to_color.py
  cleanup_different_types_cases.py

hardware:
  D455 probes
  WSLg Open3D wrapper
  realtime_single_camera_pointcloud.py
  FFS/TensorRT proof-of-life scripts that need external engines, GPU, or GUI

exhaustive:
  experiments under scripts/harness/experiments/**
  long SAM/FFS benchmark entries that do not need live camera hardware
```

Set `automatic=False` for every `validation_profile="hardware"` entry. Set `requires` with concrete labels:

```text
camera: D455 or live RealSense required
gpu: CUDA GPU required
gui: Open3D or WSLg window required
tensorrt: TensorRT engine/runtime required
external_repo: external FFS, EdgeTAM, SAM, or checkpoint tree required
```

- [ ] **Step 3: Add assertions to schema tests**

Append this test to `tests/test_harness_catalog_schema.py`:

```python
    def test_no_old_root_public_harness_paths_remain_in_catalog(self) -> None:
        old_fragments = (
            "scripts/harness/check_all.py",
            "scripts/harness/visual_",
            "scripts/harness/probe_",
            "scripts/harness/verify_",
            "scripts/harness/run_ffs",
            "scripts/harness/benchmark_",
        )
        for entry in _catalog.CATALOG:
            with self.subTest(path=entry.path):
                self.assertFalse(any(fragment in entry.path for fragment in old_fragments))
```

- [ ] **Step 4: Run schema tests**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_harness_catalog_schema
```

Expected: PASS.

- [ ] **Step 5: Commit catalog path updates**

```bash
git add scripts/harness/_catalog.py tests/test_harness_catalog_schema.py
git commit -m "refactor: catalog lifecycle harness paths"
```

## Task 6: Update catalog and boundary guards

**Files:**
- Modify: `scripts/harness/guards/check_harness_catalog.py`
- Modify: `scripts/harness/guards/check_experiment_boundaries.py`
- Create: `tests/test_harness_catalog_guard.py`
- Modify: `tests/test_experiment_boundary_smoke.py`

- [ ] **Step 1: Write guard tests for lifecycle rules**

Create `tests/test_harness_catalog_guard.py` with:

```python
from __future__ import annotations

import unittest

from scripts.harness.guards import check_harness_catalog


class HarnessCatalogGuardTest(unittest.TestCase):
    def test_current_catalog_has_no_violations(self) -> None:
        self.assertEqual(check_harness_catalog.collect_violations(), [])

    def test_no_harness_pycache_directories_are_committed(self) -> None:
        pycache_dirs = sorted(check_harness_catalog.HARNESS_ROOT.rglob("__pycache__"))
        self.assertEqual(pycache_dirs, [])

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
```

- [ ] **Step 2: Replace known categories in catalog guard**

In `scripts/harness/guards/check_harness_catalog.py`, import the new constants:

```python
from scripts.harness._catalog import CATALOG, LIFECYCLES, VALIDATION_PROFILES
```

Remove `KNOWN_CATEGORIES`.

- [ ] **Step 3: Update private file allowlist**

Use:

```python
PRIVATE_PYTHON_FILES = {
    HARNESS_ROOT / "__init__.py",
    HARNESS_ROOT / "_catalog.py",
}
PACKAGE_INIT_FILES = {path for path in HARNESS_ROOT.rglob("__init__.py")}
```

- [ ] **Step 4: Replace entry validation block**

Use this block inside `collect_violations`:

```python
    for entry in CATALOG:
        path = ROOT / entry.path
        if entry.lifecycle not in LIFECYCLES:
            violations.append(f"Unknown lifecycle for {entry.path}: {entry.lifecycle}")
        if entry.validation_profile is not None and entry.validation_profile not in VALIDATION_PROFILES:
            violations.append(f"Unknown validation profile for {entry.path}: {entry.validation_profile}")
        if not path.exists():
            violations.append(f"Catalog path does not exist: {entry.path}")
            continue
        if entry.help and path.suffix != ".py":
            violations.append(f"Non-Python path cannot have help coverage: {entry.path}")
        expected_prefix = HARNESS_ROOT / entry.lifecycle
        if entry.lifecycle in {"guards", "validation", "diagnostics", "benchmarks", "experiments", "support"}:
            if not _is_under(path, expected_prefix):
                violations.append(f"Lifecycle/path mismatch for {entry.path}: expected under scripts/harness/{entry.lifecycle}/")
        is_experiment_path = _is_under(path, HARNESS_ROOT / "experiments")
        if entry.lifecycle == "experiments" and not is_experiment_path:
            violations.append(f"Experiment entry is outside experiments/: {entry.path}")
        if is_experiment_path and entry.lifecycle != "experiments":
            violations.append(f"Experiment path has non-experiment lifecycle: {entry.path}")
        if entry.validation_profile == "hardware" and entry.automatic:
            violations.append(f"Hardware entry must be manual: {entry.path}")
```

- [ ] **Step 5: Replace uncataloged file loop**

Use:

```python
    cataloged_python = {path for path in unique_paths if path.suffix == ".py"}
    for path in sorted(HARNESS_ROOT.rglob("*.py")):
        if path in PRIVATE_PYTHON_FILES or path in PACKAGE_INIT_FILES:
            continue
        if path not in cataloged_python:
            violations.append(f"Uncataloged harness Python file: {path.relative_to(ROOT)}")

    for path in sorted(HARNESS_ROOT.rglob("__pycache__")):
        violations.append(f"Committed harness cache directory: {path.relative_to(ROOT)}")
```

- [ ] **Step 6: Update experiment boundary guard root path**

In `scripts/harness/guards/check_experiment_boundaries.py`, keep `HARNESS_ROOT = ROOT / "scripts" / "harness"` and keep `EXPERIMENT_IMPORT_PREFIXES` unchanged:

```python
EXPERIMENT_IMPORT_PREFIXES = (
    "data_process.visualization.experiments",
    "scripts.harness.experiments",
)
```

The existing `_is_under(path, HARNESS_ROOT / "experiments")` logic still covers nested experiment domain subdirectories.

- [ ] **Step 7: Update boundary smoke test imports**

In `tests/test_experiment_boundary_smoke.py`, replace imports from `scripts.harness.check_experiment_boundaries` with:

```python
from scripts.harness.guards import check_experiment_boundaries
```

- [ ] **Step 8: Run guard tests**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v \
  tests.test_harness_catalog_guard \
  tests.test_experiment_boundary_smoke
```

Expected: PASS.

- [ ] **Step 9: Commit guard updates**

```bash
git add scripts/harness/guards tests/test_harness_catalog_guard.py tests/test_experiment_boundary_smoke.py
git commit -m "test: enforce harness lifecycle catalog guard"
```

## Task 7: Update imports and CLI path references

**Files:**
- Modify: `tests/test_sam31_still_object_benchmark_smoke.py`
- Modify: `tests/test_sam31_mask_helper_smoke.py`
- Modify: `tests/test_realtime_single_camera_pointcloud_smoke.py`
- Modify: experiment scripts that call moved harness scripts.

- [ ] **Step 1: Update test command paths**

Make these replacements:

```text
scripts/harness/benchmark_sam31_still_object_views.py
-> scripts/harness/benchmarks/sam/benchmark_sam31_still_object_views.py

scripts/harness/generate_sam31_masks.py
-> scripts/harness/diagnostics/visualization/generate_sam31_masks.py

scripts/harness/realtime_single_camera_pointcloud.py
-> scripts/harness/diagnostics/demo/realtime_single_camera_pointcloud.py
```

- [ ] **Step 2: Update experiment subprocess call**

In `scripts/harness/experiments/edgetam/run_sloth_base_motion_mask_overlay_3x3_gif.py`, replace:

```python
str(ROOT / "scripts/harness/generate_sam31_masks.py"),
```

with:

```python
str(ROOT / "scripts/harness/diagnostics/visualization/generate_sam31_masks.py"),
```

- [ ] **Step 3: Scan for old moved paths in source and tests**

Run:

```bash
rg -n "scripts/harness/(check_all.py|check_|visual_|probe_|verify_|run_ffs|benchmark_|generate_sam31_masks.py|realtime_single_camera_pointcloud.py)" scripts tests AGENTS.md docs/ARCHITECTURE.md docs/SCOPE.md docs/WORKFLOWS.md docs/HARDWARE_VALIDATION.md docs/envs.md docs/external-deps.md
```

Expected after this task: matches remain only in docs that are updated in Task 8, not in `scripts/` or `tests/`.

- [ ] **Step 4: Run targeted import/path tests**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v \
  tests.test_sam31_still_object_benchmark_smoke \
  tests.test_sam31_mask_helper_smoke \
  tests.test_realtime_single_camera_pointcloud_smoke
```

Expected: PASS.

- [ ] **Step 5: Commit path reference updates**

```bash
git add tests scripts/harness/experiments
git commit -m "refactor: update harness script path references"
```

## Task 8: Update docs and branch policy references

**Files:**
- Modify: `AGENTS.md`
- Modify: `scripts/harness/README.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/SCOPE.md`
- Modify: `docs/WORKFLOWS.md`
- Modify: `docs/HARDWARE_VALIDATION.md`
- Modify: `docs/envs.md`
- Modify: `docs/external-deps.md`
- Modify: `tests/test_agents_scope_contract_smoke.py`

- [ ] **Step 1: Update validation commands**

Replace:

```bash
python scripts/harness/check_all.py
python scripts/harness/check_all.py --full
```

with:

```bash
python scripts/harness/validation/run.py --profile smoke
python scripts/harness/validation/run.py --profile deterministic
python scripts/harness/validation/run.py --profile exhaustive
```

In `AGENTS.md`, replace the required workflow bullets with:

```text
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
- use `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile exhaustive` when the change is broad enough that the default smoke profile is not sufficient
```

- [ ] **Step 2: Update File Map paths in `AGENTS.md`**

Use these replacement paths:

```text
scripts/harness/realtime_single_camera_pointcloud.py
-> scripts/harness/diagnostics/demo/realtime_single_camera_pointcloud.py
scripts/harness/check_scope.py
-> scripts/harness/guards/check_scope.py
scripts/harness/check_visual_architecture.py
-> scripts/harness/guards/check_visual_architecture.py
scripts/harness/check_experiment_boundaries.py
-> scripts/harness/guards/check_experiment_boundaries.py
scripts/harness/visual_compare_depth_panels.py
-> scripts/harness/diagnostics/depth/visual_compare_depth_panels.py
scripts/harness/visual_compare_reprojection.py
-> scripts/harness/diagnostics/depth/visual_compare_reprojection.py
scripts/harness/visual_compare_depth_video.py
-> scripts/harness/diagnostics/depth/visual_compare_depth_video.py
scripts/harness/visual_compare_depth_triplet_ply.py
-> scripts/harness/diagnostics/depth/visual_compare_depth_triplet_ply.py
scripts/harness/visual_compare_depth_triplet_video.py
-> scripts/harness/diagnostics/depth/visual_compare_depth_triplet_video.py
scripts/harness/visual_compare_rerun.py
-> scripts/harness/diagnostics/visualization/visual_compare_rerun.py
scripts/harness/visual_compare_turntable.py
-> scripts/harness/diagnostics/visualization/visual_compare_turntable.py
scripts/harness/visual_make_professor_triptych.py
-> scripts/harness/diagnostics/visualization/visual_make_professor_triptych.py
scripts/harness/visual_make_match_board.py
-> scripts/harness/diagnostics/visualization/visual_make_match_board.py
scripts/harness/audit_ffs_left_right.py
-> scripts/harness/diagnostics/depth/audit_ffs_left_right.py
scripts/harness/visual_compare_stereo_order_pcd.py
-> scripts/harness/diagnostics/visualization/visual_compare_stereo_order_pcd.py
scripts/harness/compare_face_smoothness.py
-> scripts/harness/diagnostics/depth/compare_face_smoothness.py
```

- [ ] **Step 3: Rewrite `scripts/harness/README.md` catalog section**

Replace the old category table with lifecycle/profile text:

```markdown
## Current Catalog Shape

`_catalog.py` records public harness entrypoints by lifecycle, domain, validation profile, help coverage, manual execution, and external requirements.

| Lifecycle | Meaning |
| --- | --- |
| `guards` | Deterministic repo and architecture guards. |
| `validation` | Validation profile runner and profile logic. |
| `diagnostics` | Operator-facing inspection tools and bounded diagnostics. |
| `benchmarks` | Benchmarking and external proof-of-life utilities. |
| `experiments` | Isolated research workflows. |
| `support` | Shared helper modules used by harness entrypoints. |

| Profile | Use |
| --- | --- |
| `smoke` | Cheap deterministic validation for everyday changes. |
| `deterministic` | Smoke plus broader offline tests and help checks. |
| `hardware` | Manual hardware, GUI, external-service, and environment proof-of-life checks. |
| `exhaustive` | Smoke plus deterministic plus broader long-running offline tests. |
```

- [ ] **Step 4: Update scope contract tests**

In `tests/test_agents_scope_contract_smoke.py`, replace old harness paths with the new paths listed in Step 2.

- [ ] **Step 5: Scan living docs for old validation commands**

Run:

```bash
rg -n "scripts/harness/check_all.py|check_all.py --full|--full|quick profile|full profile" AGENTS.md docs scripts/harness/README.md tests
```

Expected: no matches except historical `docs/exec-plans/active/**` if that directory is included in a broader scan. Do not update historical execution plans in this restructure.

- [ ] **Step 6: Run doc contract tests**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_agents_scope_contract_smoke
```

Expected: PASS.

- [ ] **Step 7: Commit docs**

```bash
git add AGENTS.md scripts/harness/README.md docs tests/test_agents_scope_contract_smoke.py
git commit -m "docs: document harness validation profiles"
```

## Task 9: Final validation and push

**Files:**
- No planned source edits. Fix only failures caused by the migration.

- [ ] **Step 1: Run catalog, runner, and path tests**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v \
  tests.test_harness_catalog_schema \
  tests.test_harness_catalog_guard \
  tests.test_check_all_smoke \
  tests.test_experiment_boundary_smoke \
  tests.test_agents_scope_contract_smoke
```

Expected: PASS.

- [ ] **Step 2: Run the smoke validation profile**

Run:

```bash
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke
```

Expected: PASS and final line:

```text
[validation] smoke checks passed
```

- [ ] **Step 3: Run deterministic validation**

Run:

```bash
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile deterministic
```

Expected: PASS and final line:

```text
[validation] deterministic checks passed
```

- [ ] **Step 4: Run exhaustive validation if time and environment allow**

Run:

```bash
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile exhaustive
```

Expected: PASS and final line:

```text
[validation] exhaustive checks passed
```

If this profile exceeds the local time budget, keep the failure output and finish with smoke plus deterministic results.

- [ ] **Step 5: Verify no old root public harness files remain**

Run:

```bash
find scripts/harness -maxdepth 1 -type f \( -name '*.py' -o -name '*.sh' \) | sort
```

Expected:

```text
scripts/harness/__init__.py
scripts/harness/_catalog.py
```

- [ ] **Step 6: Verify the working tree**

Run:

```bash
git status --short
```

Expected: no output.

- [ ] **Step 7: Push the completed restructure**

Run:

```bash
git push origin single-camera
```

Expected: push succeeds to `origin/single-camera`.

## Self-Review Checklist

- Every approved design requirement maps to at least one task:
  - lifecycle directories: Task 4
  - catalog source of truth and schema: Tasks 1, 2, 5
  - validation profiles: Task 3
  - no old public root scripts: Tasks 4, 6, 9
  - no compatibility shims: Tasks 4, 8
  - experiment isolation: Task 6
  - cache cleanup: Task 4
  - docs/tests updates: Tasks 7, 8
- The plan keeps formal runtime behavior unchanged.
- The plan does not run hardware checks automatically.
- The plan preserves historical execution plan text instead of rewriting old validation records.
