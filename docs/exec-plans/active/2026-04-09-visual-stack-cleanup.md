# 2026-04-09 Visual Stack Cleanup

## Goal

Refactor the aligned native-vs-FFS visualization stack into a cleaner architecture with clearer module boundaries, typed internal contracts, explicit artifact writing helpers, and lighter workflow modules, while preserving the current repo scope and existing user-facing CLI commands where practical.

## Non-Goals

- Add a new visualization product or change repo scope.
- Expand beyond preview / calibrate / record / align / aligned-case comparison visualization.
- Replace existing workflows from scratch.
- Redesign current comparison semantics unless required by the refactor.
- Require GPU, RealSense hardware, or FFS weights for software-only validation.

## Current Visual Stack Inventory

Current inventory is recorded in:

- `docs/generated/visual_stack_inventory.md`

Key implementation concentration points today:

- `data_process/visualization/turntable_compare.py`
- `data_process/visualization/pointcloud_compare.py`
- `data_process/visualization/object_compare.py`
- `data_process/visualization/object_roi.py`

## Current Pain Points

1. Workflow orchestration, geometry math, ROI logic, rendering, layout composition, and artifact writing are mixed in a few large modules.
2. The largest visualization modules have become difficult to navigate and prompt against.
3. Internal data is still passed through many loosely structured dicts across workflow boundaries.
4. Artifact writing is scattered instead of centralized.
5. Import layering is implicit rather than guarded.
6. Docs describe the user workflows, but not the actual internal visualization package map well enough.

## Target Architecture

Refactor toward these responsibility boundaries:

- `types.py`
  - shared typed contracts for scene bundles, camera clouds, crops, orbit plans, render output specs, and debug artifact plans
- `io_case.py`
  - aligned-case metadata/depth/cloud loading
- `io_artifacts.py`
  - video/gif/json/image/ply writing
- `layouts.py`
  - board composition, labels, keyframe sheets
- `views.py`
  - orbit planning and supported-coverage math
- `renderers/*`
  - rendering-only backends and shared rasterization helpers
- `workflows/*`
  - orchestration-only entrypoints where practical

Compatibility policy:

- Keep existing public module entrypoints working through re-exports or wrapper functions.
- Keep existing harness commands stable where practical.
- Keep current artifact names stable unless explicitly documented otherwise.

## Migration Strategy

1. Inventory current modules, workflows, and import dependencies.
2. Add required docs artifacts before code movement.
3. Extract stable low-level modules first:
   - typed contracts
   - artifact IO
   - layouts
   - view/orbit helpers
   - case IO
4. Rewire large modules to import from these shared helpers.
5. Keep old module APIs as compatibility shims during the transition.
6. Add architectural guard checks and architecture-level tests.
7. Update docs to reflect the new package map.

## Compatibility Strategy

- Preserve:
  - `scripts/harness/visual_compare_depth_panels.py`
  - `scripts/harness/visual_compare_reprojection.py`
  - `scripts/harness/visual_compare_depth_video.py`
  - `scripts/harness/visual_compare_turntable.py`
- Preserve current output naming for comparison artifacts.
- Preserve current test coverage for comparison workflows.
- Keep legacy import paths working from `data_process.visualization.pointcloud_compare` and `data_process.visualization.turntable_compare`.

## Validation Plan

Run:

- `python scripts/harness/check_visual_architecture.py`
- `python scripts/harness/check_all.py`

Add/update software-only tests for:

- import layering
- typed contracts
- turntable workflow orchestration
- merge diagnostics workflow orchestration
- artifact writing
- layout building

Record outcomes in:

- `docs/generated/visual_stack_refactor_validation.md`

## Risks

1. Breaking existing tests that import helpers directly from legacy modules.
2. Creating import cycles while moving helpers out of god modules.
3. Accidentally changing output paths or filenames used by current workflows.
4. Over-modeling types and making the code harder to evolve.

## Acceptance Criteria

1. The visualization stack is split into cleaner modules by responsibility.
2. CLI wrappers remain thin.
3. Major workflow modules are orchestration-first rather than utility sinks.
4. Shared internal contracts are explicit and typed.
5. Import layering is improved and checked.
6. Docs accurately reflect the current visualization architecture.
7. Existing behavior remains compatible where practical.
8. Deterministic checks pass.

## Completion Checklist

- [ ] Inventory current visualization stack
- [ ] Add `docs/generated/visual_stack_inventory.md`
- [ ] Add `data_process/visualization/types.py`
- [ ] Add lower-level shared modules for IO / layouts / views / artifact writing
- [ ] Rewire large visualization modules to use shared helpers
- [ ] Add compatibility shims or re-exports
- [ ] Add `scripts/harness/check_visual_architecture.py`
- [ ] Add architecture-level tests
- [ ] Update `AGENTS.md`
- [ ] Update `docs/ARCHITECTURE.md`
- [ ] Update `docs/WORKFLOWS.md`
- [ ] Add `docs/generated/visual_stack_refactor_validation.md`
- [ ] Run deterministic validation
