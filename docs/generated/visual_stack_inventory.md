# Visual Stack Inventory

## Purpose

This inventory captures the current aligned-case visualization stack before the 2026-04-09 cleanup refactor. It documents what currently exists, where responsibilities are mixed, and how the refactor maps old modules into clearer boundaries.

## CLI Entrypoints Under `scripts/harness/`

### `visual_compare_depth_panels.py`

- Purpose: per-camera native-vs-FFS diagnostic depth panels
- Current role: thin CLI wrapper
- Calls: `data_process.visualization.panel_compare.run_depth_panel_workflow`

### `visual_compare_reprojection.py`

- Purpose: native-vs-FFS cross-view reprojection diagnostics
- Current role: thin CLI wrapper
- Calls: `data_process.visualization.reprojection_compare.run_reprojection_compare_workflow`

### `visual_compare_depth_video.py`

- Purpose: older temporal fused native-vs-FFS point-cloud compare
- Current role: thin CLI wrapper plus preset handling
- Calls: `data_process.visualization.pointcloud_compare.run_depth_comparison_workflow`

### `visual_compare_turntable.py`

- Purpose: professor-facing single-frame object-first turntable compare
- Current role: thin CLI wrapper
- Calls: `data_process.visualization.turntable_compare.run_turntable_compare_workflow`

## Visualization Modules Under `data_process/visualization/`

### `calibration_io.py`

- Category: IO
- Responsibility: read `calibrate.pkl` and apply serial-order mapping

### `camera_frusta.py`

- Category: geometry / labels
- Responsibility: build camera pose / frustum geometry from `c2w`

### `depth_colormap.py`

- Category: rendering support
- Responsibility: shared depth visualization colormap helpers

### `depth_diagnostics.py`

- Category: workflow support / rendering / panel diagnostics
- Responsibility: per-camera depth maps, crops, residual overlays, ROI parsing

### `panel_compare.py`

- Category: workflow orchestration
- Responsibility: per-camera panel compare workflow

### `reprojection_compare.py`

- Category: workflow orchestration
- Responsibility: cross-view reprojection workflow

### `pointcloud_compare.py`

- Category: mixed god module
- Current responsibilities:
  - aligned-case IO
  - depth decoding
  - per-camera cloud generation
  - scene crop estimation
  - view config computation
  - projection math
  - rasterization
  - fallback rendering
  - panel/grid layout
  - video/gif writing
  - older temporal workflow orchestration

### `object_roi.py`

- Category: ROI / geometry
- Responsibility:
  - tabletop plane fitting
  - object-above-table extraction
  - component scoring / `graph_union`
  - object ROI bounds

### `object_compare.py`

- Category: masking / provenance / debug writing
- Responsibility:
  - object-first layer construction
  - per-camera refinement masks
  - pixel-mask filtering
  - world ROI projection into camera bboxes
  - object/cloud debug artifacts
  - compare debug metrics

### `support_compare.py`

- Category: support diagnostics
- Responsibility:
  - per-pixel support-count map computation
  - support rendering / summarization

### `source_compare.py`

- Category: provenance / residual diagnostics
- Responsibility:
  - source-camera coloring
  - source-attribution overlay
  - source-split renders
  - mismatch residual rendering
  - legend generation

### `turntable_compare.py`

- Category: mixed god module
- Current responsibilities:
  - single-frame case selection
  - object ROI refinement orchestration
  - orbit planning
  - supported coverage estimation
  - overview rendering
  - board composition
  - output planning
  - artifact writing
  - professor-facing turntable workflow orchestration

## Current Major Workflows

### Depth Panels

- Entry: `visual_compare_depth_panels.py`
- Engine: `panel_compare.py`, `depth_diagnostics.py`

### Reprojection Compare

- Entry: `visual_compare_reprojection.py`
- Engine: `reprojection_compare.py`, `depth_diagnostics.py`

### Depth Video Compare

- Entry: `visual_compare_depth_video.py`
- Engine: `pointcloud_compare.py`

### Turntable Compare

- Entry: `visual_compare_turntable.py`
- Engine: `turntable_compare.py`

### Object-First Compare

- Used inside turntable compare
- Engine: `object_compare.py`, `object_roi.py`

### Source / Support / Mismatch Compare

- Used inside turntable compare
- Engine:
  - `support_compare.py`
  - `source_compare.py`

## Current Responsibility Buckets

### IO

- `calibration_io.py`
- `pointcloud_compare.py` (case IO currently mixed in)

### Geometry / Math

- `camera_frusta.py`
- `object_roi.py`
- `pointcloud_compare.py` (projection/view math currently mixed in)
- `turntable_compare.py` (orbit math currently mixed in)

### ROI / Masking

- `object_roi.py`
- `object_compare.py`
- `pointcloud_compare.py` (scene crop estimation currently mixed in)

### Rendering

- `depth_diagnostics.py`
- `pointcloud_compare.py`
- `support_compare.py`
- `source_compare.py`

### Layout / Composition

- `pointcloud_compare.py`
- `turntable_compare.py`

### Artifact Writing

- `object_compare.py`
- `pointcloud_compare.py`
- `turntable_compare.py`

### Workflow Orchestration

- `panel_compare.py`
- `reprojection_compare.py`
- `pointcloud_compare.py`
- `turntable_compare.py`

## Main Pain Points

1. `pointcloud_compare.py` combines case IO, rendering, layout, and workflow logic.
2. `turntable_compare.py` combines orbit planning, overview layout, artifact planning, and workflow logic.
3. Artifact writing is scattered across multiple modules.
4. Stable internal contracts are mostly implicit dict shapes.
5. Import layering is not explicitly guarded.

## Refactor Mapping

### Old -> New Responsibility Mapping

- `pointcloud_compare.py`
  - split toward:
    - `io_case.py`
    - `renderers/*`
    - `layouts.py`
    - `io_artifacts.py`
    - compatibility wrapper in `pointcloud_compare.py`

- `turntable_compare.py`
  - split toward:
    - `views.py`
    - `layouts.py`
    - `io_artifacts.py`
    - typed contracts in `types.py`
    - compatibility wrapper / orchestration-first workflow

- `object_compare.py`
  - keep core object-first logic
  - reduce direct artifact-writing coupling where practical

## Expected Post-Refactor Boundaries

- case IO separated from rendering
- geometry separated from artifact writing
- layouts separated from rendering math
- workflow modules mostly orchestration
- shared contracts typed and explicit
