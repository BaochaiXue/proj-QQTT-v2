# Demo 3.2 Open3D Multi-Viewport Panel

## Goal

Replace the current OpenCV `--render-mode panel` window with a Qt/Open3D
multi-viewport window while preserving the existing fake-live strict same-seq
pipeline and optional 2D diagnostic MP4 output.

## Implementation Plan

1. Add tests that define the new panel contract:
   - Demo 3.2 panel metadata exposes `panel_backend=open3d_multi_viewport`.
   - The middle viewport receives filtered PCD layers only.
   - The right viewport receives the same filtered PCD layers plus tracker marker
     layers.
   - Existing panel HUD and delegate argv tests continue to pass.
2. Refactor runtime panel helpers:
   - Keep the current 2D panel frame renderer for `--panel-video-output`.
   - Add small layer-plan helpers that describe which geometry classes are shown
     in each Open3D panel viewport.
3. Replace `_run_panel_viewer()` with an Open3D GUI runner:
   - Left column: `gui.ImageWidget` showing latest RGB.
   - Middle column: independent `gui.SceneWidget` for filtered PCD.
   - Right column: independent `gui.SceneWidget` for filtered PCD + query
     markers.
   - Shared overlay HUD label with seq, timing, startup hold, filter preset, and
     marker/point counts.
4. Validation:
   - Run targeted unittest modules:
     `tests.test_single_demo_v3_runtime tests.test_single_demo_tapnextpp_overlay`.
   - If targeted tests pass, run smoke validation when practical.
   - Manually launch Demo 3.2 fake-live panel to confirm both Open3D viewports
     can be dragged independently.

## Notes

- Do not change headless capture/render behavior.
- Do not change query marker gating/filtering semantics.
- The saved MP4 remains the existing 2D side-by-side diagnostic view and does
  not record interactive camera rotations.
