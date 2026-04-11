## Goal

Add one professor-facing 3D point-cloud registration board that compares `Native`, `FFS-current`, and `FFS-swapped` using fixed canonical views and source-camera coloring only.

## Non-Goals

- No new broad visualization package
- No 2D depth panels as the main result
- No hero/truth/support/mismatch dashboard
- No calibrate.pkl semantic changes
- No many-artifact default output

## Files To Modify

- `data_process/visualization/stereo_audit.py`
- `data_process/visualization/layouts.py`
- `scripts/harness/visual_compare_stereo_order_pcd.py`
- `docs/WORKFLOWS.md`
- `docs/generated/depth_visualization_validation.md`
- tests related to layout / workflow / debug gating

## Approach

1. Reuse current object-first scene/crop logic to get one shared object ROI and crop.
2. Reuse current source-attribution point-cloud rendering with fixed colors:
   - Cam0 red
   - Cam1 green
   - Cam2 blue
3. Run FFS swapped inference on the same aligned IR pairs and build swapped world-space camera clouds in the same frame.
4. Render a single `3 x 4` board:
   - rows = `Native`, `FFS-current`, `FFS-swapped`
   - columns = `Oblique`, `Top`, `Front`, `Side`
5. Keep optional closeup/debug outputs behind explicit flags only.

## Validation Plan

- Add software-only tests for:
  - board layout generation
  - shared fixed views across rows
  - legend rendering
  - current vs swapped row generation
  - optional closeup gating
- Run `python scripts/harness/check_all.py`

## Acceptance Criteria

- Default output writes only:
  - `01_stereo_order_registration_board.png`
  - `match_board_summary.json`
- Board is `3 x 4` with shared frame / ROI / crop / views
- Rows are `Native / FFS-current / FFS-swapped`
- Columns are `Oblique / Top / Front / Side`
- Main evidence is 3D point-cloud-only source-color registration quality
