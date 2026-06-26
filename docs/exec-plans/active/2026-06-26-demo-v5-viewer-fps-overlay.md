# Demo v5 Viewer FPS Overlay

## Goal

Show a realtime measured FPS readout in one Demo v5 realtime viewer window.

## Scope

- Add a small rolling FPS helper in `demo_v5/visualize_track.py`.
- Draw the FPS readout on the left RGB input OpenCV window in side-by-side mode.
- Keep the right `final_data` Open3D window interactive and unchanged.
- Do not change chunk generation, tracker state, quality gates, payload schema, or realtime PhysTwin behavior.

## Validation

- Add unit coverage for FPS calculation and RGB overlay.
- Run Demo v5 focused tests.
- Run the smoke validation profile.

## Result

- Added the FPS HUD to the left RGB input window in side-by-side realtime mode.
- Kept the right Open3D `final_data` window interactive.
- Validation:
  - `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_demo_v5_realtime_phystwin.py -q`
  - `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
