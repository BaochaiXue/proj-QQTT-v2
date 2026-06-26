# Demo v5 Side-by-Side Realtime Viewer

## Goal

Replace Demo v5's default output-only point viewer with a side-by-side realtime viewer:
left panel shows raw RGB input as soon as camera/fake-live starts, right panel
shows final_data-style output chunks once they are published.

## Design Notes

- Keep Demo v5/data_process_sam3d naming and existing `visualize_track.py`
  renderer helpers.
- Add a viewer layout switch so `output-only` remains available for compatibility.
- Start the side-by-side viewer immediately after the camera subprocess starts,
  not after the first online chunk.
- Force input RGB timeline images for side-by-side runs, even when prepared-only
  output is enabled, so warmup has visible RGB.
- Keep realtime PhysTwin optimization behavior separate from this diagnostic
  viewer.

## Implementation Tasks

1. [done] Add failing tests for input RGB timeline writing, viewer command construction,
   side-by-side blank-output rendering, and runner start policy.
2. [done] Extend the headless capture writer/runtime flags to write `input_rgb/*.png`
   while preserving prepared-only chunk behavior.
3. [done] Extend `demo_v5/visualize_track.py` with `side-by-side` layout, input timeline
   tailing, right-panel output chunk playback, and scrub controls.
4. [done] Update `realtime_data_process_sam3d.py` to default to side-by-side and launch
   it immediately for live/fake-live camera runs.
5. [done] Run focused tests, smoke validation, and command-surface sanity checks.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_demo_v5_realtime_phystwin.py` passed
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke` passed
- `conda run -n demo_2_max --no-capture-output python -m py_compile demo_v5/visualize_track.py demo_v5/realtime_data_process_sam3d.py qqtt/demo/realtime_masked_edgetam_pcd.py` passed
- `conda run -n demo_2_max --no-capture-output python demo_v5/visualize_track.py --help` passed
- `conda run -n demo_2_max --no-capture-output python demo_v5/realtime_dense_track.py --help | rg -n "write-input-rgb-timeline|headless-prepared-only|headless-capture-dir"` passed
- Python command sanity confirmed default layout `side-by-side`, right renderer `sam3d-final-data`,
  immediate viewer start policy, and camera command `--write-input-rgb-timeline`.
