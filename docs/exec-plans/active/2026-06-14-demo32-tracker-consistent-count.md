# Demo 3.2 Tracker Consistent Count HUD

## Problem

The Open3D HUD currently shows the current TAPNext++ marker count, but it does
not show how many original query points have remained trackable across frames.
For demo evaluation, the operator needs a live count of consistently tracked
points in the legend/HUD area.

## Plan

- Track a per-query persistent visibility mask in the TAPNext++ worker.
- Update that mask only with points that are currently visible, finite, and
  successfully lifted through the current depth/mask gate.
- Add `consistent_visible_count` to the tracker marker packet.
- Display `consistent=X/Y` on the tracker HUD line and include it in debug logs.
- Add focused tests covering the packet field and HUD text.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_single_demo_tapnextpp_overlay`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
