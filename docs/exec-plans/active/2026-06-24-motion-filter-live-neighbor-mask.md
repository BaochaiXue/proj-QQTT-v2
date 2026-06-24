# Demo v4 Motion Filter Live Neighbor Mask

## Goal

Align Demo v4 strict motion filtering with FuturePhysTwin `data_process_sam3d`
semantics: when a query becomes invalid earlier in a frame, later queries in
that same frame must not count it as a valid neighbor.

## Plan

- Add a regression test for sequential neighbor invalidation inside one frame.
- Update the test reference implementation to filter neighbors using the live
  `motions_valid[frame_idx]` mask.
- Keep `cKDTree` acceleration, but query all points once per frame and filter
  each neighbor list against the live mask during the query loop.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_phystwin_strict_product.py -q`
- `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_demo_v4_futurephystwin_chunks.py -q`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
