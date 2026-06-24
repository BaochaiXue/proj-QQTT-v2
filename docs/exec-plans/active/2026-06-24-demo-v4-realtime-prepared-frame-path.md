# Demo v4 Realtime Prepared Frame Path

## Goal

Move Demo v4 realtime chunk materialization onto frame-level prepared artifacts
so a full window no longer re-runs RGB decode, depth backprojection, mask IO,
track IO, dense PCD generation, or radius filtering.

## Plan

- Add a `PreparedPhysTwinFrame` contract in the strict PhysTwin product layer.
- Have the realtime headless writer save one prepared artifact before appending
  each `frames.jsonl` row.
- Have Demo v4 chunk streaming prefer prepared artifacts and fall back to legacy
  row materialization for older captures.
- Keep the final FuturePhysTwin case schema unchanged.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_demo_v4_futurephystwin_chunks.py -q`
- `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_single_demo_tapnextpp_overlay.py -q`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`

## Result

- 2026-06-24: all validation commands above passed.
