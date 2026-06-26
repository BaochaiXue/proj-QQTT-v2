# Demo v5 Fixed Anchor Backup Bundle Recovery

## Goal

Keep Demo v5 controller anchor topology fixed after the first chunk while
recovering per-frame measurements from first-chunk local TAPNext++ backup query
bundles. Invalid controller quality must be written for diagnostics but must not
be published into the realtime online stream.

## Scope

- Demo v5 realtime/FuturePhysTwin chunk preprocessing only.
- No TAPNext++ recurrent-state mutation, second tracker, or per-chunk anchor
  reselection.
- No formal recording/alignment product changes.

## Implementation Notes

- Split controller observation validity into raw visibility, processed mask,
  depth, measurement, and motion flags in the strict product path.
- Bind each fixed controller anchor to a local first-frame backup bundle from
  existing controller candidates, defaulting to K=12, radius 0.03 m, and
  minimum backup count 4.
- Replace static hold fallback with per-frame primary-direct, bundle rigid or
  translation recovery, or low-confidence unrecoverable prediction.
- Persist per-frame anchor source, mode, confidence, failure reason, support
  count, recovery residual, bundle ids, and chunk quality status.
- Write normal/degraded chunks to the online stream. Write invalid chunks as
  standalone diagnostic cases, skip online publication, and return non-zero
  from the realtime runner.

## Validation Plan

- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_phystwin_strict_product`
- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_realtime_phystwin`
- `conda run -n demo_2_max --no-capture-output python -m py_compile qqtt/demo/phystwin_strict_product.py demo_v5/futurephystwin_chunk_writer.py demo_v5/headless_chunk_bridge.py demo_v5/online_chunk_output.py demo_v5/online_case_aggregate.py demo_v5/realtime_futurephystwin_chunks.py`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`

## Status

- Implemented.
- Strict product and Demo v5 focused unit tests pass locally.
- Adjacent Demo v4/Demo v5/strict product unit sweep passes locally.
- Smoke validation passes locally.
- Full source-headless chunk test on
  `result/demo_v5/full_chunks_20260626_010038/...camera_capture_20260626_010054`
  exposed the expected invalid-quality stop at chunk 13 after 12 online-published
  chunks. A diagnostic chunk video was written for chunks 1-13 at 5 FPS.
