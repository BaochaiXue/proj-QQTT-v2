# Demo v5.1 Borrow-Frame Motion Validity

## Requirement

Problem:
Origin motion validity is a forward test (`motions_valid[t]` needs frame
`t+1`), so an online chunk could not evaluate its tail row at publish time.
The interim rule published the tail row as all-True, and before that it was
all-False (14 structurally blank frames per 490-frame session in the
`visibilities & motions_valid` render mask).

Decision (user, 2026-07-04): materialize each chunk only after the next
window's first row (the "borrow" frame) arrives, so the tail row's motion
validity is computed for real — for both object points and controller
anchors (including chunk-0 selection). This trades a fixed +1 output frame
(0.2 s at 5 FPS) of publish latency; steady-state remains one chunk per 7 s.
"Publish at the instant the window closes" is explicitly abandoned.

Required final behavior:
- A chunk window `[start, end)` materializes with one lookahead row `end`
  when available. Motion consistency (object and controller, including the
  chunk-0 once-fail selection filter) runs over the window plus the borrow
  row; every published row, including the tail, carries a real motion
  verdict.
- The borrow row's own values are never published with the chunk; it is
  re-processed as the next window's first row. Borrow data must not leak
  into `source_frame_indices`, manifests, or published arrays.
- Terminal flush: at capture end (or an exact-multiple offline conversion)
  a full window without a successor row materializes without lookahead; its
  tail row publishes origin's end-of-sequence semantics
  (`motions_valid = False`). The final window is never dropped.
- Prepared-path nonterminal chunks require the borrow row's prepared frame.
  A successor row with missing prepared payload is a data-contract failure,
  not terminal flush and not a silent fallback.
- The all-True interim rule and the seam-carry machinery are removed: the
  chunk-boundary motion (prev last -> next first) is now tested inside the
  publishing chunk via the borrow row, which is exactly origin's indexing
  (the earlier frame carries the verdict). A boundary jump therefore proxies
  the publishing chunk's tail row instead of the next chunk's frame 0.
- Manifests record `motion_lookahead_frames` (1 or 0).

State changes:
- `demo_v5_1/tracking.py`: `process_window(..., lookahead_frames=)`;
  computation domain = extended window, published outputs sliced; recovery
  loop over published rows only; carry state and all-True rule deleted.
- `demo_v5_1/chunk_data_stream.py`: window builders accept lookahead
  frames/rows; offline conversion looks ahead in the row list; the live
  tail loop holds a full window pending until the borrow row arrives and
  flushes it without lookahead at capture end; backlog accounting counts a
  window complete only when its borrow row exists; prepared-path publishing
  fails fast if a nonterminal borrow row lacks its prepared frame.
- `tests/test_demo_v5_1_tracking.py`: tail-row semantics tests replace the
  all-True test; the seam-carry test becomes a borrow-frame boundary test.

Invalid cases:
- `lookahead_frames` must be >= 0 and < the extended frame count.
- Prepared-path nonterminal borrow rows must have prepared frames; missing
  borrow prepared data raises before any chunk is published.

Constraints:
- Steady-state throughput unchanged (one chunk per chunk_seconds); the
  borrow row is single extra latency, not extra work (it is re-used as the
  next window's first row).
- Frozen identity, query schema, and recovery semantics unchanged.

Unknowns:
- None blocking.

## Plan

- [ ] `tracking.py`: lookahead mode, slice-published outputs, remove carry
      and all-True rule.
- [ ] `chunk_data_stream.py`: builder lookahead params, offline list
      lookahead, live pending-window + terminal flush, backlog accounting,
      manifest field.
- [ ] Tests: tail-row real verdict (lookahead), terminal False, boundary
      jump proxies publishing chunk's tail, borrow exclusion from published
      axes, and prepared borrow-frame fail-fast; update chunk-data tests if
      pinned.
- [ ] Full test suite + smoke validation; adversarial review; commit/push.

## Validation

- 2026-07-04 fail-fast follow-up:
  - `conda run -n demo_2_max --no-capture-output python -m py_compile demo_v5_1/chunk_data_stream.py`
    passed.
  - `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo_v5_1_chunk_data tests.test_demo_v5_1_tracking`
    passed, 44 tests.
  - `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
    passed, 111 tests.
