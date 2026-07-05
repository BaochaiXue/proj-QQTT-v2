# Demo v6 Live ASAP Augmentation

## Requirement

Problem:
Downstream provided an offline full-recording postprocessor
(`july2_chunk_vis.py::write_asap_online_chunks`) that ARAP-deforms the
aligned shape-prior mesh to the tracked object points and re-emits chunks
with estimated positions for invalid points plus deformed surface/interior
points. Demo v6 must do this live, at chunk materialization, and publish
the result inside `object_points` directly.

Required final behavior: see `demo_v6/design_spec_v6.md` (authoritative).
Key points: `object_points = [filled object points, deformed surface
points, deformed interior points]`; estimates only where
`visibilities & motions_valid & finite & nonzero` fails; estimated entries
keep the original mask values (False) so downstream losses never consume
estimates; synthetic query ids at offset bases 1e9 (surface) / 2e9
(interior); default colors cyan/orange; `final_mesh.glb` missing -> fail
fast; the downstream-provided previous-frame-vertices fallback is kept with
a revisit comment; static `surface_points`/`interior_points` unchanged.

State changes:
- New: `demo_v6/asap.py`, `demo_v6/design_spec_v6.md`,
  `tests/test_demo_v6_asap.py`.
- `demo_v6/chunk_data_stream.py`: session-lived `AsapRuntime` next to the
  tracking runtime; augmentation runs in `_write_chunk_from_rows` after the
  window builds and before payload assembly; manifest gains `asap_*`
  telemetry.
- `demo_v6/main.py`: `--asap-augment/--no-asap-augment` (default on) and
  `--asap-mesh-path` threaded into both entry points.

Invalid cases:
- Missing/empty mesh or missing `shape_prior_case_dir` metadata when ASAP
  is enabled -> `AsapMeshError` at first materialized window.

Constraints:
- Keep the offline postprocessor's math and defaults; only the execution
  model changes (per-window incremental, precomputed embeddings, batched
  rigid fits).
- Borrow-frame (lookahead) slicing happens before ASAP; borrow data never
  reaches ASAP state.

Unknowns:
- Real-hardware per-chunk ARAP wall time (telemetered via `asap_ms`).

## Plan

- [x] Port + vectorize the ASAP core into `demo_v6/asap.py`.
- [x] Integrate into `demo_v6/chunk_data_stream.py` and `demo_v6/main.py`.
- [x] Write `demo_v6/design_spec_v6.md`.
- [ ] Add `tests/test_demo_v6_asap.py`.
- [ ] Full test suite + smoke validation; adversarial review; commit/push.

## Validation

- `python -m pytest tests/test_demo_v6_asap.py -q`
- `python -m pytest tests/ -q`
- `python scripts/harness/validation/run.py --profile smoke`
