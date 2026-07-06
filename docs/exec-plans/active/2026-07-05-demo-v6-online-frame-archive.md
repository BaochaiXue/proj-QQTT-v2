# Demo v6 online_data Per-Frame RGB-D Archive

## Requirement

Problem:
`online_data/` only publishes chunk pickles; downstream also needs the raw
sensor products for exactly the frames that were processed and published, in
the offline recording-case layout (`color/0/{k}.png`, `depth/0/{k}.npy`,
`calibrate.pkl`, `metadata.json`), plus a frame mapping table
(`enhance_metadata.json`).

Required final behavior (user decision, 2026-07-05):
- Filenames use the continuous online frame index `0..N-1`; only chunk
  frames are archived (unlike `capture/input_rgb`), including the chunk-0
  frame-0 shape-prior/warmup anchor.
- Depth format identical for RealSense (effective direct copy) and FFS
  (generated): `(H, W)` uint16 millimeters, invalid = 0 — matches
  `data_process_origin/data_process_pcd.py`'s `np.load(...)/1000.0`.
- Color is the frame's original RGB, written BGR-on-disk for `cv2.imread`.
- `chunks/` + `manifest.json` unchanged.
- New run clears old `color/`, `depth/`, `metadata.json`, `calibrate.pkl`.
- A chunk frame without color/depth -> fail fast (should never happen).

State changes:
- `demo_v6/phystwin_strict_product.py`: `PreparedPhysTwinFrame.depth_mm_u16`
  (+ NPZ round-trip) and `depth_m_to_mm_u16`.
- New `demo_v6/online_frame_archive.py`: `OnlineFrameArchive` writes
  color/depth per published frame, `calibrate.pkl` (list of 4x4 c2w),
  `metadata.json` (intrinsics/WH/frame_num/serial_numbers), and
  `enhance_metadata.json` (frame mapping table); rewrites happen after frame
  files land, before the chunk commit.
- `demo_v6/chunk_data_stream.py`: both entry points construct the archive
  next to `ChunkDataWriter` and `_write_chunk_from_rows` archives before
  `commit_chunk_data`.

Invalid cases:
- Missing prepared frame / legacy NPZ without depth / online index
  discontinuity / legacy sidecar reprocess path -> `OnlineFrameArchiveError`.

Constraints:
- Do not modify the chunks contract or `manifest.json` schema (manifest only
  gains `online_frame_archive_*` telemetry fields).
- Depth conversion must be bit-exact for RealSense uint16 units at the
  standard 0.001 m/unit scale.

## Plan

- [x] Add `depth_mm_u16` to the prepared frame product + NPZ round-trip.
- [x] Write `demo_v6/online_frame_archive.py`.
- [x] Wire into `_write_chunk_from_rows` and both entry points.
- [x] Update `demo_v6/design_spec_v6.md`.
- [x] Add `tests/test_demo_v6_online_frame_archive.py` + register in the
  validation harness.
- [x] Full test suite + smoke validation + end-to-end run; adversarial
  review (6 confirmed findings fixed/documented: identity-c2w +
  fx/fy/cx/cy fallbacks, clear-order invariant, >65.535 m -> invalid 0,
  metadata publish after chunk commit, fsync'd frame writes, O(N^2)
  mapping rewrite documented); commit/push.

## Validation

- `python -m pytest tests/test_demo_v6_online_frame_archive.py -q`
- `python -m pytest tests/ -q`
- `python scripts/harness/validation/run.py --profile smoke`
- End-to-end fake-live run; verify `online_data/` contents against
  `data_process_origin`-style reads.
