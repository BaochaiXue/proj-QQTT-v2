# Demo 2 Remote FFS Payload And Quality Refresh

Date: 2026-05-05

## Goal

Keep the existing full-frame `ffs_remote depth_u16` path as a measured baseline, but stop treating it as the realtime default when the network only sustains about 6 FPS. Add payload-reduction knobs and a true remote sparse FFS main path. Keep the native RealSense + remote FFS refresh mode only as a debug/fallback comparison path.

## Scope

- Extend the remote FFS protocol with optional mask payloads, sparse response modes, and payload compression.
- Add client/server CLI flags for compression and sparse return benchmarking.
- Add Demo 2 flags for native RealSense main depth plus asynchronous remote FFS quality refresh metrics, explicitly labeled fallback/debug.
- Make `ffs_remote masked_uv_depth|masked_xyz` a real FFS-derived main PCD path that sends same-frame EdgeTAM masks to the remote server.
- Add strict remote FFS engine contract metadata and validation for the Demo 2 quality artifact.
- Update deterministic smoke tests and the generated remote FFS validation note.

## Non-Goals

- Do not make native RealSense depth the official Demo 2 output.
- Do not change formal recording or alignment code.
