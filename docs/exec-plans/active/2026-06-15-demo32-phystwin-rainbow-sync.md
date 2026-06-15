# Demo 3.2 PhysTwin Rainbow Same-Seq Visualization

## Goal

Add Demo 3.2 visualization modes for filtered PCD inspection and strict same-seq
PhysTwin-style rainbow query tracking. Tracking overlays must use stable per-query
identity colors and must not combine PCD and tracker packets from different seqs.

## Planned Changes

- Add `--demo-visual-mode {pcd,tracking}` to the single Demo 3.x runtime.
- For Demo 3.2/3.3 fake-live visual modes, force sync enhanced-pt filtered RGB
  PCD. In tracking mode, keep TAPNext++ enabled and display all visible lifted
  query points; in pcd mode, disable tracker overlay while keeping masks/PCD.
- Generate deterministic rainbow colors from query ids, store them on tracker
  packets, and use them in live and headless/offline render.
- Keep live tracking render on strict `PairedRenderPacket` same-seq updates.
- Make the offline headless renderer use exact same-seq query payloads only.

## Validation

- Focused unit tests for runtime contracts, tracker colors, headless writer, and
  offline renderer exact-seq behavior.
- Run `scripts/harness/check_all.py`; run `--full` if time and environment allow.
