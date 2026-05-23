# Demo31 Demo32 Frame Bundle Exact Target

## Goal

Make Demo 3.1 and Demo 3.2 render only exact target frame bundles by default.
Tracker result, depth/lift inputs, surface anchors, and rendered PCD must share
the same target `group_id`. Mask reuse remains available in the default
`exact-target` mode, but provenance must record when the mask source group is
older than the target group.

## Scope

- Add explicit frame bundle policy values: `exact-target`, `strict-source`, and
  `latest-reuse-debug`.
- Replace default nearest render-packet fallback with exact target matching.
- Keep nearest fallback only behind an explicit debug policy.
- Protect bundle/cache entries while tracker results are outstanding.
- Emit provenance/profile fields for same-target and strict-source ratios.
- Preserve GPU0/GPU1 split and CPU NumPy IPC contract.
- Keep tracker child payload small; do not send full RGB-D/PCD to the child.

## Out Of Scope

- Tracker backend algorithm changes.
- CUDA tensor IPC or P2P transport.
- FFS or EdgeTAM inference changes.
- Removing latest-reuse mask mode entirely.

## Validation

- Unit tests for default exact-target policy and debug nearest fallback.
- Unit tests for protected cache eviction.
- Dry-run contract checks for Demo 3.1 and Demo 3.2.
- `python scripts/harness/check_all.py` before completion.

## Progress

- Added `--frame-bundle-policy` with `exact-target`, `strict-source`, and
  `latest-reuse-debug`.
- Changed the default render packet match policy to `exact-target-bundle`.
- Kept nearest pending render/PCD fallback only under
  `--tracking-render-packet-match-policy exact-then-nearest-debug`.
- Added protected bundle pruning for pending render packets, pending fusion
  bundles, lift inputs, and surface anchors.
- Added per-frame provenance/profile fields for same-target and strict-source
  accounting.
- Updated Demo 3.1 contract tests for default exact target, debug nearest
  fallback, strict-source rejection, and protected cache pruning.
