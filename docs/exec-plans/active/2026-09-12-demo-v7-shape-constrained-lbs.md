# Demo v7 Variant F: shape-constrained mesh LBS

## Goal

The A-E replay comparison on the 606-frame manipulation capture ruled out
smaller skinning neighbourhoods and aggressive held-bone healing as the main
jelly lever:

| Variant | obs2mesh p50 | silhouette IoU | edges >1.5x | jelly |
| --- | ---: | ---: | ---: | ---: |
| A baseline | 0.377 cm | 0.7731 | 4.11% | 2.889 mm |
| B heal after one missing frame | 0.377 cm | 0.7755 | 3.69% | 2.989 mm |
| C K=8 | 0.378 cm | 0.7687 | 6.10% | 3.036 mm |
| D heal@1 + K=8 | 0.379 cm | 0.7733 | 5.38% | 3.124 mm |
| E heal@1 + rogue=2 cm | 0.378 cm | 0.7752 | 3.68% | 2.969 mm |

Variant F keeps the proven baseline unchanged (`skin K=16`, normal held/stale
policy, normal rogue threshold) and adds one mechanism only: a mesh-topology
shape projection after LBS and before barycentric splat replay.

## Design

For every rest triangle, precompute its centroid-relative corner offsets, a
right-handed tangent frame, and doubled area. On each FORMAL frame:

1. Run the existing rest-anchored LBS exactly as before.
2. For each current triangle, retain its current centroid and orientation but
   construct the rigid copy of its rest triangle in that frame.
3. Area-weight and average all incident-face proposals at each shared vertex.
4. Blend the proposal against the unmodified LBS result.
5. Repeat twice and cap every vertex's total departure from LBS at 8 mm.

Fixed F parameters:

- iterations: 2
- shape strength: 0.5
- maximum per-vertex correction: 8 mm

This is a spatial regularizer, not temporal smoothing. Global rigid motion is
an exact fixed point. An articulated hinge is also a fixed point because each
face may rotate independently; the constraint suppresses only local
stretch/shear. Every frame still solves from rest, so no projected state is fed
into the next frame.

## Observable contract

`gaussian_live_stats.json` reports `shape_constrained=true`, the fixed
parameters, and the last-frame correction p50, p95, and maximum in millimetres.
This proves the production renderer applied F and bounds its departure from the
tracked LBS solution.

## Validation

Deterministic CPU tests cover rigid-motion invariance, articulated-hinge
preservation, high-strain-tail reduction, the hard correction cap,
history-free determinism, and actual integration in
`MeshAnchoredGaussianRenderer._pose_to`.

Focused local result: 10 tests passed (the six new F tests plus the existing
mesh-anchored renderer tests). On a 10,242-vertex / 20,480-face icosphere, two
CPU projection iterations measured 9.6 ms. Rigid-motion numerical error was
1.83e-8 m.

The final acceptance gate is the same A/F replay table on the real 606-frame
capture. F is accepted only if it materially lowers jelly and edge strain
without a meaningful obs2mesh or silhouette regression. No K or hygiene
changes are mixed into that comparison.

## CI repair found during validation

The clean GitHub runner could not collect the demo_v7 suite because
`runtime/utils/ffs_align.py` imports `numba` while the workflow did not install
it. Add `numba` to the CPU dependency list so the advertised suite actually
runs on a hermetic runner.
