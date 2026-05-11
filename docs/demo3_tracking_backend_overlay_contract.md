# Demo 3 Tracking Backend Benchmark + 3D Anchor Overlay Contract

Demo 3 tracking is no longer only "CoTracker visualization". It is a multi-backend tracking benchmark plus 3D PCD temporal-anchor overlay.

## Backend Contract

CoTracker3 online is the first baseline backend because it matches FuturePhysTwin's PhysTwin-style dense tracking pipeline. Other backend names are reserved for NVOFA, TAPNext, LocoTrack, TAPIR, and VPI LK, but missing optional dependencies must report `available=false` without failing deterministic checks.

All backends must eventually output a PhysTwin-compatible tracking artifact:

```text
tracks:     (T, N, 2), coordinate order y,x
visibility: (T, N) or (T, N, 1)
```

The y,x convention is mandatory because downstream mask/depth indexing uses row,column order.

## Overlay Contract

Demo overlay displays sparse lifted 3D anchors/trails, with 50-200 points by default. Dense 5000/10000-point tracks are benchmark artifacts and must not be shown in the realtime demo by default.

Tracks are lifted to 3D using:

- the same depth source as the displayed PCD
- the same object/controller mask
- the same intrinsics
- the same `calibrate.pkl` camera-to-world transforms

Existing PCD board/rendering convention is preserved: columns are camera viewpoints over the same fused object PCD, not three independent per-camera point clouds.

## FPS Contract

Tracking overlay must not contaminate main PCD FPS unless explicitly enabled. Log `tracking_model_ms`, `tracking_e2e_ms`, and `overlay_ms` separately from the formal PCD render metrics.
