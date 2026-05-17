# Demo 3 Tracking Backend Benchmark + 3D Anchor Overlay Contract

Demo 3 tracking is no longer only "CoTracker visualization".
It is a multi-backend tracking benchmark plus 3D PCD temporal-anchor overlay.

## Backend Contract

CoTracker3 online is the first baseline backend because it matches
FuturePhysTwin's PhysTwin-style dense tracking pipeline.

For live Demo 3 use, `cotracker3_online` must run as an online stream, not as a
one-shot whole-video tracker. Frames enter a rolling buffer; the first publish
occurs when the buffer reaches the CoTracker3 online window (`16` frames), and
subsequent publishes occur every CoTracker online step (`8` new frames). Each
published result carries the chunk frame range plus `tracks_yx + visibility`.
The offline benchmark may still replay saved cases, but the
`cotracker3_online` replay path feeds those saved frames through `update(frame)`
one by one and uses the same online backend contract and output convention.

Other backend names are reserved for:

- NVOFA
- TAPNext / TAPNext++
- LocoTrack
- TAPIR / BootsTAPIR
- VPI LK

Missing optional dependencies must report `available=false` without failing
deterministic checks.

All backends must eventually output a PhysTwin-compatible tracking artifact:

```text
tracks:     (T, N, 2), coordinate order y,x
visibility: (T, N) or (T, N, 1)
```

The y,x convention is mandatory because downstream mask/depth indexing uses
row,column order.

## PhysTwin Dense Export Contract

The offline benchmark supports a PhysTwin-compatible dense CoTracker mode:

```bash
python scripts/harness/experiments/run_demo3_tracking_backend_benchmark.py \
  --case-root data/<case> \
  --query-mode phystwin_dense \
  --backends cotracker3_online
```

In this mode the harness:

- reads first-frame nested masks from `mask/{camera}/*/0.png` unless
  `--mask-dir` points to another equivalent root
- unions all first-frame masks per camera before query sampling
- samples up to 5000 query points per camera, matching PhysTwin/FuturePhysTwin's
  dense CoTracker convention: masks with at least 5000 pixels use 5000 random
  points, while smaller masks use all available mask pixels
- uses FuturePhysTwin-style torch `randperm` sampling with default seed `42`
  and a per-camera offset (`seed + camera_idx`)
- writes PhysTwin-style root artifacts under `cotracker/{camera}.npz` in
  addition to the Demo 3 benchmark output tree

This dense export path is for compatibility and offline diagnostics. It does
not change the sparse overlay defaults.

## Overlay Contract

Demo overlay displays sparse lifted 3D anchors/trails, with a default cap of
30 points per camera.

Dense up-to-5000-point tracks are benchmark artifacts and must not be shown in
the realtime demo by default.

Tracks are lifted to 3D using:

- the same depth source as the displayed PCD
- the same object/controller mask
- the same intrinsics
- the same `calibrate.pkl` camera-to-world transforms

Existing PCD board/rendering convention is preserved: columns are camera viewpoints over the same fused object PCD, not three independent per-camera point clouds.

## FPS Contract

Tracking overlay must not contaminate main PCD FPS unless explicitly enabled.

Log these values separately from formal PCD render metrics:

- `tracking_model_ms`
- `tracking_e2e_ms`
- `overlay_ms`
