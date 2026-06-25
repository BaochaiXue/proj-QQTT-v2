# Demo v5 Continuous Realtime FuturePhysTwin

Demo v5 is the single-camera realtime bridge from Demo v5 fake/live capture to
one continuous `realtime_phystwin` online optimization run. It is a demo
diagnostic carveout, not the formal recording/alignment data product.

The default flow is:

```text
Demo v5 fake/live camera on GPU0
  -> RGB-D, masks, TAPNext++ strict tracks
  -> SAM3D shape-prior warmup worker on GPU1
  -> Demo v5 online FuturePhysTwin chunks at 5 FPS
  -> release the managed SAM3D worker
  -> realtime_phystwin zero-order then first-order optimization on GPU1
```

The optimization is one continuous online case. Demo v5 does not optimize each
chunk as an independent case.

## Default Contract

Run from the repo root on the `single-camera` branch:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v5/realtime_futurephystwin_chunks.py
```

Defaults:

```text
input source:                 fake-live
logical FPS:                  5
chunk length:                 7 seconds = 35 frames
camera/final-data GPU:         physical GPU0
managed SAM3D warmup GPU:      physical GPU1
optimization GPU:              physical GPU1
shape-prior worker env:        phystwin-max
optimization process env:      current Python environment
output base:                  result/demo_v5/futurephystwin_chunks
case prefix:                  demo_v5
optimization scope:            single continuous online case
```

Demo v5 keeps paths portable. The command passed to `realtime_phystwin` uses
paths relative to its working directory:

```text
--base_path ../result/demo_v5/futurephystwin_chunks/data
--online_dir ../result/demo_v5/futurephystwin_chunks/online_data/demo_v5
--static_data_path ../result/demo_v5/futurephystwin_chunks/data/demo_v5/final_data.pkl
```

## Outputs

```text
result/demo_v5/futurephystwin_chunks/
  data/<case>/
    READY
    final_data.pkl
    track_process_data.pkl
    calibrate.pkl
    metadata.json
    split.json
    color/0/<frame>.png
    mask/processed_masks.pkl
    tracking/0.npz
    cotracker/0.npz

  online_data/<case>/
    manifest.json
    chunks/chunk_000000.pkl
    chunks/chunk_000001.pkl

  <case>_chunk_0001/
  <case>_chunk_0002/
  <case>_chunks_manifest.json
```

The aggregate `data/<case>/final_data.pkl` and every online chunk contain the
topology fields required by `realtime_phystwin`:

```text
query_ids
query_semantic_labels
object_sample_query_ids
controller_sample_query_ids
topology_version
topology_hash
```

The topology version remains `demo_v4_session_topology_v1` because
`realtime_phystwin` already validates that wire contract.

## Common Runs

Short contract check:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v5/realtime_futurephystwin_chunks.py --dry-run
```

Quick fake-live smoke with reduced optimizer work:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v5/realtime_futurephystwin_chunks.py \
  --futurephystwin-base-path result/demo_v5/smoke \
  --case-prefix demo_v5_smoke \
  --max-chunks 2 \
  --optimization-zero-iterations 1 \
  --optimization-iterations 1
```

Full fake-live quality run:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v5/realtime_futurephystwin_chunks.py \
  --futurephystwin-base-path result/demo_v5/full_fake_live \
  --case-prefix demo_v5_full_fake_live \
  --max-chunks 7
```

Live camera run:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v5/realtime_futurephystwin_chunks.py \
  --input-source live \
  --futurephystwin-base-path result/demo_v5/live \
  --case-prefix demo_v5_live
```

Convert an existing headless capture without optimization:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v5/realtime_futurephystwin_chunks.py \
  --source-headless-capture result/demo_v5/capture_dir \
  --futurephystwin-base-path result/demo_v5/rechunked \
  --case-prefix demo_v5_rechunked \
  --optimization-mode disabled
```

## Quality Policy

Demo v5 does not lower optimizer settings to hit 5 FPS. The 5 FPS requirement is
for camera/fake-camera to online final-data publication. Optimization runs as a
concurrent consumer and keeps the existing `realtime_phystwin` quality defaults:

```text
zero-order iterations: 10
batch size:            4
segment length:        chunk frame count, default 35
segment stride:        16
recent window count:   8
first-order stop:      not early-stopped by default
```

Use `--optimization-iterations` only for smoke tests. Full validation should
leave it unset so first-order optimization uses the configured
`realtime_phystwin` iteration budget.

## Tracking Continuity

Demo v5 keeps fixed streaming topology selectors for the whole online session.
Object and controller columns keep fixed query ids across chunks. If a controller or object
anchor is lost, Demo v5 holds the last finite point, marks the active query id
as `-1`, and clears visibility/motion-valid flags instead of replacing that
column with a new physical query. That preserves optimizer topology and avoids
quality loss from identity swaps.

KNN/LBS revive can be added later, but any revive path must keep the same fixed
sample id and must stay bounded enough to preserve the realtime publication
cadence.

## Validation

Deterministic checks:

```bash
conda run -n demo_2_max --no-capture-output \
  python -m unittest tests.test_demo_v5_realtime_phystwin

conda run -n demo_2_max --no-capture-output \
  python scripts/harness/validation/run.py --profile smoke
```

Required final acceptance is a real fake-live to optimization run. Compare its
optimization artifacts and visible result quality against the closest offline
FuturePhysTwin run on the same or equivalent case. The online result should not
show an obvious quality regression relative to offline preprocessing plus
zero-order plus first-order optimization.
