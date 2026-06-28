# Demo v5 Online Points Viewer

Demo v5 is the single-camera realtime bridge from Demo v5 fake/live capture to
chunked `data_process_sam3d` final_data. The default demo now opens a lightweight viewer
for object points and controller points instead of starting
`realtime_phystwin` optimization. It is a demo diagnostic carveout, not the
formal recording/alignment data product.

The default flow is:

```text
Demo v5 fake/live camera on GPU0
  -> RGB-D, masks, TAPNext++ strict tracks
  -> SAM3D shape-prior warmup worker on GPU1
  -> Demo v5 online data_process_sam3d chunks at 5 FPS
  -> release the managed SAM3D worker
  -> online object/controller point viewer on GPU1
```

The viewer reads `online_data/<case>/chunks` chunk by chunk. It plays every
frame in a committed chunk at the original `--replay-fps` value, then waits for
the next chunk, so the default playback is the original 5 FPS stream.

## Install

Demo v5 uses two Python environments by default:

```text
demo_2_max    camera/fake-camera, EdgeTAM, TAPNext++, online final_data,
              online point viewer, optional realtime_phystwin optimization
phystwin-max  managed SAM3D shape-prior warmup worker
```

On the validated lab workstation these environments already exist. Check them
first:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v5/env/check_demo_v5_env.py --role main --require-cuda

conda run -n phystwin-max --no-capture-output \
  python demo_v5/env/check_demo_v5_env.py --role shape-prior --require-cuda
```

The shape-prior CUDA check also validates `nvcc` and runs a tiny
`gsplat.rasterization` smoke. A failure there means SAM3D's GS layout
post-optimization would fail or warn at runtime, so update the `phystwin-max`
CUDA toolkit before starting the managed worker.

For a new machine, use the install materials under `demo_v5/env/`:

```bash
bash demo_v5/env/install_demo_v5_env.sh create
```

Or run the environment files manually:

```bash
conda env create -f demo_v5/env/environment-demo-v5-main.yml
conda run -n demo_2_max --no-capture-output \
  python -m pip install -r demo_v5/env/requirements-demo-v5-main.txt

conda env create -f demo_v5/env/environment-demo-v5-shape-prior.yml
conda run -n phystwin-max --no-capture-output \
  python -m pip install -r demo_v5/env/requirements-demo-v5-shape-prior.txt
```

If the environments already exist, replace `conda env create` with
`conda env update -f ... --prune`. GPU PyTorch, PyTorch3D, Kaolin, and Warp are
CUDA-stack-sensitive; `demo_v5/env/validated-versions-20260625.txt` records the
versions from the machine that passed the full Demo v5 E2E.

Demo v5 also expects repo-local runtime assets:

```text
vendor/demo_runtime/EdgeTAM-hf
vendor/demo_runtime/tapnet
vendor/demo_runtime/checkpoints/tapnextpp/tapnextpp_ckpt.pt
vendor/demo_runtime/sam-3d-objects
vendor/demo_runtime/stable-diffusion-x4-upscaler
vendor/demo_runtime/FuturePhysTwin
realtime_phystwin/train_online_zero_then_first.py
table_calibrate.pkl
```

The environment checker verifies these paths without downloading weights.

## How To Run

Always run from the repo root on `single-camera`:

```bash
git switch single-camera
git pull --ff-only origin single-camera
```

Do a contract check before a live run:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v5/realtime_data_process_sam3d.py --dry-run
```

Run a short fake-live smoke:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v5/realtime_data_process_sam3d.py \
  --base-path result/demo_v5/smoke \
  --case-prefix demo_v5_smoke \
  --shape-prior-endpoint tcp://127.0.0.1:7107 \
  --max-chunks 2 \
  --capture-extra-seconds 80
```

Run with the live RealSense camera:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v5/realtime_data_process_sam3d.py \
  --input-source live \
  --base-path result/demo_v5/live \
  --case-prefix demo_v5_live
```

If you want to reconnect continuous `realtime_phystwin` optimization instead
of the viewer, disable the viewer and enable optimization explicitly:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v5/realtime_data_process_sam3d.py \
  --base-path result/demo_v5/full_fake_live \
  --case-prefix demo_v5_full_fake_live \
  --shape-prior-endpoint tcp://127.0.0.1:7108 \
  --max-chunks 5 \
  --capture-extra-seconds 120 \
  --point-viewer-mode disabled \
  --optimization-mode continuous \
  --optimization-zero-iterations 1 \
  --optimization-iterations 1 \
  --optimization-wait-timeout-s 900
```

Run a quality fake-live optimization validation:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v5/realtime_data_process_sam3d.py \
  --base-path result/demo_v5/full_fake_live \
  --case-prefix demo_v5_full_fake_live \
  --shape-prior-endpoint tcp://127.0.0.1:7108 \
  --max-chunks 5 \
  --capture-extra-seconds 120 \
  --point-viewer-mode disabled \
  --optimization-mode continuous \
  --optimization-zero-iterations 10 \
  --optimization-wait-timeout-s 3600
```

The default warmup dual-GPU routing is:

```text
GPU0: Demo v5 fake/live camera, masks, tracking, final_data, online chunks
GPU1: managed SAM3D warmup worker, then online point viewer
```

The managed SAM3D worker is intentionally stopped before the viewer starts so
GPU1 memory is available for the display process. If you enable optimization,
the same release step happens before `realtime_phystwin`.

## Default Contract

The minimal default command is:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v5/realtime_data_process_sam3d.py
```

Defaults:

```text
input source:                 fake-live
logical FPS:                  5
chunk length:                 7 seconds = 35 frames
camera/final-data GPU:         physical GPU0
managed SAM3D warmup GPU:      physical GPU1
point viewer GPU:              physical GPU1
shape-prior worker env:        phystwin-max
point viewer env:              demo_2_max
output base:                  result/demo_v5/data_process_sam3d_chunks
case prefix:                  demo_v5
viewer playback:               chunk by chunk at 5 FPS
optimization scope:            disabled by default
```

The viewer command reads the online and static case paths directly:

```text
--online-dir result/demo_v5/data_process_sam3d_chunks/online_data/demo_v5
--case-dir result/demo_v5/data_process_sam3d_chunks/data/demo_v5
--fps 5.0
--object-color-mode rainbow
```

When optimization is enabled, Demo v5 keeps paths portable. The command passed
to `realtime_phystwin` uses paths relative to its working directory:

```text
--base_path ../result/demo_v5/data_process_sam3d_chunks/data
--online_dir ../result/demo_v5/data_process_sam3d_chunks/online_data/demo_v5
--static_data_path ../result/demo_v5/data_process_sam3d_chunks/data/demo_v5/final_data.pkl
```

## Outputs

```text
result/demo_v5/data_process_sam3d_chunks/
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
query schema fields required by `realtime_phystwin`:

```text
query_ids
query_semantic_labels
object_sample_query_ids
controller_sample_query_ids
query_schema_version
query_schema_hash
```

The query schema version is `data_process_sam3d_realtime_query_schema_v1`.
Validators reject the legacy Demo v4 `demo_v4_session_topology_v1` version;
Demo v5.1 artifacts must use the data_process_sam3d realtime query schema version.

Every Demo v5 case metadata also carries:

```text
runtime_contract = data_process_sam3d_realtime_final_data_v1
```

That contract is enforced by the active writer/validator, not by a side helper:

```text
object_sample_query_ids     must reference query_semantic_labels == object
controller_sample_query_ids must reference query_semantic_labels == controller
query_schema_hash           is recomputed from query ids, semantic labels, and sample ids
chunk continuity            requires stable query_schema_hash and contiguous online frame ranges
```

This keeps Demo v5 final_data semantically aligned with
`data_process_sam3d`: first-frame semantic ownership, per-frame mask/depth
gating, SAM3D-style neighbor motion filtering, 5 mm object volume sampling,
controller final handles, and shape-prior surface/interior points are the data
product consumed by `realtime_phystwin`.

## Common Variants

Use external or already-running SAM3D worker:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v5/realtime_data_process_sam3d.py \
  --shape-prior-worker-mode external \
  --shape-prior-endpoint tcp://127.0.0.1:7100
```

Convert an existing headless capture without optimization:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v5/realtime_data_process_sam3d.py \
  --source-headless-capture result/demo_v5/capture_dir \
  --base-path result/demo_v5/rechunked \
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
anchor is lost, Demo v5 first tries a bounded KNN motion revive from nearby
direct anchors in the same fixed topology. A revived point keeps the original
sample/query id and is marked `revived` in the trace. If the local support is
not strong enough, Demo v5 holds the last finite point, marks the active query
id as `-1`, and clears visibility/motion-valid flags instead of replacing that
column with a new physical query. That preserves optimizer topology and avoids
quality loss from identity swaps.

## Validation

Deterministic checks:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v5/env/check_demo_v5_env.py --role main --require-cuda

conda run -n phystwin-max --no-capture-output \
  python demo_v5/env/check_demo_v5_env.py --role shape-prior --require-cuda

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
