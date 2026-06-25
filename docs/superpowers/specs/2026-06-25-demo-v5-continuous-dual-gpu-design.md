# Demo v5 Continuous Dual-GPU Realtime Design

## Goal

Demo v5 runs the single-camera Demo v5 realtime product into a single
FuturePhysTwin-compatible online case, then feeds that same growing case to one
continuous `realtime_phystwin/train_online_zero_then_first.py` optimization
process. It must not optimize independent per-chunk cases.

## Recommended Design

Use the established FuturePhysTwin chunk contract and strict tracking product
as the contract baseline, but make Demo v5 its own package and entrypoint. Demo
v5 keeps the `demo_v4_session_topology_v1` topology version because
`realtime_phystwin` already validates that contract: `query_ids`,
`query_semantic_labels`, fixed object/controller sample ids, and topology hash
must remain stable for the full session.

Default runtime routing is:

- warmup phase: managed SAM3D remote worker on physical GPU1
- after first shape-prior-backed online chunk: terminate the managed warmup
  worker and start continuous `realtime_phystwin` optimization on physical GPU1
- realtime camera/fake camera, masks, tracking, final data, and chunk publishing
  stay on physical GPU0

The optimization process receives `CUDA_VISIBLE_DEVICES=1` and uses
`--device cuda:0` inside that process. This gives it physical GPU1 while leaving
the optimization code's internal defaults intact.

## Data Contract

Demo v5 writes:

- `online_data/<case>/manifest.json`
- `online_data/<case>/chunks/chunk_*.pkl`
- aggregate `data/<case>/final_data.pkl`
- aggregate `data/<case>/calibrate.pkl`, `metadata.json`, `color/0`, masks,
  tracking, and `READY`
- diagnostic per-window chunk directories for inspection

`realtime_phystwin` consumes only the aggregate static case plus the online
chunk stream. Demo v5 keeps paths portable: commands are written with relative
paths. Because the optimization subprocess runs with `cwd=realtime_phystwin/`,
its data paths are relative to that directory:

```text
--base_path ../result/demo_v5/futurephystwin_chunks/data
--online_dir ../result/demo_v5/futurephystwin_chunks/online_data/<case>
--static_data_path ../result/demo_v5/futurephystwin_chunks/data/<case>/final_data.pkl
--case_name <case>
```

## Tracking Continuity

The v5 bridge uses the existing streaming object/controller anchor selectors.
They keep column identity fixed across chunks and mark missing anchors without
replacing them with a different query. If a query is unavailable, the selector
holds the last finite point, marks the active query as `-1`, and clears
visibility/motion-valid flags. This is conservative for optimizer quality and
prevents topology drift.

KNN/LBS-style revive can be added later as a bounded quality improvement, but it
must preserve the fixed sample ids and must not replace a physical column with a
new semantic identity. The initial v5 completion prioritizes a strict continuous
case contract over speculative revive behavior.

## Quality And Realtime Policy

Demo v5 does not reduce zero-order or first-order optimization quality settings
to hit cadence. The defaults keep the existing optimizer values:

- zero-order iterations: `10`
- batch size: `4`
- segment length: the chunk frame count, default `35`
- segment stride: `16`
- recent window count: `8`

The 5 FPS requirement applies to the camera/fake-camera to online final-data
publication path. The optimizer runs concurrently as a consumer of the growing
online stream. Demo v5 does not pass `--stop_when_finished` by default; for
finite fake-live runs, first-order optimization continues to its configured
iteration budget so quality remains closer to offline FuturePhysTwin.

## Validation

Deterministic validation should cover:

- v5 parser defaults: fake-live, 5 FPS, 35-frame chunks, managed warmup,
  realtime GPU0, optimization GPU1
- v5 owns its runtime modules internally
- v5 starts one continuous optimization process after the first committed online
  chunk and does not start optimization per chunk
- source-headless conversion refuses continuous optimization unless explicitly
  disabled
- generated command line points `realtime_phystwin` at `online_dir` plus
  aggregate `static_data_path`
- repo smoke validation
- final validation must include a real fake-live to optimization run and compare
  the online result against the closest offline FuturePhysTwin baseline without
  obvious quality regression.
