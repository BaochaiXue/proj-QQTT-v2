# Demo v6.2 fake-camera and Phystwin proof — 2026-07-12

## Scope

Verified the current single-camera Demo v6.2 fake-live path in two stages:

- one bounded upstream run through canonical mask processing, tracking, shape
  prior, prepared frames, and one committed chunk;
- one formal default-output launch through the Phystwin_shen supervisor,
  combined HTML viewer, and the first Stage 1 realtime export.

The formal replay and 100-iteration train were still running when this proof
was captured. This record does not claim terminal full-pipeline success.

## Preconditions

- Repository branch: `main`
- Conda environment: `demo_2_max`
- Fake-live case: `data_collect/sloth_new_20260705_230611`
- Camera calibration: `table_calibrate.pkl`, serial `239222300740`
- GPUs: two NVIDIA GeForce RTX 4090 devices

## Bounded upstream run

Command:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v6_2/main.py \
  --input-source fake-live \
  --fake-live-case data_collect/sloth_new_20260705_230611 \
  --max-chunks 1 \
  --base-path /tmp/demo_v6_2_fake_camera_20260712_1chunk \
  --downstream-mode disabled \
  --no-warmup-rgb-preview
```

Outcome: orchestrator exit 0.

- `chunk_count=1`, `chunk_frame_count=5`
- `shape_prior_complete=true`
- `prepared_frame_count=5`
- `track_process_status=normal`
- object/controller point counts: 2087/30
- shape-prior surface/interior point counts: 493/1250
- ASAP fallback frame count: 0
- final-data write offset: 102.310 s
- stop reason: `max_chunks_reached`

The camera child return code was `-15` because the orchestrator deliberately
sent SIGTERM after the requested chunk committed. The outer command returned
0, and `pipeline_status.jsonl` ended with `run_finished`, `ok=true`.

## Formal default-output launch

Command:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v6_2/main.py \
  --input-source fake-live \
  --no-warmup-rgb-preview
```

Resolved runtime contract:

- base path: `/home/xinjie/single_proj_qqtt/outputs`
- downstream: `phystwin_shen`
- Phystwin repo: `/home/xinjie/Phystwin_shen`
- combined viewer: `http://127.0.0.1:8765/`
- Stage 1 source:
  `/home/xinjie/Phystwin_shen/experiments_online_cma/outputs/realtime`
- Train source:
  `/home/xinjie/Phystwin_shen/experiments_online_train/outputs/realtime`

Observed outcome at proof capture:

- shape-prior warm-up reached `ready` and opened the formal timeline;
- the points-ready trigger launched one Phystwin_shen supervisor;
- the viewer listened on `127.0.0.1:8765` and returned HTTP 200;
- Chrome reported that it opened the URL in the existing browser session;
- Stage 1 loaded online chunks and exported realtime candidate iteration 0.

One later Stage 1 batch logged a non-finite candidate loss while subsequent
simulation work continued. The producer later reached chunk 114; recent chunks
reported `track_process_status=degraded`, meaning at least one controller
anchor used the documented local-rigid proxy. Chunks continued committing and
no fatal status event was present. No terminal supervisor result had been
observed at proof capture, so this is launch/live-path evidence rather than a
completed training acceptance result.

## Warm-up profile

The formal run wrote `outputs/capture/shape_prior_profile.json` using timing
schema version 1.

- runtime start to shape-prior submit: 16.186 s
- submit critical path: 58.511 s
- shape-prior ready to formal gate open: 75.8 ms
- total warm-up: 74.773 s
- generate: 29.415 s, 50.27%
- align: 14.139 s, 24.16%
- upscale: 11.225 s, 19.18%
- sample: 3.133 s, 5.35%

All three prewarmed long stages reported `ready_before_go=true` and zero
startup tail on the critical path. Generate reported 14.567 s of pipeline run;
align reported 7.366 s rendering candidates, 4.081 s SuperGlue matching, and
about 1.213 s across the two ARAP phases.
