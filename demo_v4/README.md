# Demo v4 FuturePhysTwin Chunk Runner

Demo v4 turns the single-camera Demo 3.2 realtime output into an online
FuturePhysTwin stream. It is the isolated bridge for acceptance testing:
Demo 3.2 handles RGB-D, masks, tracking, and optional SAM3D shape prior; Demo v4
publishes `online_data/<case>/manifest.json` plus `chunks/chunk_*.pkl`, with a
matching static `data/<case>/final_data.pkl` for realtime_phystwin consumers.

This is not the formal aligned-case product. The normal recording/alignment
pipeline still ends at `data_process/`.

## Quick Start

Run from the repo root on the `single-camera` branch.

Terminal 1: start the SAM3D shape-prior worker. Use the SAM3D/FuturePhysTwin
environment for this process.

```bash
CUDA_VISIBLE_DEVICES=1 \
conda run -n phystwin-max --no-capture-output \
  python services/shape_prior_remote/server.py \
  --bind tcp://127.0.0.1:7103 \
  --device cuda:0 \
  --preload-models \
  --debug
```

Terminal 2: run Demo v4. Use the integrated realtime demo environment.

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v4/realtime_futurephystwin_chunks.py \
  --realtime-gpu-mode single \
  --warmup-gpu-mode dual \
  --shape-prior-endpoint tcp://127.0.0.1:7103 \
  --case-prefix demo_v4
```

This default path uses:

```text
input source:        fake-live
depth backend:       native-realsense
output FPS:          5
chunk length:        5 seconds = 25 frames
shape prior:         enabled, remote worker
Demo 3.2 GPU:        CUDA_VISIBLE_DEVICES=0
shape-prior GPU:     cuda:1
dense pcd/ output:   disabled
output base path:    result/demo_v4/futurephystwin_chunks
```

For a short smoke run, add `--max-chunks 2`. When `--max-chunks` is used, Demo
v4 stops the Demo 3.2 subprocess after the requested number of chunks. A summary
with `demo32_stop_reason=max_chunks_reached` is expected.

## What Demo v4 Runs

The default realtime chain is:

```text
Demo 3.2 fake-live camera
  -> native RealSense color-aligned depth
  -> EdgeTAM object/controller masks
  -> TAPNext++ strict same-sequence tracks
  -> async SAM3D single-view shape prior
  -> Demo v4 online FuturePhysTwin chunks
```

Demo v4 derives the frame count from time: `round(--replay-fps *
--chunk-seconds)`. The default is `round(5 * 5) = 25` frames. Prefer changing
`--chunk-seconds` for operator runs; use `--chunk-frame-count` only when you
need an explicit test/debug override.

## Output Locations

Demo v4 writes into `--futurephystwin-base-path`. The default consumer-facing
layout combines a complete aggregate FuturePhysTwin-style case with the
`realtime_phystwin/scripts/fake_online_tracker.py` online stream:

```text
<base>/
  data/<case-prefix>/
    READY
    final_data.pkl
    track_process_data.pkl
    calibrate.pkl
    metadata.json
    split.json
    color/0/0.png
    color/0/1.png
    ...
    mask/processed_masks.pkl
    tracking/0.npz
    cotracker/0.npz
    pcd/0.npz              # only with --write-final-pcd

  online_data/<case-prefix>/
    manifest.json
    chunks/
      chunk_000000.pkl
      chunk_000001.pkl

  <case-prefix>_chunks_manifest.json
  <case-prefix>_demo32_capture_<timestamp>/
    metadata.json
    frames.jsonl
    shape_prior_profile.json
    shape_prior/points.npz
    ...
  <case-prefix>_chunk_0001/
    READY
    manifest.json
    final_data.pkl
    track_process_data.pkl
    calibrate.pkl
    metadata.json
    split.json
    color/0/0.png
    color/0/1.png
    ...
    mask/processed_masks.pkl
    tracking/0.npz
    cotracker/0.npz
    pcd/0.npz              # only with --write-final-pcd
```

Use these paths with realtime_phystwin online scripts:

```bash
--online_dir <base>/online_data/<case-prefix> \
--static_data_path <base>/data/<case-prefix>/final_data.pkl
```

`online_data/<case-prefix>/manifest.json` is atomically updated after each
`chunks/chunk_*.pkl` file is committed. The aggregate `data/<case-prefix>/`
case is rebuilt from committed diagnostic chunks and gets `READY` only when the
online stream finishes. Its per-frame artifacts use received-frame numbering:
the first published frame is `0`, the next is `1`, and later chunks continue
from the current aggregate frame count. That numbering is independent of any
fake-live source frame number preserved for traceability in online chunk
metadata.

The per-window `<case-prefix>_chunk_XXXX/` directories remain diagnostic
compatibility artifacts. Consumers that read those directories should only read
ones containing `READY`. Demo v4 writes each diagnostic case under
`<base>/.publishing/`, validates it, writes `READY`, and then atomically renames
it to `<base>/<case-prefix>_chunk_XXXX/`.

The top-level `<case-prefix>_chunks_manifest.json` summarizes the whole run.
Each chunk also has its own `manifest.json` with publish timing, point counts,
shape-prior status, and validation metadata.

## Check A Run

List ready chunks:

```bash
find result/demo_v4/futurephystwin_chunks \
  -maxdepth 2 -name READY -print | sort
```

Inspect the run summary:

```bash
python - <<'PY'
import json
from pathlib import Path

base = Path("result/demo_v4/futurephystwin_chunks")
summary = json.loads((base / "demo_v4_chunks_manifest.json").read_text())
print("chunks:", summary["chunk_count"])
print("online:", summary["online_dir"])
print("static:", summary["static_data_path"])
print("stop:", summary.get("demo32_stop_reason"))
print("first ready wall s:", summary.get("first_ready_chunk_wall_s"))
print("max backlog:", summary.get("max_backlog_chunks"))
print("validation cases:", summary.get("validation_chunk_cases"))
PY
```

Validate one published chunk with the writer's built-in checker:

```bash
python - <<'PY'
from demo_v4.futurephystwin_chunk_writer import validate_futurephystwin_case

case = "result/demo_v4/futurephystwin_chunks/demo_v4_chunk_0001"
print(validate_futurephystwin_case(case, require_ready=True))
PY
```

The generated diagnostic case can still be used from FuturePhysTwin as a normal
case root under `data/different_types`-style expectations: `final_data.pkl`,
`calibrate.pkl`, `metadata.json`, `split.json`, masks, RGB, and tracking files
are present. For online training and rollout, prefer the `online_dir` and
`static_data_path` pair shown above.

## Common Commands

Short debug run with two chunks:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v4/realtime_futurephystwin_chunks.py \
  --shape-prior-endpoint tcp://127.0.0.1:7103 \
  --futurephystwin-base-path result/demo_v4/debug_cases \
  --case-prefix demo_v4_debug \
  --max-chunks 2
```

Full default run. Published chunk metadata and Demo 3.2 fake-live pacing both
stay at 5 FPS.

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v4/realtime_futurephystwin_chunks.py \
  --shape-prior-endpoint tcp://127.0.0.1:7103 \
  --futurephystwin-base-path result/demo_v4/cadence_cases \
  --case-prefix demo_v4_cadence \
  --replay-fps 5 \
  --shape-prior-timeout-ms 240000 \
  --shape-prior-chunk-wait-timeout-s 240
```

For cadence stress testing only, add a slightly faster Demo 3.2 source pace:

```bash
  --demo32-source-replay-fps 5.2 \
  --demo32-lossless-max-backlog-seconds 45
```

That keeps published chunks at `fps=5` when `--replay-fps 5` remains set, but
feeds fake-live frames about 4% faster to prove the pipeline has margin.

Run without SAM3D shape prior. This is useful for debugging RGB/mask/tracking
chunking without waiting for the worker.

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v4/realtime_futurephystwin_chunks.py \
  --no-shape-prior-warmup \
  --futurephystwin-base-path result/demo_v4/no_shape_cases \
  --case-prefix demo_v4_no_shape \
  --max-chunks 2
```

Write dense per-frame PCD files inside each chunk:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v4/realtime_futurephystwin_chunks.py \
  --shape-prior-endpoint tcp://127.0.0.1:7103 \
  --futurephystwin-base-path result/demo_v4/pcd_cases \
  --case-prefix demo_v4_pcd \
  --write-final-pcd \
  --max-chunks 2
```

Convert an existing Demo 3.2 headless capture without launching Demo 3.2:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v4/realtime_futurephystwin_chunks.py \
  --source-headless-capture result/demo_v4/debug_cases/demo_v4_debug_demo32_capture_YYYYMMDD_HHMMSS \
  --futurephystwin-base-path result/demo_v4/rechunked_cases \
  --case-prefix demo_v4_rechunked \
  --max-chunks 2
```

Print the resolved contract and exit:

```bash
python demo_v4/realtime_futurephystwin_chunks.py --dry-run
```

## Important Options

### Input And Chunk Timing

| Option | Default | Use |
| --- | --- | --- |
| `--input-source {fake-live,live}` | `fake-live` | Choose replay-like fake camera input or live camera input for Demo 3.2. |
| `--replay-fps` | `5.0` | Logical FPS written to chunk `metadata.json` and used for chunk window math. |
| `--demo32-source-replay-fps` | unset | Optional Demo 3.2 fake-live pacing FPS. Leave unset for normal runs; use values like `5.2` only for cadence stress tests. |
| `--chunk-seconds` | `5.0` | Preferred way to change chunk duration. |
| `--chunk-frame-count` | unset | Explicit frame-count override. Still requires positive `--chunk-seconds` and `--replay-fps`. |
| `--max-chunks` | unset | Stop after N chunks for debug runs. Omit for a full source run. |
| `--capture-extra-seconds` | `10.0` | Extra Demo 3.2 runtime for max-chunk runs so startup/warmup latency does not cut capture short. |

### Output

| Option | Default | Use |
| --- | --- | --- |
| `--futurephystwin-base-path` | `result/demo_v4/futurephystwin_chunks` | Directory where chunk cases and the run manifest are published. |
| `--case-prefix` | `demo_v4` | Prefix for chunk case names and the top-level manifest. |
| `--demo32-capture-dir` | auto under output base | Where the intermediate Demo 3.2 headless capture is written. |
| `--source-headless-capture` | unset | Rechunk an existing Demo 3.2 capture instead of launching Demo 3.2. |
| `--write-final-pcd` | off | Also write dense `pcd/<frame>.npz` files in each chunk. |
| `--no-write-final-pcd` | on by default | Keep chunks smaller; `final_data.pkl` and tracking outputs still exist. |

### GPU Routing

| Option | Default | Use |
| --- | --- | --- |
| `--realtime-gpu-mode {single,dual}` | `single` | Select Demo 3.2 subprocess CUDA visibility. `single` maps to `CUDA_VISIBLE_DEVICES=0`; `dual` maps to `CUDA_VISIBLE_DEVICES=1`. |
| `--warmup-gpu-mode {single,dual}` | `dual` | Select default shape-prior device. `single` maps to `cuda:0`; `dual` maps to `cuda:1`. |
| `--gpu-mode` | `single` | Backward-compatible alias for realtime routing. Prefer `--realtime-gpu-mode`. |
| `--demo32-cuda-visible-devices` | derived | Explicit override for the Demo 3.2 subprocess. |
| `--shape-prior-device` | derived | Explicit override for the shape-prior device passed to Demo 3.2. |
| `--demo32-device` | `cuda` | Segmentation/runtime device inside the Demo 3.2 subprocess namespace. |
| `--demo32-tracker-device` | `cuda` | TAPNext++ tracker device inside the Demo 3.2 subprocess namespace. |
| `--demo32-dtype` | `bfloat16` | Demo 3.2 segmentation/runtime dtype. |

Default production routing is `--realtime-gpu-mode single --warmup-gpu-mode
dual`: Demo 3.2 realtime runs on GPU0 and the external SAM3D worker runs on
GPU1.

Single-GPU fallback:

```bash
python demo_v4/realtime_futurephystwin_chunks.py \
  --realtime-gpu-mode single \
  --warmup-gpu-mode single
```

Realtime isolation on GPU1:

```bash
python demo_v4/realtime_futurephystwin_chunks.py \
  --realtime-gpu-mode dual
```

### Shape Prior

| Option | Default | Use |
| --- | --- | --- |
| `--shape-prior-warmup` / `--no-shape-prior-warmup` | on | Enable or disable SAM3D shape prior. |
| `--shape-prior-execution {remote-worker,local-subprocess}` | `remote-worker` | Worker mode used by Demo 3.2. Demo v4 runs should use `remote-worker`. |
| `--shape-prior-endpoint` | `tcp://127.0.0.1:7100` | Endpoint of `services/shape_prior_remote/server.py`. |
| `--shape-prior-timeout-ms` | `180000` | Demo 3.2 request timeout for the worker. |
| `--shape-prior-chunk-wait-timeout-s` | `300` | How long Demo v4 waits for required shape-prior points before writing chunks. |
| `--shape-prior-start-policy` | `async-after-first-mask-depth-pair` | When Demo 3.2 submits the shape-prior request. |
| `--shape-prior-profile-json` | capture dir | Where Demo 3.2 writes shape-prior timing/status JSON. |
| `--shape-prior-skip-route-visualizations` | on | Skip worker route visualizations. |
| `--shape-prior-render-route-visualizations` | off | Render worker route visualizations for debugging. |

Shape-prior start policies:

```text
async-after-first-mask-depth-pair   default; request starts as soon as mask+depth are available
async-after-first-strict-pair       wait for strict tracking pair before request
blocking-before-first-output        block first output until shape prior is done
after-teardown                      submit after capture teardown for offline diagnostics
```

### Mask And PCD Filtering

| Option | Default | Use |
| --- | --- | --- |
| `--mask-radius-outlier-filter` | on | Apply data-process-style 3D radius-outlier cleanup before final-data chunking. |
| `--no-mask-radius-outlier-filter` | off | Disable cleanup for tiny synthetic fixtures only. |
| `--mask-radius-outlier-radius-m` | `0.01` | Radius for the mask outlier filter. |
| `--mask-radius-outlier-nb-points` | `40` | Neighbor threshold for the mask outlier filter. |
| `--surface-points-npy` | unset | External `Nx3` surface points override/augmentation for tests. |
| `--interior-points-npy` | unset | External `Nx3` interior points override/augmentation for tests. |

## Shape-Prior Worker Options

The worker command is:

```bash
python services/shape_prior_remote/server.py --help
```

Useful worker options:

| Option | Default | Use |
| --- | --- | --- |
| `--bind` | `tcp://0.0.0.0:7100` | ZeroMQ REP endpoint. Match Demo v4 `--shape-prior-endpoint`. |
| `--sam3d-root` | `vendor/demo_runtime/sam-3d-objects` | Repo-local SAM3D Objects checkout copy. |
| `--futurephystwin-root` | `vendor/demo_runtime/FuturePhysTwin` | Repo-local FuturePhysTwin checkout copy used by worker imports. |
| `--config` | SAM3D default YAML | Explicit SAM3D pipeline config. |
| `--device` | `cuda:0` | Device visible inside the worker process. |
| `--max-points` | `60000` | Maximum observation/aligned point count returned by worker. |
| `--upscale-category` | default category text | Prompt category used by the x4 upscaler. |
| `--echo-observation` | off | Debug mode: return the observation PCD without loading SAM3D. |
| `--preload-models` | off | Load upscaler and SAM3D model before binding the endpoint. |
| `--warmup-models` | off | Also run a dummy warmup pass; implies `--preload-models` and needs more VRAM. |
| `--debug` | off | Print worker diagnostics. |

Prefer `--preload-models` for normal runs. `--warmup-models` can OOM on the
SAM3D decode path on tighter GPUs; use it only when validating warmup behavior.

## Published Case Contract

Each ready chunk contains the files FuturePhysTwin needs:

```text
final_data.pkl
track_process_data.pkl
calibrate.pkl
metadata.json
split.json
color/0/<frame>.png
mask/processed_masks.pkl
tracking/0.npz
cotracker/0.npz
manifest.json
READY
```

For the aggregate online-primary case, `<frame>` is the global received-frame
index within Demo v4's published stream. For example, with 25-frame chunks,
`chunk_0001` writes `color/0/0.png` through `color/0/24.png`, and `chunk_0002`
continues at `color/0/25.png`.

`final_data.pkl` contains:

```text
object_points
object_colors
object_visibilities
object_motions_valid
controller_points
controller_mask
surface_points
interior_points
```

FuturePhysTwin uses the first observed object points plus the SAM3D
surface/interior samples as structure points:

```text
object_points[0] + surface_points + interior_points
```

`metadata.json` includes the output FPS, frame count, image size, one-camera
intrinsics, serial number, `camera_count=1`, `demo_version=demo_v4`, and depth
backend fields. Recent FuturePhysTwin code must consume `metadata["fps"]` so
5 FPS chunks simulate one frame over 0.2 seconds.

## Manifest Fields To Watch

The top-level run manifest and each chunk manifest include:

```text
chunk_count
validation_chunk_cases
first_ready_chunk_wall_s
first_shape_prior_ready_chunk_wall_s
steady_state_publish_interval_max_s
max_backlog_chunks
demo32_stop_reason
publish_latency_ms
publish_lag_ms
backlog_chunks
shape_prior_complete
shape_prior_target_counts_met
object_point_count
controller_point_count
surface_point_count
interior_point_count
```

For realtime cadence, watch `steady_state_publish_interval_max_s`,
`publish_latency_ms`, and `backlog_chunks`. A stable run should not show
steadily growing backlog after startup. `publish_wall_s` is the consumer-visible
READY publish time.

When at least five chunks exist, Demo v4 selects validation cases from the
second-last and fifth-last chunks. This avoids validating only an early chunk
before the controller has moved enough.

## Notes On Data-Process Compatibility

Demo v4 mirrors the data-process behavior that matters for FuturePhysTwin:

- object/controller labels come from first-frame semantic masks
- per-frame masks gate visibility
- masks are intersected with valid depth before track processing
- 3D mask radius-outlier cleanup defaults to 1 cm radius and 40 neighbors
- controller/object overlap resolves with controller priority
- motion filtering follows the data-process 1 cm / 5-neighbor / 5 mm policy
- controller points are selected down to 30 points
- object points are sampled on a 5 mm grid
- shape-prior surface and interior samples stay in separate final-data fields
- table-world z-down coordinates are preserved

Scheduling is different from offline `data_process_sam3d`: Demo v4 streams
chunks from the realtime/fake-live timeline, while SAM3D shape prior runs
asynchronously and does not rewrite EdgeTAM masks, TAPNext++ tracks, or strict
tracking identities.

## Troubleshooting

- No chunk folders appear: check the Demo 3.2 capture directory for
  `metadata.json` and `frames.jsonl`, and confirm the shape-prior worker
  endpoint matches `--shape-prior-endpoint`.
- Chunk folder exists without `READY`: ignore it. A valid consumer should only
  read ready chunks.
- Shape prior is slow: increase `--shape-prior-timeout-ms` and
  `--shape-prior-chunk-wait-timeout-s`, or run a debug pass with
  `--no-shape-prior-warmup`.
- GPU contention: keep the default split,
  `--realtime-gpu-mode single --warmup-gpu-mode dual`, and start the worker
  with `CUDA_VISIBLE_DEVICES=1`.
- Need smaller output: keep the default `--no-write-final-pcd`.
- Need dense point-cloud diagnostics: add `--write-final-pcd`.
