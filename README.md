# proj-QQTT-v2 Single-Camera Demo v4

The `main` branch is currently organized around Demo v4: a
single-camera realtime preprocessing bridge that publishes FuturePhysTwin-ready
chunk cases. Demo v4 launches the sanctioned Demo 3.2 headless runtime, collects
RGB-D, masks, strict TAPNext++ tracks, and optional SAM3D single-view shape
prior, then writes complete chunk folders for FuturePhysTwin.

The older camera preview, calibration, recording, alignment, and visualization
tools still live in this repo. They are secondary in this README; see
`docs/WORKFLOWS.md`, `docs/ARCHITECTURE.md`, and `docs/SCOPE.md` for the full
single-camera repository map. The former multi-camera baseline is preserved on
the `multiple-camera` branch.

## What Demo v4 Produces

Demo v4 publishes one FuturePhysTwin case per time window:

```text
Demo 3.2 fake-live or live camera
  -> native RealSense or IR-FFS depth
  -> EdgeTAM object/controller masks
  -> TAPNext++ strict same-sequence tracks
  -> optional SAM3D single-view shape prior
  -> FuturePhysTwin chunk case folders
```

Default Demo v4 contract:

```text
input_source=fake-live
depth_backend=native-realsense
replay_fps=5.0
chunk_seconds=5.0
chunk_frame_count=25
shape_prior_warmup=true
shape_prior_execution=remote-worker
shape_prior_start_policy=async-after-first-mask-depth-pair
realtime_gpu_mode=single
warmup_gpu_mode=dual
demo32_cuda_visible_devices=0
shape_prior_device=cuda:1
write_final_pcd=false
futurephystwin_base_path=result/demo_v4/futurephystwin_chunks
```

Run `python demo_v4/realtime_futurephystwin_chunks.py --dry-run` to print the
resolved contract for your exact command.

## Environments

Use these local environments for the validated Demo v4 path:

```text
demo_2_max     Demo v4 launcher, Demo 3.2 realtime runtime, EdgeTAM, TAPNext++
phystwin-max   long-lived SAM3D shape-prior worker
```

Demo runtime code and lightweight configs live under `vendor/demo_runtime/`.
Heavy checkpoint payloads live under `vendor/demo_runtime/checkpoints/`, which
is intentionally gitignored.

```text
vendor/demo_runtime/sam-3d-objects
```

## Quick Start

Run all commands from the repo root on branch `main`.

Terminal 1: start the shape-prior worker on GPU1. The worker process sees GPU1
as `cuda:0` because of `CUDA_VISIBLE_DEVICES=1`.

```bash
CUDA_VISIBLE_DEVICES=1 \
conda run -n phystwin-max --no-capture-output \
  python services/shape_prior_remote/server.py \
  --bind tcp://127.0.0.1:7103 \
  --device cuda:0 \
  --preload-models \
  --debug
```

Terminal 2: run Demo v4.

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v4/realtime_futurephystwin_chunks.py \
  --realtime-gpu-mode single \
  --warmup-gpu-mode dual \
  --shape-prior-endpoint tcp://127.0.0.1:7103 \
  --case-prefix demo_v4
```

For a short smoke run, add `--max-chunks 2` to the Demo v4 command. The
expected controlled stop is recorded in the run summary as
`demo32_stop_reason=max_chunks_reached`.

## Common Run Commands

### Short Debug Run

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v4/realtime_futurephystwin_chunks.py \
  --shape-prior-endpoint tcp://127.0.0.1:7103 \
  --futurephystwin-base-path result/demo_v4/debug_cases \
  --case-prefix demo_v4_debug \
  --max-chunks 2
```

### Full Default Run

Use this for a normal full fake-live pass. It keeps both Demo v4 output timing
and Demo 3.2 fake-live pacing at the default 5 FPS.

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

That option does not change published chunk metadata when `--replay-fps 5`
remains set; it only makes fake-live input arrive about 4% faster to prove the
pipeline can keep up with more than the target 5 FPS.

### Run Without SAM3D Shape Prior

Use this when debugging RGB, masks, tracking, chunk publishing, or output paths
without waiting for the worker.

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v4/realtime_futurephystwin_chunks.py \
  --no-shape-prior-warmup \
  --futurephystwin-base-path result/demo_v4/no_shape_cases \
  --case-prefix demo_v4_no_shape \
  --max-chunks 2
```

### Write Dense PCD Files

The default skips dense `pcd/` files to keep publishing fast. Add
`--write-final-pcd` when a diagnostic/export consumer needs per-frame point
clouds.

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v4/realtime_futurephystwin_chunks.py \
  --shape-prior-endpoint tcp://127.0.0.1:7103 \
  --futurephystwin-base-path result/demo_v4/pcd_cases \
  --case-prefix demo_v4_pcd \
  --write-final-pcd \
  --max-chunks 2
```

### Convert An Existing Headless Capture

Use `--source-headless-capture` to chunk a previously recorded Demo 3.2 headless
capture without launching Demo 3.2 again.

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v4/realtime_futurephystwin_chunks.py \
  --source-headless-capture result/demo_v4/debug_cases/demo_v4_debug_demo32_capture_YYYYMMDD_HHMMSS \
  --futurephystwin-base-path result/demo_v4/rechunked_cases \
  --case-prefix demo_v4_rechunked \
  --max-chunks 2
```

### Live Camera Mode

Fake-live is the default. To use a live camera source, switch the input source
explicitly and keep the rest of the publishing options the same.

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v4/realtime_futurephystwin_chunks.py \
  --input-source live \
  --shape-prior-endpoint tcp://127.0.0.1:7103 \
  --futurephystwin-base-path result/demo_v4/live_cases \
  --case-prefix demo_v4_live
```

## Output Results

Demo v4 writes outputs under `--futurephystwin-base-path`.

```text
<base>/
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
    color/0/<frame>.png
    mask/processed_masks.pkl
    tracking/0.npz
    cotracker/0.npz
    pcd/<frame>.npz        # only when --write-final-pcd is enabled
```

Only chunk directories with `READY` are complete. Demo v4 stages each chunk
under `<base>/.publishing/`, validates it, writes `READY`, then atomically
renames the directory into place.

The top-level `<case-prefix>_chunks_manifest.json` summarizes the run. Each
chunk also has `manifest.json` with publish timing, backlog, point counts,
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
print("stop:", summary.get("demo32_stop_reason"))
print("first ready wall s:", summary.get("first_ready_chunk_wall_s"))
print("first shape prior ready wall s:", summary.get("first_shape_prior_ready_chunk_wall_s"))
print("max backlog:", summary.get("max_backlog_chunks"))
print("validation cases:", summary.get("validation_chunk_cases"))
PY
```

Validate a published chunk:

```bash
conda run -n demo_2_max --no-capture-output python - <<'PY'
from demo_v4.futurephystwin_chunk_writer import validate_futurephystwin_case

case = "result/demo_v4/futurephystwin_chunks/demo_v4_chunk_0001"
print(validate_futurephystwin_case(case, require_ready=True))
PY
```

## Demo v4 Options

### Input And Chunk Timing

| Option | Default | Use |
| --- | --- | --- |
| `--input-source {fake-live,live}` | `fake-live` | Choose fake-live replay-style input or a live camera. |
| `--depth-backend {native-realsense,ir-ffs}` | `native-realsense` | Select the depth path passed to Demo 3.2. |
| `--replay-fps` | `5.0` | Logical output FPS written to chunk `metadata.json` and used for chunk windows. |
| `--demo32-source-replay-fps` | unset | Optional fake-live pacing FPS for Demo 3.2. Leave unset for normal runs; use values like `5.2` only for cadence stress tests. |
| `--chunk-seconds` | `5.0` | Preferred chunk duration control. |
| `--chunk-frame-count` | unset | Explicit frame-count override; defaults to `round(replay_fps * chunk_seconds)`. |
| `--chunk-poll-interval-s` | `0.001` | Polling interval while tailing Demo 3.2 `frames.jsonl`. |
| `--max-chunks` | unset | Stop after N chunks for debug runs. Omit for a full source run. |
| `--capture-extra-seconds` | `10.0` | Extra Demo 3.2 runtime for max-chunk runs. |

### Output

| Option | Default | Use |
| --- | --- | --- |
| `--futurephystwin-base-path` | `result/demo_v4/futurephystwin_chunks` | Published chunk cases and run manifest. |
| `--case-prefix` | `demo_v4` | Prefix for chunk names and the run manifest. |
| `--demo32-capture-dir` | auto under output base | Intermediate Demo 3.2 headless capture directory. |
| `--source-headless-capture` | unset | Rechunk an existing capture instead of launching Demo 3.2. |
| `--write-final-pcd` | off | Write dense per-frame `pcd/*.npz` files. |
| `--no-write-final-pcd` | on by default | Keep output smaller and faster. |

### GPU Routing

| Option | Default | Use |
| --- | --- | --- |
| `--realtime-gpu-mode {single,dual}` | `single` | Demo 3.2 CUDA visibility. `single` maps to `CUDA_VISIBLE_DEVICES=0`; `dual` maps to `CUDA_VISIBLE_DEVICES=1`. |
| `--warmup-gpu-mode {single,dual}` | `dual` | Shape-prior device preset. `single` maps to `cuda:0`; `dual` maps to `cuda:1`. |
| `--gpu-mode` | `single` | Backward-compatible alias for realtime routing. Prefer `--realtime-gpu-mode`. |
| `--demo32-cuda-visible-devices` | derived | Explicit CUDA visibility override for Demo 3.2. |
| `--shape-prior-device` | derived | Explicit shape-prior device passed into Demo 3.2. |
| `--demo32-device` | `cuda` | Segmentation/runtime device inside Demo 3.2. |
| `--demo32-tracker-device` | `cuda` | TAPNext++ tracker device inside Demo 3.2. |
| `--demo32-dtype` | `bfloat16` | Demo 3.2 segmentation/runtime dtype. |

The preferred two-GPU route is `--realtime-gpu-mode single
--warmup-gpu-mode dual`: Demo 3.2 realtime work runs on GPU0 and the external
SAM3D worker runs on GPU1.

### Shape Prior

| Option | Default | Use |
| --- | --- | --- |
| `--shape-prior-warmup` / `--no-shape-prior-warmup` | on | Enable or disable SAM3D shape prior. |
| `--shape-prior-execution {remote-worker,local-subprocess}` | `remote-worker` | Shape-prior execution mode passed to Demo 3.2. |
| `--shape-prior-endpoint` | `tcp://127.0.0.1:7100` | Worker endpoint. Set to `tcp://127.0.0.1:7103` when using the quick-start worker. |
| `--shape-prior-timeout-ms` | `180000` | Demo 3.2 request timeout. |
| `--shape-prior-chunk-wait-timeout-s` | `300` | How long Demo v4 waits for shape-prior points before writing a chunk. |
| `--shape-prior-start-policy` | `async-after-first-mask-depth-pair` | When Demo 3.2 submits the shape-prior request. |
| `--shape-prior-profile-json` | capture dir | Shape-prior timing/status JSON path. |
| `--shape-prior-skip-route-visualizations` | on | Skip worker route visualizations. |
| `--shape-prior-render-route-visualizations` | off | Render worker route visualizations for debugging. |

Shape-prior start policies:

```text
async-after-first-mask-depth-pair   default; submit when mask+depth are available
async-after-first-strict-pair       wait for strict tracking pair
blocking-before-first-output        block first output until shape prior finishes
after-teardown                      submit after capture teardown for diagnostics
```

### Mask, PCD, And External Shape Points

| Option | Default | Use |
| --- | --- | --- |
| `--mask-radius-outlier-filter` | on | Apply data-process-style 3D mask outlier cleanup. |
| `--no-mask-radius-outlier-filter` | off | Disable cleanup for tiny synthetic fixtures only. |
| `--mask-radius-outlier-radius-m` | `0.01` | Radius for mask outlier cleanup. |
| `--mask-radius-outlier-nb-points` | `40` | Neighbor threshold for mask outlier cleanup. |
| `--surface-points-npy` | unset | External `Nx3` surface points for tests or ablations. |
| `--interior-points-npy` | unset | External `Nx3` interior points for tests or ablations. |

## Shape-Prior Worker Options

Worker command:

```bash
python services/shape_prior_remote/server.py --help
```

Useful worker options:

| Option | Default | Use |
| --- | --- | --- |
| `--bind` | `tcp://0.0.0.0:7100` | ZeroMQ REP bind endpoint. |
| `--sam3d-root` | local default | External SAM3D Objects checkout. |
| `--futurephystwin-root` | local default | FuturePhysTwin checkout used by worker imports. |
| `--config` | SAM3D default YAML | Explicit SAM3D pipeline config. |
| `--device` | `cuda:0` | Worker-visible device. |
| `--seed` | `42` | Worker random seed. |
| `--max-points` | `60000` | Maximum returned observation/aligned point count. |
| `--upscale-category` | default category text | x4 upscaler prompt category. |
| `--echo-observation` | off | Debug mode returning the observation PCD without SAM3D. |
| `--preload-models` | off | Load upscaler and SAM3D before binding. |
| `--warmup-models` | off | Run a dummy warmup pass; implies preload and uses more VRAM. |
| `--debug` | off | Print worker diagnostics. |

Prefer `--preload-models` for normal Demo v4 runs. Use `--warmup-models` only
when explicitly validating warmup behavior.

## FuturePhysTwin Case Contract

Each ready chunk contains:

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

FuturePhysTwin uses:

```text
object_points[0] + surface_points + interior_points
```

`metadata.json` records `fps`, `frame_num`, `WH`, one-camera intrinsics,
`camera_count=1`, `demo_version=demo_v4`, serial numbers, and depth backend
fields. FuturePhysTwin must consume `metadata["fps"]` so default Demo v4
5 FPS chunks simulate one frame over 0.2 seconds.

## Other Project Workflows

These commands still exist, but they are not the primary Demo v4 path.

Preview:

```bash
python cameras_viewer.py
```

Calibrate:

```bash
python cameras_calibrate.py --width 1280 --height 720 --fps 5
```

Record a raw RGB-D case:

```bash
python record_data.py --case_name my_case --capture_mode rgbd
```

Align a raw case:

```bash
python data_process/record_data_align.py \
  --case_name my_case \
  --start 0 \
  --end 120 \
  --depth_backend realsense
```

Native-vs-FFS depth comparison:

```bash
python scripts/harness/diagnostics/depth/visual_compare_depth_panels.py \
  --aligned_root ./data \
  --realsense_case native_case \
  --ffs_case ffs_case \
  --write_mp4 \
  --use_float_ffs_depth_when_available
```

## Validation

Demo v4 focused tests:

```bash
conda run -n demo_2_max --no-capture-output \
  python -m pytest tests/test_demo_v4_futurephystwin_chunks.py -q
```

Default smoke profile:

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/validation/run.py --profile smoke
```

Broader deterministic profile:

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/validation/run.py --profile deterministic
```

More Demo v4 details live in `demo_v4/README.md`.
