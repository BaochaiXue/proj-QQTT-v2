# Demo 3 High-Performance Tracking Backends

Demo 3 keeps CoTracker3 as the PhysTwin-compatible baseline and adds
dependency-gated probes for high-performance candidates.

| Backend | Role | Current Integration |
| --- | --- | --- |
| CoTracker3 online | Robust PhysTwin-compatible baseline | Implemented, lazy model load |
| NVOFA | Frame-to-frame optical flow / point propagation | Availability and helper probe |
| VPI LK | Sparse pyramidal Lucas-Kanade point tracking | Availability-gated Python backend |
| TAPNext / TAPNext++ | Online neural TAP candidate | External tapnet probe |
| LocoTrack | Efficient near-dense neural tracking candidate | External repo/weights probe |
| TAPIR / BootsTAPIR | Optional TAP baseline | External tapnet probe |
| ONNX Runtime CUDA/TensorRT | Acceleration probe for exportable submodules | Provider config and non-fatal probe |

## Semantics

- NVOFA is frame-to-frame flow propagation, not long-term TAP identity
  tracking. It needs periodic re-anchor.
- VPI LK is sparse point tracking and is best treated as a fast fallback.
- TAPNext is the priority neural online TAP candidate.
- LocoTrack is the priority near-dense high-speed neural candidate.
- CoTracker3 remains the reference backend for PhysTwin-compatible
  `tracks_yx + visibility`.
- The Demo 3 `cotracker3_online` live path uses CoTracker3's online rolling
  buffer contract: `window_len=16`, `step=8`, first publish at 16 frames, then
  one publish every 8 new frames. The saved-case replay benchmark uses the same
  frame-by-frame `update(frame)` path for this backend. PhysTwin dense
  up-to-5000-point runs remain cached or offline artifacts unless explicitly
  enabled outside the render hot path.

## PhysTwin-Compatible CoTracker Export

Demo 3 benchmark/export defaults to the dense PhysTwin-compatible query mode
when the goal is to reproduce FuturePhysTwin-style CoTracker artifacts rather
than a sparse visualization overlay:

```bash
python scripts/harness/experiments/run_demo3_tracking_backend_benchmark.py \
  --case-root data/<case> \
  --backends cotracker3_online
```

`phystwin_dense` uses nested first-frame union masks, samples up to 5000
query points per camera with FuturePhysTwin-style torch `randperm`
(`seed + camera_idx`, default seed `42`), and writes `cotracker/{camera}.npz`.
Masks with fewer than 5000 pixels use all available mask pixels.
Pass `--query-mode object_sparse` explicitly for sparse overlay screening.
The existing Demo 3 overlay remains sparse by default and should consume dense
tracks only as an offline/cached artifact.

## Installation

Optional dependencies are installed manually through:

```bash
bash scripts/harness/experiments/demo3_tracking_backend_install/install_tracking_backends_optional.sh \
  --env demo3_trackers \
  --base-env demo_2_max \
  --install-locotrack \
  --install-tapnet
```

The installer clones external repos under
`/home/zhangxinjie/external_tracking_backends` by default and writes logs under
`data/experiments/demo3_tracking_backend_install_logs/`.

## Validation

Run:

```bash
python scripts/harness/experiments/check_demo3_tracking_backend_stack.py
```

The report is dependency-gated. Missing optional backends are reported as
unavailable and do not fail deterministic checks.

## 2026-05-10 Probe Result

The first install/probe pass used a `demo3_trackers` conda environment cloned
from `demo_2_max` to preserve Python, PyTorch, CUDA, and TensorRT
compatibility.

| Probe | Result |
| --- | --- |
| TensorRT Python | Available (`10.16.1.11`) |
| ONNX Runtime CUDA EP | Available |
| ONNX Runtime TensorRT EP | Available |
| tapnet / TAPNext / TAPIR repo | Installed/importable, but checkpoints and runtime wrappers are not configured |
| LocoTrack repo | Cloned and repo-path importable through `locotrack_pytorch`, but checkpoint/runtime wrapper is not configured |
| NVOFA SDK | Cloned, but no flow helper/binding is built yet |
| VPI LK | Unavailable; `import vpi` fails in the cloned environment |

Current recommendation: use CoTracker3 or cached/offline tracks for functional
Demo 3 overlay, use ONNX/TensorRT only as a provider/export probe for now, and
build an NVOFA helper or install VPI Python bindings before claiming a
hardware-accelerated live tracking backend.

## Backend Bring-Up Notes

### NVOFA

The Python backend already supports this helper contract:

```bash
run_nvofa_flow_helper \
  --prev frame_t.png \
  --next frame_t1.png \
  --out flow.npy
```

`flow.npy` must contain an `H,W,2` float array in x,y flow-vector order.
The Python backend samples this flow at `query_points_yx` and propagates
`tracks_yx` frame to frame.

Current blocker:

```text
/home/zhangxinjie/external_tracking_backends/NVIDIAOpticalFlowSDK
contains headers and README only. A runnable helper still needs the official
NVIDIA Optical Flow SDK sample package / libraries from the developer site or
an equivalent local C++ helper build.
```

### LocoTrack

LocoTrack is treated as a repo-path backend, not a required pip package.
Use one of these environment variables when the repo lives outside the default
external root:

```bash
export DEMO3_LOCOTRACK_REPO=/path/to/locotrack
export DEMO3_LOCOTRACK_CHECKPOINT=/path/to/weights
```

The current wrapper probes `locotrack_pytorch` / `locotrack` from that repo path
and stays unavailable until a stable checkpoint and runtime adapter are present.

### TAPNext / TAPNext++

TAPNext uses the tapnet repo path and checkpoint flags:

```bash
export DEMO3_TAPNET_REPO=/path/to/tapnet
export DEMO3_TAPNEXT_CHECKPOINT=/path/to/tapnext_checkpoint
```

No ONNX/TensorRT export should be attempted until the PyTorch/JAX runtime
wrapper produces `tracks_yx + visibility` on the benchmark harness.

## 2026-05-10 CoTracker3 Baseline Profile

The first real baseline profile was run on:

```text
case: data/still_object/ffs203048_iter4_trt_level5/both_30_still_object_round1_20260428
backend: cotracker3_online
cameras: 0,1,2
frames: 30
query points: 100,256,512,1024
mask: sam31_masks/mask
depth: native
```

Profile summary:

| Points | Three-camera serial FPS | Median camera E2E ms | p95 camera E2E ms |
| ---: | ---: | ---: | ---: |
| 100 | 1.211 | 586.150 | 1247.001 |
| 256 | 1.402 | 713.287 | 719.718 |
| 512 | 0.996 | 1012.185 | 1033.870 |
| 1024 | 0.652 | 1539.325 | 1548.438 |

Run-level profile:

```text
total_wall_ms: 15239.101
frame_load_ms_total: 657.400
mask_load_ms_total: 93.028
max_rss_mb: 1763.617
torch_cuda_peak_mb: 2605.316
backend_load_ms: 1492.178
```

This is a valid PhysTwin-compatible baseline result, not a high-performance
live backend result. It is too slow for a realtime three-camera overlay unless
used sparsely, cached, or moved out of the render hot path.
