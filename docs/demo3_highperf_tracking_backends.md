# Demo 3 High-Performance Tracking Backends

Demo 3 keeps CoTracker3 as the PhysTwin-compatible baseline and adds dependency-gated probes for high-performance candidates.

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

- NVOFA is frame-to-frame flow propagation, not long-term TAP identity tracking. It needs periodic re-anchor.
- VPI LK is sparse point tracking and is best treated as a fast fallback.
- TAPNext is the priority neural online TAP candidate.
- LocoTrack is the priority near-dense high-speed neural candidate.
- CoTracker3 remains the reference backend for PhysTwin-compatible `tracks_yx + visibility`.

## Installation

Optional dependencies are installed manually through:

```bash
bash scripts/harness/experiments/demo3_tracking_backend_install/install_tracking_backends_optional.sh \
  --env demo3_trackers \
  --base-env demo_2_max \
  --install-locotrack \
  --install-tapnet
```

The installer clones external repos under `/home/zhangxinjie/external_tracking_backends` by default and writes logs under `data/experiments/demo3_tracking_backend_install_logs/`.

## Validation

Run:

```bash
python scripts/harness/experiments/check_demo3_tracking_backend_stack.py
```

The report is dependency-gated. Missing optional backends are reported as unavailable and do not fail deterministic checks.

## 2026-05-10 Probe Result

The first install/probe pass used a `demo3_trackers` conda environment cloned from `demo_2_max` to preserve Python, PyTorch, CUDA, and TensorRT compatibility.

| Probe | Result |
| --- | --- |
| TensorRT Python | Available (`10.16.1.11`) |
| ONNX Runtime CUDA EP | Available |
| ONNX Runtime TensorRT EP | Available |
| tapnet / TAPNext / TAPIR repo | Installed/importable, but checkpoints and runtime wrappers are not configured |
| LocoTrack repo | Cloned, but not pip-installable as a package in this probe |
| NVOFA SDK | Cloned, but no flow helper/binding is built yet |
| VPI LK | Unavailable; `import vpi` fails in the cloned environment |

Current recommendation: use CoTracker3 or cached/offline tracks for functional Demo 3 overlay, use ONNX/TensorRT only as a provider/export probe for now, and build an NVOFA helper or install VPI Python bindings before claiming a hardware-accelerated live tracking backend.
