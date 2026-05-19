# Harness Engineering

This repo treats harness engineering as a first-class part of the system. The goal is not to add more
scripts for their own sake; it is to make the camera/demo stack legible enough that an agent can debug,
validate, and improve it without relying on hidden chat context.

## Operating Model

- `AGENTS.md` is the short injected map.
- `docs/` is the durable knowledge store.
- `docs/exec-plans/` records intent, decisions, validation, and follow-up state for non-trivial changes.
- `scripts/harness/` is the executable control surface: checks, probes, summaries, comparisons, and bounded diagnostics.
- `docs/generated/` and `result/` are session evidence, not source-of-truth runtime dependencies.

When a task fails, prefer adding a missing map, guard, probe, or summary over only patching the immediate symptom.

## Stable Interfaces

Harnesses should stay stable even as model behavior and runtime implementation change.

| Interface | Repo Shape | Contract |
| --- | --- | --- |
| Session evidence | `docs/generated/`, `result/` | Append-only or timestamped artifacts that a future agent can summarize. |
| Harness commands | `scripts/harness/*.py` | Thin CLIs with catalog entries and help coverage when useful. |
| Runtime hands | `demo_*`, `qqtt/demo/`, `qqtt/env/camera/` | The code that touches cameras, GPUs, Open3D, and external models. |
| Source-of-truth docs | `AGENTS.md`, `docs/*.md`, exec plans | Small maps with links to deeper details and validation commands. |
| Mechanical guards | `scripts/harness/check_*.py`, tests | Enforce important invariants instead of leaving them as prose. |

## Demo 2.3 Failure Packet

For Demo 2.3 FPS or fused-PCD issues, start with:

```bash
conda run -n demo_2_max --no-capture-output python scripts/harness/summarize_demo23_failure_packet.py \
  --profile-json docs/generated/demo23_dual4090_no_render_profile.json \
  --output-json docs/generated/demo23_failure_packet.json \
  --output-md docs/generated/demo23_failure_packet.md
```

The packet is intentionally compact. It extracts:

- Demo 2.3 contract: pipeline, render mode, FFS batch size, TensorRT path, builderOptimizationLevel.
- Throughput: capture, raw fusion, filter, fusion, render FPS, worker periods, queue drops, stale drops.
- Same-group safety: ready joins, depth/mask wait balance, and mismatch counters when present.
- Calibration mapping: runtime serials, calibration reference serials, identity/inverted c2w debug mode, camera-center spacing.
- Calibration preflight risk: per-camera ChArUco counts and reprojection pass/fail when detection reports exist.
- Risk flags: stale calibration, identity c2w, weak calibration, temporal skew, queue drop pressure, and misleading no-render target deficit.

This is the first artifact a future agent should read before changing Demo 2.3 fusion code.

## Problem-Specific Escalation

For current experiments, use `stuffed animal` as the object and `towel` as the controller unless the user explicitly switches cases.

If Demo 2.3 fused PCD looks wrong, investigate in this order:

1. Calibration report and serial mapping.
2. Debug flags that alter fusion (`--debug-identity-c2w`, `--debug-invert-c2w`, `--debug-only-camera-idx`).
3. Temporal skew and capture-group drops.
4. Same-group join counters and depth/mask stale drops.
5. Per-camera PLY and mask overlay artifacts.
6. FFS depth validity and batch-3 TensorRT contract.

If Demo 2.3 FPS is low, investigate in this order:

1. Worker periods for FFS and EdgeTAM.
2. Raw fusion and async filter timings.
3. Render-mode-specific metrics; do not use `render_fps` as the deficit denominator for `--render-mode none`.
4. Queue drops and result collector latency.
5. GPU utilization for both 4090s.

## Mechanical Rules

- Every public harness script must be registered in `scripts/harness/_catalog.py`.
- Harness scripts should summarize existing evidence rather than depending on external checkpoints or generated images.
- Generated artifacts that establish reusable claims should be linked from a doc or exec plan.
- Hardware checks remain manual and documented; deterministic checks must not fake cameras or GPUs.
- New problem-specific harnesses should include a small unit test with synthetic JSON.
