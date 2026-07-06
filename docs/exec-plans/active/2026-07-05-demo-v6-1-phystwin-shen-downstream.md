# Demo v6.1 downstream.mode + Phystwin_shen Auto-Launch

## Requirement

Problem:
Phystwin_shen online training (`train_online_warp.py`) and its HTML viewer
(`scripts/html_realtime_viewer.py`) were launched manually after a demo run.
They should start automatically once the shape prior is ready and the second
GPU is freed, as a config-selectable alternative to the demo visualizer.

Required final behavior (user decision, 2026-07-05):
- Config gains an explicit enum `downstream.mode: disabled | demo_visualizer
  | phystwin_shen`, replacing the visualizer-only design (`visualizer_mode`).
- phystwin_shen mode always launches BOTH the trainer and the HTML viewer.
- Trigger: shape-prior ready; `train_online_warp.py` keeps waiting for the
  first chunk on its own.
- Subprocesses get `CUDA_VISIBLE_DEVICES=1` (configurable) and
  `--device cuda:0`.
- Phystwin_shen repo path and conda env live in YAML; defaults
  `/home/xinjie/Phystwin_shen` and `demo_2_max`.
- Viewer binds 127.0.0.1:8765; an occupying process is killed; kill failure
  fails fast.

State changes:
- `demo_v6_1/config/default.yaml`: new `downstream:` + `phystwin_shen:`
  sections, `gpu.phystwin_shen_cuda_visible_devices`, `visualizer_mode`
  removed.
- New `demo_v6_1/phystwin_shen_launch.py`: settings dataclass, command
  builders mirroring the manual script, psutil-based `ensure_port_free`
  (SIGTERM -> SIGKILL -> verify, fail fast), launch + summary record.
- `demo_v6_1/main.py`: `--downstream-mode` (+ phystwin overrides),
  `resolve_downstream_mode` runtime enum validation, launch trigger polled
  via the stream's `before_poll` (fires once when `shape_prior/points.npz`
  exists; immediately when warmup is off; `on_chunk_written` as safety net),
  run-summary `phystwin_shen_*` fields, exit-code policy.
- `demo_v6_1/online_frame_archive.py` + `chunk_data_stream.py`:
  `initialize_case` seeds `calibrate.pkl`/`metadata.json` (frame_num=0) at
  capture-metadata time because both Phystwin_shen tools read the case dir
  at startup, before the first chunk commits.

Invalid cases:
- Unknown downstream mode, missing repo/scripts, empty conda env or GPU
  namespace, invalid port -> fail at `validate_runtime_args`.
- Port occupied by unkillable/unidentifiable process -> fail fast at launch.

Constraints:
- chunks/manifest contracts unchanged; demo_visualizer behavior preserved
  (side-by-side / output-only start policies, RGB-timeline gating).
- Launched processes are left running at demo exit (viewer-window policy);
  their exit status is recorded and non-zero codes propagate.

## Plan

- [x] Config: downstream/phystwin_shen sections, gpu namespace key.
- [x] `demo_v6_1/phystwin_shen_launch.py`.
- [x] `main.py` wiring (CLI, validation, trigger, summary, exit codes).
- [x] Case-dir seeding at capture-metadata time.
- [x] Tests (`tests/test_demo_v6_1_downstream.py`) + harness registration.
- [ ] End-to-end fake-live run with `--downstream-mode phystwin_shen`;
  adversarial review; commit/push.

## Validation

- `python -m pytest tests/test_demo_v6_1_downstream.py -q`
- `python -m pytest tests/ -q`
- `python scripts/harness/validation/run.py --profile smoke`
- End-to-end: occupy port 8765, run demo with phystwin_shen mode, verify
  the occupant is killed, viewer serves, trainer trains on GPU 1, summary
  fields recorded.
