# Demo v6.2 Phystwin_shen CLI contract sync

> Historical plan: superseded by
> `2026-07-09-demo-v6-2-full-pipeline-integration.md`. The current Demo no
> longer launches trainer/viewer directly; it launches one full-pipeline
> supervisor with explicit local-config overrides.

## Requirement

Problem:
`~/Phystwin_shen` now follows `origin/online@5b8c071`. Demo v6.2 still sends
arguments from the previous trainer/viewer CLI, so both downstream processes
exit during argument parsing. The YAML keys also add `train_` / `viewer_`
prefixes that do not match Shen's argparse destinations.

Required final behavior:
- Demo v6.2 launches the current `train_online_warp.py` and
  `scripts/html_realtime_viewer.py` with only supported arguments.
- Existing configurable YAML leaf keys use Shen's argparse names directly.
- The viewer receives the required RGB directory produced by Demo v6.2.
- Capture, tracking, ASAP, chunk, and `data/final_data.pkl` products do not
  change.

Inputs:
- Demo runtime `base_path` and `phystwin_shen` YAML section.
- The external checkout at `phystwin_shen.repo_path`.

Outputs:
- Trainer command reading `<base_path>/online_data`.
- Viewer command reading camera metadata and RGB from
  `<base_path>/online_data` and training snapshots from the existing explicit
  realtime directory.

State changes:
- Rename the existing `phystwin_shen` YAML option keys.
- Update settings resolution and both command builders.
- Update tests and operator-facing design documentation.

Invalid cases:
- Missing external entry scripts, empty GPU namespace/environment, or invalid
  viewer port keep failing before launch.
- Unsupported/missing external CLI arguments must be caught by command
  validation against the current Shen checkout.

Constraints:
- Work only on `single-camera` and preserve all existing uncommitted Demo v6.2
  refactor/camera-serial changes.
- Do not add a legacy CLI path or pass the removed static-data argument.
- Keep external repos and weights outside this repository.

Unknowns:
- None. The current upstream parser contract was inspected at `5b8c071`.

## Plan

- [x] Rename YAML leaves to `host`, `port`, `cam_idx`, `point_mode`,
  `point_stride`, `device`, `batch_size`, `segment_len`, `segment_stride`,
  `poll_sec`, `recent_window_count`, `realtime_vis_every`, and
  `stop_when_finished`; remove `viewer_image_index_mode`.
- [x] Remove trainer `--base_path`, `--case_name`, and `--static_data_path`.
- [x] Remove viewer `--image_index_mode`; add
  `--rgb_dir <base_path>/<case_name>/color`.
- [x] Update focused tests and documentation.
- [x] Validate generated flags against the real Shen parsers, then run the
  Demo v6.2 downstream tests and repository smoke profile.

## Validation

- Branch preflight: `single-camera` equals `origin/single-camera`. The mandated
  `git pull --ff-only origin main` was attempted and refused because the two
  branches have diverged (`HEAD...origin/main = 287/1`); no merge/rebase was
  performed.
- `conda run -n demo_2_max --no-capture-output python -m pytest \
  tests/test_demo_v6_2_downstream.py -q`
- Static generated-command flag comparison against the argparse declarations
  in `~/Phystwin_shen`.
- Bounded real parser probes for both generated commands (no camera required).
- `conda run -n demo_2_max --no-capture-output python \
  scripts/harness/validation/run.py --profile smoke`

Results:
- Focused downstream tests: `18 passed`.
- Static upstream argparse comparison: trainer generated 12 supported flags
  and supplied required `--online_dir`; viewer generated 9 supported flags and
  supplied required `--case_name` and `--rgb_dir`.
- Real bounded probes: viewer served `/api/manifest` from the existing output;
  trainer accepted the command and reached the intentional zero-chunk runtime
  boundary without an argparse error.
- Ruff formatting passed for all touched Python files. Scoped Ruff checks passed
  with the pre-existing `main.py` re-export/bootstrap `E402/F401` categories
  excluded.
- Repository smoke profile: `196 tests`, all guards/help checks passed.
