# Demo v7 fake-live execution and failure-reporting fixes

## Environment and inputs

- Branch: `main`, initially `bf6033a`, clean and up to date after
  `git pull --ff-only origin main`.
- Python: `conda run -n demo_2_max --no-capture-output`; two RTX 4090 GPUs.
- Case: `data_collect/sloth_new_20260705_230611`, 5,658 recorded RGB-D frames,
  recorded at 30 FPS and sampled at the intended 5 FPS output cadence.
- Defaults: native RealSense depth from recording, TRELLIS.2, SD upscale,
  TripoSplat. Mesh generation was fresh; no canonical mesh cache was used.
- A concurrent local session committed mesh-surface changes as `51fbb32`
  during this task, also including our initial plan and three new regressions.
  The final suite includes those concurrent tests; their implementation is
  outside this task's changes.

## Reproduced and fixed

1. A parent-process chunk exception never reached the GUI and left capture
   running without its consumer. It now emits the existing `chunk_stream`
   error event, requests formal stop, and stops the downstream process group.
2. `wait_for_state(FINISHED)` returned success before checking a latched chunk
   error. Failures now take precedence over a matching target state.
3. Fatal parent/link errors could leave the GUI tracking or later show a clean
   finish. The GUI now enters its failure screen for terminal chunk/link
   errors and retains that screen across late FINISHED/HELLO messages. Link
   failures also latch a failed run status instead of disappearing at shutdown.

The initial three chunk regressions failed before the correction. Final
coverage also checks both link sources, late HELLO state, and the persisted
failure status. No recording/alignment or tracking algorithm was changed.

## Actual default GPU drive

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v7/tests/drive_fake_live.py \
  --base-path outputs/v7debug0912a \
  --formal-after-wrap-s 1 --target-chunks 10 \
  > /tmp/v7debug0912a.log 2>&1
```

PASS, exit 0. Completed PREVIEW, capture/confirm, masks, fresh shape-prior
generation/alignment/sampling, Gaussian generation and seed-123 re-roll,
REVIEW, replay wrap/reposition, FORMAL, stop/drain, FINISHED, and shutdown.

- REVIEW after approximately 66 seconds; seed re-roll complete at 83 seconds.
- Formal start at 192 seconds after waiting for the recording to wrap back to
  the captured pose; this is intentional continuous fake-camera playback.
- 11 committed chunks and `shape_prior/points.npz`.
- 57 contiguous formal frame sequence numbers; 71 unique input timeline frame
  numbers; source indices strictly increase; every prepared frame exists.
- Strict manifest: 57 prepared frames and 5,000 tracking queries.
- 55 received Gaussian frames; 56 processed steps; rest seed present, no
  render failure; bone-to-splat median distance 0.30 cm, p90 0.86 cm.
- Self-alignment selected its improved result: silhouette IoU 0.8245 to 0.8877.
- Last pipeline status: `run_finished`, `ok=true`, `chunk_count=11`.
- This existing driver intentionally disables PhysTwin; the GUI run below
  separately exercises the configured downstream.

## Qt GUI and downstream

The scripted driver uses the actual `AppController`, `MainWindow`, Qt event
loop, real button `.click()` calls, and the normal GPU service. It saves screen
captures in the run directory. Only the Qt display platform is offscreen;
models, image rendering, IPC, tracking, and downstream processes are real.
The local helper is preserved at `outputs/v7debug0912_gui_driver.py`.

```bash
QT_QPA_PLATFORM=offscreen \
  conda run -n demo_2_max --no-capture-output \
  python /tmp/v7debug0912_gui.py > /tmp/v7debug0912gui2.log 2>&1
```

The first temporary GUI driver incorrectly waited for `gaussian_prepared.npz`
(the actual preview is a PNG). That diagnostic assertion was corrected before
the fresh `outputs/v7debug0912gui2` run. Interrupting the first service also
confirmed that terminal disconnects only appeared in the status bar before
the link-error GUI fix. That interrupted diagnostic is not counted as a pass.

Fresh GUI run: PASS, exit 0, after 222 seconds. Captured, retook, confirmed,
reviewed masks/mesh artifacts, repositioned after wrap, started formal,
clicked stop, and reached FINISHED. Produced 26 chunks; screenshots include
`gui_review.png`, `gui_formal.png`, and `gui_finished.png`. Visually inspected
the review and formal screenshots: RGB/depth dock and tracking composite
render correctly.

PhysTwin's default pipeline launched, read the actual online camera metadata
and chunks, ran its two CMA iterations, wrote `optimal_params.pkl`, and
started Train-Online with 26 chunks / 130 frames. This verifies the live
handoff; training was then stopped by normal Demo teardown, not run to
training completion. The detailed log is
`outputs/v7debug0912gui2/phystwin_shen/online_full_pipeline.log`.

The GUI run has 131 contiguous formal frames. Its final status is
`run_finished`, `ok=true`, `chunk_count=26`.

## EOF and shape-prior disabled

Created a short case from the first 300 frames of the same recording, without
modifying its original metadata or images:

```python
import json
from pathlib import Path

source = Path("data_collect/sloth_new_20260705_230611").resolve()
case = Path("outputs/v7debug0912case")
case.mkdir(exist_ok=False)
metadata = json.loads((source / "metadata.json").read_text())
recording = metadata["recording"]["0"]
metadata["recording"]["0"] = dict(
    sorted(recording.items(), key=lambda item: int(item[0]))[:300]
)
(case / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
for stream in metadata["streams_present"]:
    (case / stream).symlink_to(source / stream, target_is_directory=True)
```

```bash
V7_TEST_EOF=1 QT_QPA_PLATFORM=offscreen \
  conda run -n demo_2_max --no-capture-output \
  python outputs/v7debug0912_gui_driver.py > /tmp/v7debug0912eof.log 2>&1
```

PASS, exit 0, after 32 seconds. The actual GUI captured/retook/confirmed,
skipped shape prior and Gaussian generation, reviewed observed points,
waited for a wrap, and started formal tracking. Playback ended naturally;
the GUI displayed its replay-ended dialog and FINISHED screen. The driver
acknowledged that dialog. Produced 32 contiguous formal frames and 6 chunks;
no `points.npz`, as required with shape prior disabled. Last pipeline status:
`run_finished`, `ok=true`, `chunk_count=6`.

After teardown, process inspection found no remaining Demo camera service or
PhysTwin pipeline supervisor from these runs.

## Deterministic checks

```bash
conda run -n demo_2_max --no-capture-output \
  python -m pytest demo_v7/tests -q
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/validation/run.py --profile smoke
git diff --check
```

- Demo v7: 159 passed, including seven new failure-reporting regression cases
  (including the expected disconnect race during intentional shutdown).
- Repository smoke: passed, including 66 unittest cases and all configured
  CLI/architecture/scope guards.
- No real camera hardware check was performed; all camera input in this
  report is recorded RGB-D replay.
