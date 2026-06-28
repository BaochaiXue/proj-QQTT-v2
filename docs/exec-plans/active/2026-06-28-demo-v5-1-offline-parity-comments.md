# Demo v5.1 Offline-Parity Comments

Goal: annotate Demo v5.1 realtime data-process code with explicit references to
the matching offline `data_process_sam3d` script ranges.

Scope:
- Add comments only; do not change runtime behavior, output layout, or schema.
- Prefer comments next to the realtime stage that performs the equivalent work.
- Use `data_process_sam3d/<file>.py:Lx-Ly` references for traceability.

Validation:
- Run Python compile checks for edited modules.
- Inspect the diff to ensure no behavior changes were introduced.

Results:
- Added `Offline parity` comments next to Demo v5.1 realtime stages and the
  shared strict-product helpers they call.
- The touched files already contained unrelated uncommitted non-comment
  changes in this worktree; this pass leaves those changes in place.
- `conda run -n demo_2_max --no-capture-output python -m py_compile ...`
  passed for the edited Demo v5.1/shared modules.
- `git diff --check` passed for the edited Demo v5.1/shared modules.
- `conda run -n demo_2_max --no-capture-output python
  scripts/harness/validation/run.py --profile smoke` reached unittest but
  failed because the active conda environment prepends
  `/home/xinjie/proj-QQTT-v2` to `PYTHONPATH`, causing old tests from that tree
  to import modules from this checkout.
