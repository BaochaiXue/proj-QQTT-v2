# Demo 3.2 LiteTracker Batch=3

## Goal

Make Demo 3.2 use LiteTracker camera-view batch execution by default, with
three synchronized camera views treated as the batch dimension. The default
path is strict `batch-views`; operators can still explicitly choose `auto` or
`serial` for diagnostics.

## Plan

1. Mark LiteTracker as an experimental batch-view capable backend.
2. Add `LiteTrackerAdapter.initialize_batch()` and `update_batch()` using one
   LiteTracker model state for `B=3` camera views.
3. Default Demo 3.2 to `batch-views` tracking execution with `batch` update
   mode so LiteTracker runs camera-view batch=3 by default.
4. Use `min-common` query count policy for Demo 3.2 batch safety.
5. Update Demo 3.2 contract/docs/tests to report `litetracker_batch3`.
6. Patch LiteTracker's online forward at adapter load time so B>1 previous
   frame buffers keep their time axis and do not broadcast from `[B,1,N,2]`
   into `[B,B,N,2]` on the second update.

## Validation

- Focused adapter tests for batch camera ordering, xy/yx conversion, unequal
  query rejection, and fake-model shapes.
- Real LiteTracker smoke with `B=3`, `N=4096`, two updates, matching the live
  crash shape.
- Demo 3.2 dry-run/contract tests.
- `scripts/harness/check_all.py` quick profile if the focused tests pass.
