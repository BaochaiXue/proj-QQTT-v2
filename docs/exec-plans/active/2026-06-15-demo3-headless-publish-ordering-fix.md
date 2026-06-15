# Demo 3.x Headless Lossless Publish Ordering Fix

## Objective

Fix full headless Demo 3.2 lossless capture failures where artifact writing
holds the pairer/publish lock long enough for `mask-tracker` backlog to exceed
the strict 15-frame bound.

## Design

- Keep the strict same-seq invariant and 15-frame backlog fatal behavior.
- Split pairer submission from ordered pair publication:
  - pairer lock is held only while adding/closing PCD or tracker results,
  - complete pairs are placed into an ordered pair-output queue,
  - a dedicated output worker performs headless artifact writes and viewer
    publication,
  - a separate condition variable enforces ordered publication by `seq`.
- Do not add CLI flags and do not disable tracker/PCD/filter work.
- Allow the lossless controller filter budget to reduce its input cap below the
  normal 5000-point floor. Object filtering keeps the existing floor; controller
  output can still fall back to raw current-frame points when retain-ratio checks
  would otherwise hide the controller.

## Validation

- Add a regression test proving slow headless publication does not hold the
  pairer submission lock and still publishes pairs in order.
- Add coverage that the lossless controller filter budget can reduce below the
  default minimum cap.
- Run targeted unit tests.
- Run the full fake-live headless capture and render both PCD and tracking
  videos from the resulting artifacts.
