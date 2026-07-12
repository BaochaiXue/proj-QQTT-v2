# Demo v6.2 remove runtime PCD filter

## Requirement

Demo v6.2 has one canonical PhysTwin radius-outlier mask filter in
`phystwin_strict_product.py`. The realtime `MaskedPcdPacket` path must not
carry a second configurable filter framework. Remove that runtime framework,
its point caps, fallback policies, scheduling modes, telemetry, CLI surface,
and unreachable tracker residual-filter branches.

The strict product continues to apply the original PhysTwin rule to the full
masked point set: radius `0.01 m`, at least `40` neighbors, with rejected 3D
points cleared from the processed 2D masks.

## Final behavior

- The runtime PCD path back-projects every valid masked depth pixel. It has no
  point-count cap and no runtime spatial-filter algorithm.
- The formal prepared PhysTwin product remains the only owner of PT mask
  filtering.
- Runtime CLI has no filter enable, mode, preset, per-class algorithm, cap,
  retain-ratio, fallback, async scheduling, or enhanced-filter options.
- Runtime packets and metadata contain no obsolete filter/cap telemetry.
- Tracker initialization and lifting use their direct target-mask/depth rules;
  unreachable PCD-filter residual and filtered-marker retirement logic is
  removed.

## Implementation

1. Remove runtime filter constants, CLI arguments, validation, subprocess
   flags, worker setup, budget controllers, contracts, and utility module.
2. Delete `_PcdMixin` filter methods and make PCD construction directly use
   all valid masked pixels.
3. Remove point-cap parameters and measurements from RGB-D back-projection.
4. Remove runtime filter telemetry from packets, headless writing, metadata,
   and timing.
5. Remove unreachable tracker residual-filter and filtered-marker retirement
   state.
6. Keep and directly test strict-product full-grid radius mask filtering.

## Validation

- Search for removed runtime filter identifiers and CLI options.
- Compile touched Python modules.
- Run focused Demo v6.2 tests.
- Run the repository smoke validation profile.

## Results

- Demo v6.2 runtime identifiers/options audit: no runtime filter, cap,
  retain-ratio, fallback, async-worker, or residual-marker references remain.
- `python -m compileall -q demo_v6_2`: passed.
- `python -m pytest -q tests`: 10 passed, 3 subtests passed.
- Four standalone harness guards passed: catalog, Demo v5 compatibility,
  experiment boundaries, and visual architecture.
- The required smoke profile is blocked by pre-existing deleted Demo v5.1
  files: the catalog help probe imports `demo_v5_1`, and the scope guard still
  requires five `demo_v5_1/shape_prior_*.py` paths. The failure occurs before
  any Demo v6.2-specific check and is unrelated to this change.
