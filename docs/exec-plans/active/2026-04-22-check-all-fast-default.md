# 2026-04-22 Check All Fast Default

## Goal

Refactor `scripts/harness/check_all.py` so the default invocation finishes in
under one minute on a healthy repo environment, while preserving an explicit
full deterministic validation path for broader coverage.

## Non-Goals

- no removal of existing full deterministic validation coverage
- no change to repo scope guard rules
- no change to camera / alignment / visualization runtime behavior
- no attempt to make hardware or external proof-of-life checks part of CI-like
  deterministic validation

## Files To Touch

- `scripts/harness/check_all.py`
- `tests/test_check_all_smoke.py`
- `AGENTS.md`
- `README.md`
- `docs/envs.md`
- `scripts/harness/README.md`
- this exec plan

## Implementation Plan

1. split the current monolithic check list into named profiles inside
   `check_all.py`
2. make the default profile `quick`, focused on:
   - core CLI import/help coverage
   - repo guards
   - a curated cross-section of smoke tests spanning camera, align, FFS
     backend, and compare visualization contracts
3. keep the old broad deterministic coverage reachable through an explicit
   `--full` flag
4. batch test modules into fewer Python invocations so the quick path does not
   spend most of its time in interpreter startup
5. add tests that pin:
   - default profile selection
   - quick vs full command planning differences
   - the presence of the full-mode escape hatch
6. update operator-facing docs to explain when to use default quick validation
   vs `--full`

## Validation Plan

- `python -m unittest -v tests.test_check_all_smoke`
- `python scripts/harness/check_all.py --help`
- `python scripts/harness/check_all.py`
- `python scripts/harness/check_all.py --full --help`

## Progress

- [x] Split quick/full profiles while preserving full deterministic coverage.
- [x] Batched quick unittest modules into one Python invocation.
- [x] Further trimmed default quick coverage to core camera/record/alignment,
  repo guards, FFS geometry contracts, and current comparison visualization
  smoke tests.
- [x] Kept experiment CLIs, realtime demo smoke tests, TensorRT/benchmark
  helpers, and pytest probe schema checks in `--full`.
- [x] Run targeted validation after the second quick-profile trim.

## Latest Validation

- `conda run -n FFS-max-sam31-rs python -m unittest -v tests.test_check_all_smoke`: passed.
- `conda run -n FFS-max-sam31-rs python scripts/harness/check_all.py --help`: passed.
- `conda run -n FFS-max-sam31-rs python scripts/harness/check_all.py`: passed in `18.70s`.
- `conda run -n FFS-max-sam31-rs python scripts/harness/check_all.py --full --help`: passed.

## Risks

- shrinking the default validation set too aggressively could let some
  non-critical regressions slip through unless contributors remember to use
  `--full` for larger changes
- batching tests into fewer invocations reduces startup overhead but also makes
  a single failure fan out over a larger command group
