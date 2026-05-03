# Local Smoke Artifact Cleanup

Date: 2026-05-03

## Scope

Removed local smoke-test artifacts and generated caches only. Deterministic
smoke test source files under `tests/` were preserved.

## Removed Generated Docs

- `docs/generated/hf_edgetam_streaming_processor_session_smoke.txt`
- `docs/generated/hf_edgetam_streaming_realcase_previous_mask_smoke_benchmark.md`
- `docs/generated/hf_edgetam_streaming_realcase_previous_mask_smoke_quality.json`
- `docs/generated/hf_edgetam_streaming_realcase_previous_mask_smoke_results.json`
- `docs/generated/hf_edgetam_streaming_realcase_smoke_benchmark.md`
- `docs/generated/hf_edgetam_streaming_realcase_smoke_quality.json`
- `docs/generated/hf_edgetam_streaming_realcase_smoke_results.json`
- `docs/generated/hf_edgetam_streaming_smoke.txt`

## Removed Result Roots

- `result/hf_edgetam_streaming_realcase_previous_mask_smoke`
- `result/hf_edgetam_streaming_realcase_smoke`

## Removed Local Cache

- `tests/__pycache__/*smoke*.pyc`

## Preserved

- `tests/test_*_smoke.py`

These are source tests, not local result artifacts. They remain part of the
deterministic harness contract.

## Post-Cleanup Check

This scan returned no local smoke-named artifacts in generated/result/cache
locations:

```bash
find docs/generated result tests/__pycache__ -iname '*smoke*' -print
```

The source test inventory still reports 120 smoke test files under `tests/`.
