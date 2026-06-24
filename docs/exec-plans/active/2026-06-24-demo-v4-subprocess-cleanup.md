# Demo v4 Subprocess Cleanup Exec Plan

## Goal

Ensure Demo v4 always stops the spawned Demo 3.2 process when realtime chunk
streaming or chunk finalization raises an exception.

## Evidence

- Branch confirmed: `single-camera`.
- `git pull --ff-only origin main` completed with "already up to date".
- Current `demo_v4/realtime_futurephystwin_chunks.py` starts Demo 3.2 with
  `subprocess.Popen(...)`, then calls `stream_chunks_from_headless_capture(...)`,
  then calls `_stop_process(process)` only on the normal return path.

## Design

1. Add a focused regression test that makes `stream_chunks_from_headless_capture`
   raise after `Popen` succeeds.
2. Assert `_stop_process(process)` is still invoked exactly once and the original
   streaming exception is propagated.
3. Fix the realtime path with `try/finally` around streaming. Keep existing
   summary/return behavior unchanged for successful streaming.

## Validation

```bash
conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v4_futurephystwin_chunks.FuturePhysTwinChunkWriterTest.test_demo_v4_stops_demo32_process_when_streaming_raises
conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v4_futurephystwin_chunks
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke
```

## Results

- RED confirmed: the new streaming-exception regression test failed before the
  fix because `_stop_process` was called zero times.
- PASS: focused regression test after wrapping streaming in `try/finally`.
- PASS: `tests.test_demo_v4_futurephystwin_chunks` ran 38 tests.
- PASS: `python -m py_compile demo_v4/realtime_futurephystwin_chunks.py
  tests/test_demo_v4_futurephystwin_chunks.py`.
- PASS: `git diff --check`.
- PASS: `scripts/harness/validation/run.py --profile smoke` ran 301 tests.
