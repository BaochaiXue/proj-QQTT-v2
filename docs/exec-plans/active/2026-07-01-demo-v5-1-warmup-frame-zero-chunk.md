# Demo v5.1 Warmup Frame-Zero Chunk Timeline

**Goal:** Put the warmup data-process frame back into the formal online chunk
timeline as frame 0 of `online_data/chunks/chunk_000000.pkl`.

## Requirement

Demo v5.1 warmup produces the first usable data-process frame. That frame must
be part of the same online final-data stream consumed by chunks and aggregate
`data/final_data.pkl`.

Current code writes the warmup frame as `chunk_warmup.pkl` with `chunk_id=-1`
and starts formal chunks from the first post-warmup row. That makes chunk frame
0 differ from the actual process frame 0.

## Planned Changes

- Remove the warmup sidecar chunk writer and constants.
- Keep delayed warmup rows in the rows passed to chunking.
- Start chunk manifests from `chunk_index=0`.
- Keep `source_frame_indices` and timestamps as source metadata while
  `start_frame`/`end_frame` remain contiguous online frame indices.
- Update docs and tests to assert warmup is frame 0 inside `chunk_000000.pkl`.

## Validation

- `python -m pytest tests/test_demo_v5_1_chunk_data.py`
- `python scripts/harness/validation/run.py --profile smoke`
- `git diff --check -- demo_v5_1 tests docs`
