# Demo v4 Atomic Chunk Publish

## Goal

Make Demo v4 FuturePhysTwin chunk cases appear consumable only after every
artifact is written, validated, and marked ready.

## Root Cause

`demo_v4.futurephystwin_chunk_writer.write_futurephystwin_chunk_case()` creates
the final case directory before writing artifacts. A base-directory watcher can
therefore discover a partial case while RGB, masks, tracking, PCD, metadata, or
`final_data.pkl` are still being materialized. The bridge also rewrites
`manifest.json` after the writer returns, so `manifest.json` is not a safe
readiness signal.

## Design

- Materialize each chunk into a staging directory outside the final case name.
- Let the bridge provide final manifest extras before the case is validated.
- Validate the staged case.
- Write `READY` last in the staged case.
- Publish by renaming the staged directory to the final case directory with
  `os.replace`.
- Treat `READY` as the consumer-facing readiness contract.
- Keep temporary publishing directories under a reserved staging root so normal
  case-name scans do not mistake them for final chunks.

## Scope

- Modify `demo_v4/futurephystwin_chunk_writer.py`.
- Modify `demo_v4/headless_chunk_bridge.py`.
- Update Demo v4 tests in `tests/test_demo_v4_futurephystwin_chunks.py`.
- Update Demo v4 docs and the PhysTwin-like contract.

## Tasks

- [x] Add failing tests proving `READY` exists on complete chunks and partial
      materialization is not visible at `<base>/<case>`.
- [x] Update the chunk writer to use staging, final manifest extras,
      validation, `READY`, and atomic rename.
- [x] Update the bridge so cadence/backlog fields are passed into the writer
      before publish instead of rewriting published manifests.
- [x] Document that FuturePhysTwin consumers must require `READY` and ignore
      staging directories.
- [x] Run focused Demo v4 tests and the repo smoke validation profile.

## Validation

- Passed: `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_demo_v4_futurephystwin_chunks.py`
- Passed: `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
