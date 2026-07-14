# Demo v6.2 metadata contract cleanup

## Requirement

Remove metadata fields that are emitted by Demo v6.2 but are not read for
runtime semantics by either Demo v6.2 or the configured read-only
`/home/xinjie/Phystwin_shen` checkout. Do not modify Phystwin_shen.

## Consumer boundary

- Capture metadata retains only fields consumed by chunk streaming, strict
  product finalization, visualization, shape-prior readiness, or online-case
  calibration publication.
- Shape-prior case metadata retains the fields read by the alignment path or
  required by Phystwin's case loader.
- Online camera metadata retains `intrinsics`, `WH`, `fps`, `frame_num`, and
  `serial_numbers`, because Phystwin reads those fields.
- Profiling files, run summaries, JSONL frame rows, chunk manifests, and
  final-data payloads are outside this cleanup; they have separate consumers
  and contracts.

## Changes

- Delete the table-Z descriptor constants and all metadata-only coordinate,
  filter-policy, model-policy, and provenance fields with zero readers.
- Delete the EdgeTAM metadata dictionary that is only formatted and printed.
- Remove hidden CLI flags whose only effect is populating deleted identity
  metadata.
- Remove the dead shape-prior table-Z request field and dead case metadata,
  while retaining source/depth fields read by the inspection tool and source
  sequence/timestamp fields used by warmup profiling.
- Remove unused descriptive keys from online metadata while preserving every
  key read by Phystwin_shen.
- Add focused tests for the retained metadata contracts.

## Validation

- Audit deleted field names against Demo v6.2 and Phystwin_shen readers.
- Compile all touched Demo v6.2 modules.
- Run focused Demo v6.2 tests and the repository smoke profile.
- Run `git diff --check` and confirm Phystwin_shen has no modifications.

## Status

- [x] Capture metadata reduced to fields with concrete Demo v6.2 readers.
- [x] Shape-prior case metadata keeps the PhysTwin/alignment fields and the
  three provenance fields used by the Demo inspection report; dead table-Z and
  duplicated source timing fields are gone.
- [x] Online camera metadata reduced to the five fields consumed by Demo or
  Phystwin_shen; enhance metadata reduced to its consumed mapping fields.
- [x] Metadata-only EdgeTAM logging, hidden identity CLI options, and duplicated
  static/manifest identity labels removed.
- [x] Focused 22-test suite, Ruff, compileall, repository smoke profile,
  `git diff --check`, and regenerated 21-page A4 pipeline PDF all pass.
- [x] `/home/xinjie/Phystwin_shen` remains unmodified at
  `0441dc607796724a02fbdb5aecb450aa52bd5aa6`.
