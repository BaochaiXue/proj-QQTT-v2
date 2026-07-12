# Demo v6.2 remove source-headless-capture mode

## Requirement

Demo v6.2 now supports only the `fake-live` and `live` input modes. Remove the
deprecated `--source-headless-capture` completed-capture conversion path instead
of retaining a compatibility branch.

## Scope

- Remove the CLI option and dry-run contract field.
- Remove the lower camera CLI's deprecated `recording` input alias and
  `--recording-case` compatibility option.
- Remove the offline branch from the orchestrator.
- Remove the now-unreachable completed-capture conversion entrypoint and its
  dead JSONL reader.
- Require RGB-D frames to be streamed before their chunk commits; remove the
  batch-write fallback that existed only for completed-capture conversion.
- Make startup cleanup unconditionally clear the generated capture directory.
- Update Demo v6.2 pipeline documentation and regression coverage.
- Keep the realtime headless capture directory and
  `stream_chunk_data_from_headless_capture`; both `fake-live` and `live` use
  that internal producer/consumer boundary.
- Do not change frozen Demo v6.1 or older demo versions.

## Invalid input behavior

Passing `--source-headless-capture` must fail at argument parsing as an
unrecognized option. Any `--input-source` value other than `fake-live` or
`live` must fail at argument parsing. `--recording-case` must also fail as an
unrecognized compatibility option; fake-live uses `--fake-live-case`.

## Validation

- Focused parser regression tests for the two supported modes and the removed
  option.
- Import/compile checks for touched Demo v6.2 modules.
- Static reference audit for the removed option and Python attribute.
- One-chunk fake-live end-to-end run proving that five streamed RGB-D frames
  commit without the removed batch archive fallback.
- Repository smoke validation profile.

## Status

- [x] Implementation complete.
- [x] Focused validation passes: three regression tests pass, including the
  stream-before-commit archive contract.
- [x] One-chunk fake-live validation passes: one normal five-frame chunk was
  published with five RGB images and five depth arrays.
- [x] Repository smoke profile passes on 2026-07-12.
