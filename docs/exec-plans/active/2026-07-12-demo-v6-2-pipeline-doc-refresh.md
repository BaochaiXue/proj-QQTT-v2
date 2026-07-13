# Demo v6.2 pipeline documentation refresh

## Requirement

Update `demo_v6_2/PIPELINE.md` and regenerate `demo_v6_2/PIPELINE.pdf` so both
describe the current Demo v6.2 runtime after the input-source, strict pairing,
and mixin-contract refactors. The guide now answers 25 questions, including an
up-front process/thread architecture rationale and a dedicated Phystwin_shen
startup and filesystem handoff question.

## Scope

- Revalidate all 25 pipeline answers against current source.
- Remove references to deleted packet and helper types.
- Update moved symbols and stale source line anchors.
- Document the shared mixin typing contract where it affects architecture.
- Regenerate the PDF from the final Markdown.
- Record the 2026-07-12 bounded fake-camera run and formal Phystwin launch.
- Distinguish the Phystwin combined HTML viewer from Demo pipeline-status UI.
- Separate the training read threshold from Phystwin process launch and data
  handoff mechanics.
- Document the complete process tree, camera-thread topology, non-thread
  workers, and the design tradeoff at every concurrency boundary.

## Validation

- Audit Markdown source links and removed symbol names.
- Verify generated PDF text contains the updated descriptions.
- Run `git diff --check` and the repository smoke profile.

## Status

- [x] Markdown synchronized with current code and 186 local links checked.
- [x] PDF regenerated as a 20-page A4 document without browser
  headers/footers; updated process tree, concurrency rationale, viewer, timing,
  tracking-status, and Phystwin handoff text verified.
- [x] Runtime proof records one completed upstream chunk and the formal
  points-ready → supervisor → HTTP-200 viewer → Stage 1 export path without
  misreporting the still-running replay/train as terminal success.
- [x] Validation completed: the focused 18 Demo v6.2 runtime-mode tests and 3
  subtests pass. The full repository smoke profile also passes, including 19
  Demo v6.2 unit tests, guards, and CLI help probes.
