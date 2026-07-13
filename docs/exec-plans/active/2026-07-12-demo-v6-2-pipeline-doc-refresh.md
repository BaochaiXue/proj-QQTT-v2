# Demo v6.2 pipeline documentation refresh

## Requirement

Update `demo_v6_2/PIPELINE.md` and regenerate `demo_v6_2/PIPELINE.pdf` so both
describe the current Demo v6.2 runtime after the input-source, strict pairing,
and mixin-contract refactors. The guide now answers 24 questions, including a
dedicated Phystwin_shen startup and filesystem handoff question.

## Scope

- Revalidate all 24 pipeline answers against current source.
- Remove references to deleted packet and helper types.
- Update moved symbols and stale source line anchors.
- Document the shared mixin typing contract where it affects architecture.
- Regenerate the PDF from the final Markdown.
- Record the 2026-07-12 bounded fake-camera run and formal Phystwin launch.
- Distinguish the Phystwin combined HTML viewer from Demo pipeline-status UI.
- Separate the training read threshold from Phystwin process launch and data
  handoff mechanics.

## Validation

- Audit Markdown source links and removed symbol names.
- Verify generated PDF text contains the updated descriptions.
- Run `git diff --check` and the repository smoke profile.

## Status

- [x] Markdown synchronized with current code and 176 local links checked.
- [x] PDF regenerated as an 18-page A4 document without browser
  headers/footers; updated viewer, timing, tracking-status, and Phystwin
  handoff text verified.
- [x] Runtime proof records one completed upstream chunk and the formal
  points-ready → supervisor → HTTP-200 viewer → Stage 1 export path without
  misreporting the still-running replay/train as terminal success.
- [x] Validation completed: 18 Demo v6.2 runtime-mode tests and 3 subtests pass.
  The full smoke runner was attempted but remains blocked by the pre-existing
  `render_demo32_headless_capture.py` import of removed `demo_v5_1` code.
