# Demo v6.2 pipeline documentation refresh

## Requirement

Update `demo_v6_2/PIPELINE.md` and regenerate `demo_v6_2/PIPELINE.pdf` so both
describe the current Demo v6.2 runtime after the input-source, strict pairing,
and mixin-contract refactors.

## Scope

- Revalidate all 23 pipeline answers against current source.
- Remove references to deleted packet and helper types.
- Update moved symbols and stale source line anchors.
- Document the shared mixin typing contract where it affects architecture.
- Regenerate the PDF from the final Markdown.

## Validation

- Audit Markdown source links and removed symbol names.
- Verify generated PDF text contains the updated descriptions.
- Run `git diff --check` and the repository smoke profile.

## Status

- [x] Markdown synchronized with current code and 168 local links checked.
- [x] PDF regenerated as A4 without browser headers/footers; updated text verified.
- [x] Validation completed: the full smoke runner was attempted but is currently
  blocked by pre-existing references to removed `demo_v5_1` files in
  `render_demo32_headless_capture.py` and the scope guard. All other reachable
  smoke guards pass, and the seven Demo v6.2 regression tests pass.
