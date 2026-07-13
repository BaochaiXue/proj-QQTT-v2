# Demo v6.2 root facade cleanup

## Requirement

Keep a Python file at the `demo_v6_2/` root only when `PIPELINE.md` directly
names that `.py` file. Move every other root Python module into a responsibility
folder, without compatibility wrappers.

## Root policy

The retention whitelist is derived from every `.py` token in answers Q2-Q25 of
`demo_v6_2/PIPELINE.md`, not from the introduction or the Q1 architecture
overview and not only from Markdown links. External checkout paths and already
nested local modules do not create root files. A deterministic test will reject
any root Python file not cited by Q2-Q25.

## Planned moves

- `orchestration/`: configuration and layout helpers used by the retained
  `main_*` facade.
- `streaming/`: chunk payload and JSONL-tail internals.
- `perception/`: FFS, segmentation, tracker geometry, and SAM3.1 helpers.
- `shape_prior/`: shape-prior matching internals.
- `visualization/`: camera, timeline, renderer, and video-export helpers.

Files retain descriptive basenames inside the new packages. All imports,
script paths, docs, and tests move to the canonical package paths; no root
re-export or forwarding module is retained.

## Validation

- Import/path audit finds no old module or script references.
- The root-policy test proves every remaining root `.py` is directly named in
  `PIPELINE.md`.
- Focused Demo v6.2 tests pass.
- Repository smoke and exhaustive validation profiles pass, or any unrelated
  blocker is recorded exactly.
- `git diff --check` passes.

## Status

- [x] Moved 17 internal modules into five responsibility packages and updated
  every import/script path without adding forwarding wrappers.
- [x] Reduced the root Python facade from 55 files to the 38 files cited by
  Q2-Q25; the deterministic root-policy test reports no uncited file.
- [x] Updated `PIPELINE.md`, regenerated its 20-page PDF, and documented the
  same package contract in `readme.md`.
- [x] Focused tests pass (19 tests plus 3 subtests), the repository smoke
  profile passes (20 tests), imports compile, both executable entrypoints
  answer `--help`, all 186 local `PIPELINE.md` links resolve, and
  `git diff --check` passes. The exhaustive profile was attempted but is
  blocked before Demo v6.2 validation by the pre-existing missing entrypoint
  `demo_v5_1/main.py`.
