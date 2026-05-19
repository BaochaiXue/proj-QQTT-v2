# Demo 3.1 PCD RGB Color

## Goal

Restore Demo 3.1 rendered point-cloud coloring so the shared live runtime uses
live RGB colors by default instead of inheriting the Demo 2.1.5 fast-native
class-color preset.

## Constraints

- Do not change CoTracker query semantics, dense sampling, overlay caps, or GPU
  ownership.
- Keep debug camera-color mode available as an explicit diagnostic override.
- Keep the shared runtime preset reuse, but make Demo 3.1 color intent explicit
  in its own CLI/contract and forwarded shared-runtime argv.

## Plan

- Add a Demo 3.1 `--pcd-color-mode {rgb,class}` CLI option with default `rgb`.
- Include the resolved PCD color mode in the Demo 3.1 contract/dry-run output.
- Forward `--pcd-color-mode` to the shared runtime so preset defaults cannot
  silently change RGB point coloring.
- Add deterministic tests covering the default and explicit class override.

## Validation

- Focused Demo 3.1 contract tests.
- Touched-module `py_compile`.
- Quick deterministic harness when the focused checks pass.
