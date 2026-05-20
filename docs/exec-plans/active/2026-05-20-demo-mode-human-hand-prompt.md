# Demo Mode Human Hand Prompt

## Goal

Improve Demo 3.x / Demo 3.2 live SAM3.1 first-frame initialization for a real
operator hand by using a more explicit controller text prompt: `human hand`.

## Scope

- Change demo-mode controller prompt from `hand` to `human hand`.
- Keep the controller concept as hand/human hand, and keep experiment mode
  unchanged.
- Make shared runtime controller-label detection treat `human hand` as a
  controller label.
- Update contract tests and docs.

## Validation

- Focused Demo 3 / Demo 3.1 / Demo 3.2 contract tests.
- Quick deterministic harness.
