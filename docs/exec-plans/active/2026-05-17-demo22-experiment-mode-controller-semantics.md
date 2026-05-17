# Demo 2.2 Experiment Mode Controller Semantics

## Goal

Make the controller semantic explicit instead of relying on an implicit
`--controller-prompt` convention.

## Contract

Two modes are supported:

- `controller-object-exp`: current lab experiment mode; controller is `towel`.
- `demo-mode`: formal live demo mode; controller is `hand`.

Demo 2.2 current experiment presets keep the existing default behavior and
resolve to `controller-object-exp` unless the operator explicitly asks for
`demo-mode`.

## Validation

- Public Demo 2.2 CLI exposes and forwards `--experiment-mode`.
- Runtime contract/profile JSON records the mode, expected controller semantic,
  actual controller prompt, and whether they match.
- Smoke tests cover both modes.
