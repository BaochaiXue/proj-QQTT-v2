# Demo 2.2 Controller-Only Mode

## Goal

Add a first-class `controller-only` tracking mode for demos that need to isolate the controller layer without tracking or rendering the object layer.

## Scope

- Add shared track-mode semantics in the masked EdgeTAM runtime helpers.
- Expose `controller-only` in Demo 2.1 and Demo 2.2 CLIs.
- Make SAM3.1/saved-mask initialization request only the active semantic layers.
- Keep fused PCD layers filtered separately and omit inactive layers from the semantic contract.
- Add smoke tests for CLI translation, active object ids, semantic layers, and dry-run contracts.

## Validation

- Focused unit tests for Demo 2.1 / Demo 2.2 track-mode contracts.
- Py-compile touched demo modules.
- Full harness can be run after the mode-level smoke tests pass.
