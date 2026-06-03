Objective: render tracked object and controller points at the same time with distinct overlay colors.

Scope:
- Keep Demo 3.1/3.3 tracker inputs, masks, lifting, and shape prior data flow unchanged.
- Use existing query label metadata from the point-tracker worker to color rendered tracker points.
- Preserve camera-debug coloring as an explicit override.
- Make Demo 3.3 default to `--overlay-display-scope union` while respecting explicit user overrides.
- Add focused tests for semantic colors and profile/contract fields.

Validation:
- Run focused Demo 3.1 overlay contract tests.
- Run the quick harness check if the focused tests pass and time allows.
