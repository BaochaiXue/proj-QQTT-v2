# Demo 3.3 Query Overlay Enhanced Filter

## Objective

Filter Demo 3.3 all-tracks/query overlay points with the same enhanced-PT spatial cleanup used by the rendered PCD, so visible tracked object/controller markers do not show isolated lifted outliers.

## Scope

- Apply only to render overlay points after depth lift and before marker expansion.
- Apply a current-frame object/controller semantic mask gate before depth lift so all-tracks
  query markers outside the active mask are not rendered as isolated speckles.
- Preserve tracker inputs, query counts, backend outputs, shape prior generation, and PCD fusion settings.
- Use semantic labels to filter object and controller query overlays with their existing component policies.
- Add profile fields for semantic mask gate and 3D reference filter timing plus per-camera kept/rejected counts.
- Add focused tests and run the deterministic harness.

## Verification

- Focused unit tests for all-tracks overlay filtering.
- `check_all.py` in the repo default environment.
- A Demo 3.3 live rerun to confirm profile fields and rendered behavior.
