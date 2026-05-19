# Demo Services Refactor Phase 1-3

## Goal

Introduce shared demo services so Demo 2.3 and Demo 3.1 stop manually copying
profile, latest-wins, render, and object volume filtering behavior. This phase
extracts service APIs around existing behavior first; it does not change live
demo semantics.

## Constraints

- Demo 3.0 and Demo 3.1 stay online-only.
- Demo 3.1 keeps GPU0 main runtime and GPU1 CoTracker process ownership.
- Demo 3.1 IPC remains CPU latest-wins; no CUDA tensor transfer.
- Demo 3.0 and Demo 3.1 keep FuturePhysTwin dense tracking semantics.
- Demo 2.3 keeps FFS + EdgeTAM split semantics.
- Old demo entrypoints remain in place.

## Phase Scope

1. Add `qqtt/demo/services/profile_schema.py` for shared profile keys and
   Demo 3.1 empty summary construction.
2. Add `qqtt/demo/services/latest_wins.py` and route Demo 3.1 IPC through it.
3. Add `qqtt/demo/services/object_volume_filter_service.py` and route Demo 2.3
   object phystwin-volume filtering through it.
4. Add a first `render_pcd_service.py` facade around existing latest render
   buffer and render micro-profile helpers without moving the Open3D loop yet.

## Validation

- Focused service tests for profile schema, latest-wins, object volume service,
  and render service.
- Focused existing Demo 2.3 / Demo 3.1 contract tests.
- Full `check_all.py` after this branch stabilizes.
