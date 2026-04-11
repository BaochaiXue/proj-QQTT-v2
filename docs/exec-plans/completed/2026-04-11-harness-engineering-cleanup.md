# Harness Engineering Cleanup

## Goal

Clean up the current harness-engineering surface so it is easier to navigate and maintain:

- archive stale active exec plans
- merge or replace redundant generated harness docs
- remove obvious garbage/cache files
- fix any references that break after cleanup

## Non-Goals

- no redesign of the camera/FFS pipelines
- no behavior change to recording, alignment, or visualization outputs
- no deletion of user data under `data/` or `data_collect/`

## Audit Targets

- `scripts/harness/`
- `docs/generated/`
- `docs/exec-plans/active/`
- harness references in `README.md`, `docs/WORKFLOWS.md`, `docs/ARCHITECTURE.md`, and `AGENTS.md`

## Planned Actions

1. Identify stale exec plans in `docs/exec-plans/active/` and move completed ones to `completed/`.
2. Consolidate redundant generated harness docs into a smaller, current set.
3. Remove cache/junk files such as `__pycache__` under harness/visualization surfaces.
4. Update docs/references so the cleaned layout is the new source of truth.
5. Run targeted validation after cleanup.

## Validation

- `python scripts/harness/check_scope.py`
- `python scripts/harness/check_visual_architecture.py`
- `python scripts/harness/check_all.py` if cleanup touches test-checked references

## Risks

- deleting a file still referenced by docs/tests
- moving active exec plans without preserving history
- over-cleaning generated docs that still carry unique information
