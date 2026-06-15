# Harness Engineering Restructure Design

## Purpose

Restructure `scripts/harness/` into a catalog-driven validation system that is
easier to reason about, extend, and verify. The current harness surface already
has a catalog and guards, but the directory layout, validation runner, unit-test
lists, and public documentation have grown from historical needs. The redesign
makes the catalog the single machine-readable source of truth, makes directory
layout express lifecycle, and replaces the old `check_all.py` entrypoint with a
validation subsystem.

## Goals

- Make harness responsibilities clear from paths and catalog metadata.
- Make validation profiles explicit and scientific instead of relying on
  `quick` / `full` lists embedded in one script.
- Move all public harness scripts out of the harness root.
- Remove old harness paths instead of keeping compatibility shims.
- Keep experiment-only workflows isolated and mechanically guarded.
- Clean committed harness cache artifacts such as `__pycache__/`.
- Update documentation, tests, and branch-policy references in the same change.

## Non-Goals

- This design does not change formal recording, alignment, camera runtime, demo
  runtime, or visualization behavior.
- This design does not add new diagnostics or benchmarks.
- This design does not preserve old harness script paths.
- This design does not make hardware validation automatic by default.

## Target Directory Layout

`scripts/harness/` will keep only package support files at the root:

```text
scripts/harness/
  __init__.py
  _catalog.py
  guards/
  validation/
    run.py
  diagnostics/
    demo/
    depth/
    visualization/
    hardware/
  benchmarks/
    ffs/
    sam/
  experiments/
    edgetam/
    ffs/
    sam/
    visualization/
  support/
```

The lifecycle meaning is:

- `guards/`: deterministic repo and architecture guards.
- `validation/`: validation matrix runner and profile logic.
- `diagnostics/`: operator-facing inspection tools and bounded diagnostics.
- `benchmarks/`: benchmarking and proof-of-life utilities.
- `experiments/`: isolated research-style workflows.
- `support/`: helper modules imported by harness entrypoints.

The old root-level public harness scripts will be removed after migration. No
wrapper or deprecation shim will be kept.

## Catalog Schema

`scripts/harness/_catalog.py` remains the source of truth, but its entries will
describe the validation matrix rather than only path, category, summary, and
help profile.

The target shape is:

```python
ValidationProfile = Literal["smoke", "deterministic", "hardware", "exhaustive"]
Lifecycle = Literal[
    "guards",
    "validation",
    "diagnostics",
    "benchmarks",
    "experiments",
    "support",
]

@dataclass(frozen=True)
class HarnessEntry:
    path: str
    lifecycle: Lifecycle
    domain: str
    summary: str
    validation_profile: ValidationProfile | None = None
    help: bool = False
    automatic: bool = True
    requires: tuple[str, ...] = ()
```

Rules:

- `path` must exist.
- `lifecycle` must match the path prefix.
- `validation_profile=None` marks support code that is cataloged but not run.
- `help=True` means the validation runner checks `--help`.
- `automatic=False` means the entry is listed but not run unless explicitly
  requested.
- `requires` records external needs such as `camera`, `gpu`, `gui`, `tensorrt`,
  or `external_repo`.

The catalog guard will reject uncataloged public harness Python files,
nonexistent paths, lifecycle/path mismatches, experiment paths outside
`experiments/`, non-experiment entries inside `experiments/`, duplicate paths,
and harness `__pycache__` directories.

## Validation Profiles

The validation subsystem replaces `scripts/harness/check_all.py` with:

```bash
python scripts/harness/validation/run.py --profile smoke
python scripts/harness/validation/run.py --profile deterministic
python scripts/harness/validation/run.py --profile hardware
python scripts/harness/validation/run.py --profile exhaustive
```

Default profile is `smoke`.

Profile semantics:

- `smoke`: cheap deterministic validation for everyday work.
- `deterministic`: `smoke` plus broader offline tests and help checks.
- `hardware`: hardware, GUI, external-service, and environment proof-of-life
  commands. These are listed by default and only executed with an explicit
  manual-run flag such as `--run-hardware`.
- `exhaustive`: `smoke` plus `deterministic` plus broader long-running offline
  tests.

The runner generates harness help checks from catalog metadata instead of
hard-coding public script lists. Unit-test batches may live in the validation
subsystem, but command generation should be centralized there rather than
spread through documentation or guard scripts.

`--full` is not preserved. Documentation and repo policy must use the new
profile names.

## Migration Map

The migration will use `git mv` for tracked files.

General mapping:

- `scripts/harness/check_*.py` -> `scripts/harness/guards/`
- `scripts/harness/check_all.py` -> `scripts/harness/validation/run.py`
- D455 probe, stream probe, WSLg, and hardware report scripts ->
  `scripts/harness/diagnostics/hardware/`
- Demo render and demo replay diagnostics ->
  `scripts/harness/diagnostics/demo/`
- Depth comparison and reprojection diagnostics ->
  `scripts/harness/diagnostics/depth/`
- Visual comparison and professor-facing render tools ->
  `scripts/harness/diagnostics/visualization/`
- FFS proof-of-life, TensorRT verification, and FFS benchmarks ->
  `scripts/harness/benchmarks/ffs/`
- SAM benchmarks and mask generation helpers are split between
  `benchmarks/sam/` and `support/` depending on whether they are public CLIs or
  imported helpers.
- Shared harness helpers such as object registries, mask helpers, and geometry
  utilities -> `scripts/harness/support/`.
- Existing experiment scripts stay under `scripts/harness/experiments/`, but are
  grouped by domain under `edgetam/`, `ffs/`, `sam/`, and `visualization/`.

Public docs, tests, `AGENTS.md`, branch-policy text, and imports must be
updated in the same change. Old script paths should not remain in source text
except in historical design documents where changing them would distort the
record.

## Boundary Rules

Formal runtime code must still not import experiment-only code. The experiment
boundary guard will treat every module under `scripts.harness.experiments.*` as
experiment-only, regardless of the new experiment subdirectory.

Public harness entrypoints may import `scripts.harness.support.*`. Core
recording and alignment code must not import harness modules.

Hardware entries must be cataloged with `automatic=False` and explicit
requirements. The validation runner must not run them during `smoke`,
`deterministic`, or `exhaustive`.

## Testing Strategy

Update or add tests for:

- New catalog schema and profile metadata.
- Catalog guard lifecycle/path validation.
- Absence of old root-level public harness scripts.
- Absence of `scripts/harness/**/__pycache__`.
- Validation runner default profile.
- Profile expansion:
  - `smoke` runs only smoke entries and smoke unit tests.
  - `deterministic` includes smoke plus deterministic.
  - `exhaustive` includes smoke, deterministic, and exhaustive.
  - `hardware` lists manual entries unless manual execution is explicitly
    enabled.
- Generated command paths exist.
- Unit-test modules in each profile exist.
- Experiment boundary guard still blocks imports from
  `scripts.harness.experiments.*`.
- Documentation references the new validation command and no longer points users
  to `scripts/harness/check_all.py`.

Implementation validation commands:

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/validation/run.py --profile smoke
```

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/validation/run.py --profile deterministic
```

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/validation/run.py --profile exhaustive
```

## Rollout Plan

1. Introduce the new catalog schema and validation runner.
2. Move harness files into the target directory layout.
3. Update catalog entries to the new paths and metadata.
4. Update imports, tests, docs, AGENTS, and guard fragments.
5. Remove committed `__pycache__` artifacts under `scripts/harness/`.
6. Run targeted catalog and validation-runner tests.
7. Run the new `smoke` validation profile.
8. Run `deterministic`.
9. Run `exhaustive` when the local environment can afford the broader run.

## Risks And Mitigations

- Old user commands break because no shims are kept. Mitigate by updating every
  repo-owned reference in the same change and documenting the new command
  clearly.
- Large path migration can break imports. Mitigate by preferring package imports
  from `scripts.harness.support` and running import-focused tests.
- Hardware checks might accidentally run in automatic profiles. Mitigate through
  catalog `automatic=False`, `requires`, and validation-runner tests.
- Catalog and directory layout might drift. Mitigate with lifecycle/path guard
  checks.

## Approved Decisions

- Use catalog-first full migration.
- Allow real file moves and path renames.
- Do not keep old-path compatibility shims.
- Keep `_catalog.py` as the source of truth.
- Use validation profiles `smoke`, `deterministic`, `hardware`, and
  `exhaustive`.
- Replace `scripts/harness/check_all.py` with
  `scripts/harness/validation/run.py`.
- Use lifecycle top-level directories and domain subdirectories for diagnostics
  and experiments.
- Clean harness `__pycache__` artifacts during migration.
