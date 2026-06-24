# PhysTwin Quality-Consistent Optimization Audit

## Goal

Identify what currently blocks speed and quality optimizations while preserving
PhysTwin-like product consistency. Consistency here means matching PhysTwin
artifact semantics, schemas, filtering/sampling rules, and qualitative product
behavior, not bitwise or random-seed reproduction.

## Steps

- [x] Inspect current PhysTwin-like contract and Demo v4 product path.
- [x] Map the product pipeline from capture to chunk publication.
- [x] Identify speed optimization blockers that would not change product
  semantics.
- [x] Identify quality consistency risks where optimization could drift from
  PhysTwin/DataProcess behavior.
- [x] Write prioritized audit findings and next actions.
- [x] Run lightweight validation for touched docs/tests.

## Output

- `docs/generated/phystwin_quality_optimization_audit_20260624.md`

## Validation

- `git diff --check`
- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v4_futurephystwin_chunks`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
