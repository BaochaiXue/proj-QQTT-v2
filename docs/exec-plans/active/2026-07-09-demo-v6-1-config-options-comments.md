# Demo v6.1 Config Option Comments

## Requirement

Problem:
`demo_v6_1/config/default.yaml` has useful intent comments, but not every
parameter states its allowed values or value shape.

Required final behavior:
- Every leaf parameter in `demo_v6_1/config/default.yaml` has an adjacent
  `Options:` comment.
- Discrete runtime choices match the parser/runtime enums.
- Free-form values describe their accepted type, range, or null behavior.
- Defaults and runtime behavior are unchanged.

State changes:
- Update comments in `demo_v6_1/config/default.yaml` only.

Invalid cases:
- Unknown enum options remain invalid at existing parser/runtime validation
  boundaries.
- Numeric/path validity is documented in comments; no new validation is added.

Constraints:
- Preserve existing local YAML value changes.
- Do not touch generated `outputs_v6_1/` artifacts.

## Plan

- [x] Inspect current YAML and config consumers.
- [x] Add `Options:` comments for every YAML leaf key.
- [x] Validate the YAML still parses and the config defaults still load.

## Validation

- `python - <<'PY' ... yaml.safe_load(...) ... PY`
- `python - <<'PY' ... assert every leaf key has an Options comment ... PY`
- `conda run -n demo_2_max --no-capture-output python - <<'PY' ... load_default_config() ... PY`
- `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_demo_v6_1_downstream.py -q`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
