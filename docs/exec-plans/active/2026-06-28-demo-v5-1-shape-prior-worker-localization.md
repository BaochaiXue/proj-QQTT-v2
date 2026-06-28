# Demo v5.1 Shape-Prior Worker Localization

## Goal

Move the Demo v5.1 managed shape-prior worker process entrypoint from the
generic `services/shape_prior_remote/server.py` location into `demo_v5_1/` so
the process code lives with the demo runtime it serves.

## Design

- Keep the remote protocol/client helpers under `services/shape_prior_remote/`
  for now because the camera runtime imports the client package directly.
- Move the long-lived worker CLI implementation to
  `demo_v5_1/shape_prior_worker.py`.
- Keep the worker's single-view shape alignment and sampling helpers local to
  `demo_v5_1/`, so the worker does not import `qqtt.demo` runtime code.
- Update Demo v5.1 command construction, docs, scope guard carveouts, and tests
  to use the new worker path without leaving a legacy server entrypoint.

## Checklist

- [x] Add failing tests for the new worker path and removed old server path.
- [x] Move the worker CLI file into `demo_v5_1/`.
- [x] Localize the worker-only shape-prior helpers under `demo_v5_1/`.
- [x] Update command construction, docs, and scope guard references.
- [x] Run focused tests, compile checks, and smoke validation.

## Validation

- Red check: the new worker path tests failed while Demo v5.1 still launched
  `services/shape_prior_remote/server.py`.
- Green check: the new worker path tests pass after moving the worker to
  `demo_v5_1/shape_prior_worker.py`.
- `conda run -n demo_2_max --no-capture-output python demo_v5_1/shape_prior_worker.py --help`
  passed.
- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_1_default_config tests.test_demo_v5_legacy_key_cleanup -v`
  passed.
- `conda run -n demo_2_max --no-capture-output python -m py_compile demo_v5_1/shape_prior_worker.py demo_v5_1/main.py tests/test_demo_v5_1_default_config.py tests/test_demo_v5_legacy_key_cleanup.py scripts/harness/guards/check_scope.py`
  passed.
- `conda run -n demo_2_max --no-capture-output python -m scripts.harness.guards.check_scope`
  passed.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
  passed.
- Follow-up cleanup: simplified the worker repo-root resolver to the
  `demo_v5_1` file location, removed the `QQTT_REPO_ROOT` override path, moved
  worker imports from `qqtt.demo.*` to local `demo_v5_1.*` helpers, and reran
  focused tests plus smoke validation successfully.
