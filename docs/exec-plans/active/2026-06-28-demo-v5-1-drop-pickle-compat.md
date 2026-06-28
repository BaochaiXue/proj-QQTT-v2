# Demo v5.1 Drop Pickle Compatibility

## Goal

Remove Demo v5.1's legacy NumPy pickle compatibility helper and write Demo v5.1
pickle artifacts with the standard library pickle writer.

## Scope

- Replace `demo_v5_1.pickle_compat.dump_pickle_legacy_numpy` call sites with
  `pickle.dump(..., protocol=pickle.HIGHEST_PROTOCOL)`.
- Delete the unused `demo_v5_1/pickle_compat.py` helper.
- Add a narrow regression test that prevents Demo v5.1 from reintroducing the
  legacy pickle compatibility module or imports.

## Validation

- Run the targeted Demo v5.1 test that covers the import/code removal.
- Run the repo smoke validation profile if the local environment is available.
