# FFS Live vs Proxy Boundary Cleanup

## Goal

Clarify the current FFS realtime status across harness-facing docs:

- live PyTorch 3-camera path is not realtime on the RTX 5090 laptop
- best recorded live PyTorch 3-camera result is `scale=0.5`, aggregate FFS about `22.6 FPS`, about `7.5 FPS` per camera
- static replay / TensorRT proxy results must stay separate from live realtime claims
- `20-30-48`, `valid_iters=4`, `848x480 -> 864x480` padding, and `builderOptimizationLevel=5` is the current target-reaching TensorRT proxy/default artifact

## Scope

- Update README, workflow, hardware, environment, and generated validation notes.
- Update harness README/catalog wording where benchmark intent could be confused.
- Do not change camera runtime behavior, FFS inference code, or generated data artifacts.

## Validation

- `python scripts/harness/check_harness_catalog.py`
- `python scripts/harness/check_experiment_boundaries.py`
- `python scripts/harness/check_all.py`
- `git diff --check`
