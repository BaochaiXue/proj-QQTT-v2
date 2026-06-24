# Shape Prior Worker Preload And Warmup

## Goal

Move SAM3D Objects and x4 upscaler cold-start work out of the first
shape-prior request by adding explicit worker startup preload/warmup flags.
The worker must not report ready until requested startup work completes, while
Demo 3.2/Demo v4 live pipelines remain fail-soft if the worker is unavailable.

## Steps

1. Add failing tests for `--preload-models`, `--warmup-models`, preload timing,
   strict warmup failure, and request-path timing fields.
2. Add worker startup helpers that preload the upscaler and SAM3D inference
   model, run a deterministic dummy upscaler + SAM3D + mesh-conversion warmup,
   and store startup timing metadata.
3. Change `server.py` startup so preload/warmup runs before ZeroMQ bind/ready
   logging, and failures exit nonzero before accepting requests.
4. Keep request-path timing clean: report request-local model load separately
   from image upscale, and include worker startup timing in every response.
5. Update Demo 3.2/Demo v4/external-dependency docs so benchmark worker
   commands use `--preload-models --warmup-models` and explain cold-start vs
   warm-request latency.
6. Run focused shape-prior tests, `py_compile`, and smoke validation.
