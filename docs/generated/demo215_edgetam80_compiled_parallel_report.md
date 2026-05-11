# Demo 2.1.5 EdgeTAM 80ms Compile Report

Target: `edgetam_stage_wall_ms p50 < 80.00 ms`.

Strict replicated 3-worker pass count: `0`.
Batch-vision shared-model pass count: `1`.

| variant | mode | compiled | graph policy | stage p50 | p90 | p95 | p99 | mask group FPS | GPU med/p90/max | pass |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| replicated-3-worker | none | 0 | `none` | 134.62 | 146.73 | 151.23 | 161.90 | 7.35 | 33/35/42 | no |
| batch-vision-shared-model | vision-reduce-overhead | 1 | `clone` | 77.92 | 86.24 | 89.65 | 97.55 | 12.48 | 46/49/51 | yes |
| replicated-3-worker | components-max-autotune-no-cudagraphs | 12 | `none` | n/a | n/a | n/a | n/a | 0.00 | 34/64/91 | no |
| replicated-3-worker | components-reduce-overhead | 12 | `clone` | n/a | n/a | n/a | n/a | 0.00 | 0/41/90 | no |
| replicated-3-worker | vision-max-autotune-no-cudagraphs | 3 | `none` | 125.18 | 138.62 | 143.48 | 161.54 | 7.86 | 30/58/90 | no |
| replicated-3-worker | vision-reduce-overhead | 3 | `clone` | 96.11 | 105.34 | 107.96 | 117.72 | 10.30 | 36/39/40 | no |

## Blockers

- `replicated-3-worker / none`: p50 `134.62 ms`, complete mask group FPS `7.35`.
- `replicated-3-worker / components-max-autotune-no-cudagraphs`: no valid stage samples; complete mask group FPS `0.00`.
- `replicated-3-worker / components-reduce-overhead`: no valid stage samples; complete mask group FPS `0.00`.
- `replicated-3-worker / vision-max-autotune-no-cudagraphs`: p50 `125.18 ms`, complete mask group FPS `7.86`.
- `replicated-3-worker / vision-reduce-overhead`: p50 `96.11 ms`, complete mask group FPS `10.30`.
