# Demo 2.2 PR3b stage scheduler profiling

Date: 2026-05-17

Repo commit under test: `79be7ae` for the scheduler implementation, with profiling run from `c25bc8c`.

Experiment mode: `controller-object-exp`

Object prompt: `stuffed animal`

Controller prompt: `towel`

Visible RealSense serials:

- `239222300412`
- `239222300781`
- `239222303506`

## Question

This profiling run answers two scheduler questions:

1. Does PR3b make FFS depth ready before EdgeTAM mask for the same group?
2. If depth is ready first but FPS does not improve, is same-GPU FFS/EdgeTAM overlap likely a negative tradeoff?

## No-render scheduler matrix

| Run | Mode | filter FPS | display period p50/p90/p95 ms | GPU-owner p50/p90/p95 ms | FFS stage p50 ms | EdgeTAM stage p50 ms | depth ready before mask | depth wait after mask ms | mask wait after depth ms | stale depth drops |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | PR2 single-owner batch vision | 5.803 | 164.560 / 181.325 / 194.608 | 164.667 / 180.795 / 191.401 | n/a | n/a | 0.000 | n/a | n/a | 0 |
| B | PR3b `mask-gated` | 5.441 | 175.314 / 194.875 / 208.007 | 175.718 / 192.117 / 204.571 | 76.735 | 175.718 | 0.000 | 77.776 | 0.000 | 0 |
| C | PR3b `edge-start` | 5.611 | 167.319 / 219.667 / 243.227 | 166.592 / 221.463 / 243.071 | 82.327 | 166.563 | 1.000 | 0.000 | 87.470 | 1 |
| D | PR3b `bounded-lookahead`, lookahead 1 | 3.717 | 254.676 / 286.167 / 403.297 | 254.247 / 276.437 / 302.558 | 78.834 | 254.228 | 1.000 | 0.000 | 130.820 | 906 |
| E | PR3b `bounded-lookahead`, lookahead 2 | 3.657 | 250.657 / 297.721 / 484.611 | 250.590 / 269.744 / 279.060 | 76.793 | 250.580 | 1.000 | 0.000 | 127.170 | 873 |

## Pointcloud confirmation

The pointcloud run compares the current stable path against the best PR3b no-render mode, `edge-start`.

| Run | Mode | filter FPS | render FPS | display period p50/p90/p95 ms | GPU-owner p50/p90/p95 ms | FFS stage p50 ms | EdgeTAM stage p50 ms | render p50/p90/p95 ms | render backpressure |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- | ---: |
| PR2 | single-owner batch vision | 5.278 | 5.278 | 183.065 / 198.662 / 207.013 | 182.397 / 198.509 / 207.225 | n/a | n/a | 1.810 / 2.254 / 2.451 | 0 |
| PR3b | `edge-start` | 5.210 | 5.210 | 185.636 / 209.752 / 222.009 | 184.621 / 211.379 / 219.949 | 90.366 | 184.628 | 2.061 / 2.582 / 2.802 | 0 |

## Interpretation

`edge-start` answers the scheduler semantics question: depth was ready before mask for all joined no-render and pointcloud groups (`depth_ready_before_mask_ratio = 1.0`). The old `mask-gated` failure mode is therefore fixed by PR3b reservation dispatch.

The performance result is still negative. No-render `edge-start` is slower than PR2 (`5.611` FPS vs `5.803` FPS), and true pointcloud is also slower (`5.210` FPS vs `5.278` FPS). The EdgeTAM stage wall time expands while FFS is active, so the likely blocker is same-GPU FFS/EdgeTAM contention or hidden synchronization, not renderer throughput.

Lookahead is not viable in this implementation. Lookahead 1 and 2 both make depth ready before mask, but they reduce no-render FPS to about `3.7` FPS and produce hundreds of stale depth drops. That is wasted FFS work and extra GPU contention.

Renderer is not the bottleneck in this matrix. Pointcloud render p50 stayed near `2 ms`, `render_backpressure_count = 0`, and `render_fps` matched `filter_fps`.

## Decision

PR2 single-owner + batch vision remains the stable Demo 2.2 performance path.

PR3b should stay experimental and should not be used as the final performance path. It proved the corrected scheduler can make depth ready before mask, but same-GPU overlap does not improve throughput on this setup.

Next optimization should move to FFS static-buffer/CUDA Graph or EdgeTAM decoder-stage parallelization. Do not optimize renderer for this bottleneck.

## Profile files

- `demo22_pr3b_A_pr2_single_owner_batchvision_no_render_profile.md`
- `demo22_pr3b_B_mask_gated_no_render_profile.md`
- `demo22_pr3b_C_edge_start_no_render_profile.md`
- `demo22_pr3b_D_bounded_lookahead1_no_render_profile.md`
- `demo22_pr3b_E_bounded_lookahead2_no_render_profile.md`
- `demo22_pr3b_pr2_baseline_pointcloud_profile.md`
- `demo22_pr3b_best_pointcloud_profile.md`
