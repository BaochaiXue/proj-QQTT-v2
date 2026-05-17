# Demo 2.2 PR3 overlapped-stages profile summary

## Runs

| run | profile |
| --- | --- |
| PR3 overlapped-stages no-render | `docs/generated/demo22_pr3_overlapped_stages_no_render_fair_profile.json` |
| PR3 overlapped-stages pointcloud | `docs/generated/demo22_pr3_overlapped_stages_pointcloud_profile.json` |
| PR2 batch-vision no-render baseline | `docs/generated/demo22_controller_object_exp_batchvision_warmup_no_render_profile.json` |
| PR2 batch-vision pointcloud baseline | `docs/generated/demo22_controller_object_exp_batchvision_real_pointcloud_profile.json` |

## Warmup-excluded results

| mode | raw_fusion_fps | filter_fps | render_fps | period p50 ms | gpu_owner p50 ms | ffs p50 ms | edgetam p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PR2 no-render | 6.105 | 6.106 | 0.000 | n/a | 157.995 | 67.849 | 89.384 |
| PR3 no-render | 5.254 | 5.254 | 0.000 | 181.379 | 181.286 | 79.005 | 181.286 |
| PR2 pointcloud | 5.690 | 5.689 | 5.689 | n/a | 168.104 | 72.725 | 94.806 |
| PR3 pointcloud | 5.373 | 5.373 | 5.372 | 179.788 | 179.610 | 77.245 | 179.610 |

## Decision

PR3 overlapped-stages is not a performance win in the current implementation.

The mask-gated depth dispatch avoids the earlier same-group join starvation, but it effectively serializes the group behind the EdgeTAM stage:

```text
EdgeTAM stage finishes
  -> dispatch FFS for the same group
  -> join and fuse
```

That makes the effective period track the full staged group latency instead of `max(FFS, EdgeTAM, fusion)`.

The pointcloud run is the valid render-path check for this PR3 profile:

```text
PR3 pointcloud render_fps = 5.372
PR2 pointcloud render_fps = 5.689
```

So the current PR3 should not be merged as a performance improvement. The next scheduler iteration needs a bounded lookahead policy that preserves same-group joins without waiting for the mask stage before launching every FFS job.

## Recommended next patch

Implement a reorder-buffer scheduler:

```text
capture group N:
  dispatch EdgeTAM(N) immediately
  dispatch FFS(N + lookahead) from a bounded queue
  join only exact group_id matches
  drop stale groups only after a newer complete group is ready
```

This keeps semantic correctness while allowing real cross-group overlap.
