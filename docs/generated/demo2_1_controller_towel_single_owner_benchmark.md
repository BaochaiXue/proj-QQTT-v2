# Demo 2.1 Towel-Controller Single-Owner Benchmark

Date: 2026-05-06

## Contract

This benchmark used the formal Demo 2.1 controller-object live path:

```text
controller_prompt=towel
object_prompt=stuffed animal
init_mode=sam31-first-frame
depth_source=ffs
FFS=20-30-48 valid_iters=4 480x864 builderOpt5
temporal_grouping=timestamp-nearest max_skew=33.4ms
object_filter=enhanced-pt
controller_filter=pt-filter
fallback_allowed=false
```

`towel` is a temporary experiment controller prompt for the current two-cloth
scene. The default controller prompt remains `hand`.

## Sanity Result

Live SAM3.1 initialized all three cameras with nonzero masks:

```text
cam0 object_px=19074 controller_px=26153
cam1 object_px=18203 controller_px=17556
cam2 object_px=11255 controller_px=16938
```

The 60s sanity run produced complete fused/rendered groups.

## 120s Benchmark Summary

After-warmup metrics:

| Mode | Render FPS | Fusion FPS | Complete / Total | Complete Ratio | Timeouts | FFS p95 ms | GPU-owner p95 ms | EdgeTAM cycle p95 ms | Object pts min/median | Controller pts min/median | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| separate-workers gate2 | 0.51 | 0.51 | 38 / 367 | 10.4% | 170 | 561.8 | n/a | n/a | 11488 / 11590.5 | 19486 / 19531.0 | Too many partial-group timeouts |
| single-owner no-pin | 3.85 | 3.85 | 315 / 367 | 85.8% | 0 | 106.7 | 298.8 | 191.7 | 11332 / 11593.0 | 19482 / 19535.0 | Best current candidate |
| single-owner pin-ffs | 3.59 | 3.59 | 299 / 383 | 78.1% | 2 | 114.4 | 341.9 | 218.9 | 11415 / 11607.0 | 19487 / 19538.0 | Pinned FFS staging did not help |
| single-owner edge-first | 3.74 | 3.74 | 313 / 360 | 86.9% | 1 | 74.2 | 304.8 | 232.7 | 11320 / 11597.0 | 19483 / 19535.0 | Stable, but slower than ffs-then-edgetam |

## Interpretation

The single-owner pipeline is the main win. It changes scheduling, not quality:
one temporal-coherent capture group enters the GPU owner and exits with both
FFS depths and EdgeTAM masks, so fusion no longer waits on separately produced
partial groups.

Pinned FFS staging did not improve this scene. It increased FFS p95 and lowered
render/fusion FPS, so it should remain an ablation flag rather than becoming a
default.

The best current towel-controller candidate is:

```bash
python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset visual-5fps-single-owner \
  --track-mode controller-object \
  --controller-prompt "towel" \
  --object-prompt "stuffed animal" \
  --init-mode sam31-first-frame \
  --single-owner-order ffs-then-edgetam
```

## Artifacts

```text
docs/generated/demo2_1_controller_towel_visual5fps_sanity_60s.json
docs/generated/demo2_1_controller_towel_separate_workers_visual5fps_120s.json
docs/generated/demo2_1_controller_towel_single_owner_no_pin_120s.json
docs/generated/demo2_1_controller_towel_single_owner_pin_ffs_120s.json
docs/generated/demo2_1_controller_towel_single_owner_edge_first_120s.json
```

