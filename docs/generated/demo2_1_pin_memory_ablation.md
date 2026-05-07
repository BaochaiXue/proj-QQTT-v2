# Demo 2.1 Pinned-Memory Transfer Ablation

Status: implementation ready; live matrix still needs hardware runs.

This ablation is quality-preserving. It does not change the official FFS depth contract, EdgeTAM compile mode, live SAM3.1 first-frame initialization, temporal grouping, or semantic filters.

## Contract

```text
depth_source=ffs
ffs_checkpoint=20-30-48
ffs_valid_iters=4
ffs_input_shape=480x864
ffs_builderOptimizationLevel=5
edgetam_compile_mode=vision-reduce-overhead
init_mode=sam31-first-frame
capture_group_policy=timestamp-nearest
object_filter=enhanced-pt
controller_filter=pt-filter
object_controller_union_before_filter=false
```

## CLI

New flags:

```text
--pin-memory
--pin-memory-mode off|edge|ffs|all
--pinned-ring-size 3
--h2d-stream-mode default|dedicated
--profile-h2d
--ffs-input-staging pinned|pageable
```

Precedence:

```text
default:
  pin_memory=false
  pin_memory_mode=off
  ffs_input_staging=pinned

--pin-memory without --pin-memory-mode:
  pin_memory_mode=all

--pin-memory-mode edge|ffs|all:
  pin_memory=true

true no-pin baseline:
  must explicitly pass --ffs-input-staging pageable
```

Important finding: the FFS TensorRT runner already used pinned host input buffers by default. The new `--ffs-input-staging pageable` branch exists so the no-pin baseline is real.

## Matrix

All commands should also include the usual live object-only demo fields: `--preset visual-5fps --track-mode object-only --init-mode sam31-first-frame --object-prompt "stuffed animal" --duration-s 120 --debug --profile-pipeline --profile-filter --profile-visualization --profile-gpu-gate --profile-h2d --profile-warmup-exclude-s 40`.

| Mode | Extra flags | Purpose |
| --- | --- | --- |
| true no-pin baseline | `--ffs-input-staging pageable` | Compare against all pinned transfer paths |
| EdgeTAM pin only | `--pin-memory-mode edge --h2d-stream-mode dedicated --ffs-input-staging pageable` | Measure CPU processor + pinned H2D for EdgeTAM only |
| FFS pin only | `--pin-memory-mode ffs --h2d-stream-mode dedicated --ffs-input-staging pinned` | Preserve existing FFS pinned path while EdgeTAM stays baseline |
| all pin | `--pin-memory-mode all --h2d-stream-mode dedicated --ffs-input-staging pinned` | Measure all available pinned staging paths |

## Metrics

The profile JSON now records:

```text
h2d.camN.edge.pin_copy_ms
h2d.camN.edge.h2d_enqueue_ms
h2d.camN.edge.h2d_wait_ms
h2d.camN.ffs.stage_ms
h2d.camN.ffs.h2d_enqueue_ms
h2d.camN.ffs.h2d_wait_ms
complete_group_ratio
fusion_fps
render_fps
```

Success means lower H2D wait/jitter or better complete group ratio without changing point-cloud quality. It is not expected to solve the full 5 FPS gap alone.
