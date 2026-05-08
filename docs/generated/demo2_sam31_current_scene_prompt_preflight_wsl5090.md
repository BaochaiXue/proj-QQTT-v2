# Demo 2 SAM3.1 Current-Scene Prompt Preflight

Date: 2026-05-07

## Target

```text
machine: WSL Ubuntu RTX 5090 Laptop
camera: local RealSense D455
remote server: not used in this preflight
object prompt: "stuff toy"
controller prompt: "rag"
```

The current scene labels were updated from the old examples:

```text
old object prompt: "stuffed animal"
old controller prompt: "hand"
current object prompt: "stuff toy"
current controller prompt: "rag"
```

No automatic prompt fallback is allowed for formal runs.

## Object-Only Preflight

Command:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2/realtime_masked_edgetam_pcd.py \
  --serial 239222300412 \
  --profile 848x480 \
  --fps 60 \
  --depth-source none \
  --init-mode sam31-first-frame \
  --track-mode object-only \
  --object-prompt "stuff toy" \
  --pcd-mode none \
  --render-mode none \
  --compile-mode vision-reduce-overhead \
  --dtype bfloat16 \
  --duration-s 40 \
  --debug \
  --profile-cuda-events
```

Log:

```text
/tmp/demo2_sam31_init_stuff_toy_object_only_5090.log
```

Result:

```text
status = fail-fast
reason = SAM3.1 did not produce a mask for label 'stuff toy'
remote FFS requests = none
masked_uv_depth benchmark = not run
```

The earlier 8 second preflight ended during model initialization and produced a
shutdown abort, so it is not used as the prompt decision. The 40 second run
completed the SAM3.1/EdgeTAM initialization path and produced the explicit
prompt failure above.

## Controller-Object Preflight

Skipped.

Reason:

```text
object prompt "stuff toy" failed first-frame SAM3.1 initialization.
controller-object remote testing requires both object and controller masks.
```

## Decision

```text
current-scene prompt preflight: fail
blocking prompt: object="stuff toy"
controller prompt "rag": not tested
next step: user must explicitly choose the next object prompt
```

This is not a remote FFS or ZeroMQ transport failure. The 4090 server should
remain in `masked_uv_depth/lz4` mode, but WSL-5090 should not run remote masked
PCD until live SAM3.1 first-frame initialization produces a non-empty object
mask.
