# Demo 2.1 Controller Prompt Probe

Date: 2026-05-06

## Context

The temporary controller-object experiment used:

```text
controller_prompt=cloth
object_prompt=stuffed animal
```

Live SAM3.1 failed fast on cam0 because it did not produce a controller mask
for `cloth`. This is the expected no-fallback behavior.

## Cam0 Probe

Current cam0 RGB was captured to:

```text
docs/generated/demo2_1_prompt_probe/cam0_239222300412_rgb.png
```

SAM3.1 one-frame prompt probe results:

```text
towel:          2 masks, union 26121 px
fabric:         2 masks, union 26153 px
two towels:     2 masks, union 26151 px
cloth:          1 mask,  union 14772 px
green cloth:    1 mask,  union 14751 px
blue cloth:     1 mask,  union 11366 px
cleaning cloth: 0 masks
rag:            0 masks
```

The best practical prompt for the current two-cloth setup is:

```text
--controller-prompt "towel"
```

It returns both cloth/towel instances in cam0 and keeps the controller slot
semantics unchanged.

## Three-Camera Static Check

Using:

```text
text_prompt=stuffed animal,towel
```

SAM3.1 one-frame segmentation returned nonzero object and controller masks for
all three captured views:

| Camera | Serial | Stuffed animal px | Towel/controller px |
| --- | --- | ---: | ---: |
| cam0 | 239222300412 | 19044 | 26121 |
| cam1 | 239222300781 | 18232 | 17563 |
| cam2 | 239222303506 | 11266 | 16946 |

## Live Sanity

A 30s live run with:

```text
--controller-prompt "towel"
--object-prompt "stuffed animal"
--track-mode controller-object
```

successfully initialized cam0:

```text
cam0 object_px=19038
cam0 controller_px=26111
```

The short run ended after writing summary/profile artifacts before all camera
EdgeTAM workers finished initialization, so it should not be used as an FPS
benchmark.

## Recommendation

For the current physical setup with two cloths on the table, use:

```bash
--controller-prompt "towel"
```

Do not change the default controller prompt. The formal default remains:

```text
controller_prompt=hand
```

