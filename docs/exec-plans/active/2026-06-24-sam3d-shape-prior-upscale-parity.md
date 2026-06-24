# SAM3D Shape Prior Upscale Parity

## Goal

Make Demo 3.2 shape-prior warmup match the existing `data_process_sam3d`
ordering: object-mask crop, x4 image upscaling, then SAM3D inference.

## Steps

1. Add focused tests around the remote worker input preparation.
2. Move the crop/upscale behavior into the worker without importing heavy
   upscaler dependencies at module import time.
3. Resize the cropped object mask to the upscaled RGB size before SAM3D.
4. Record `image_upscale_ms`, mask timing, and input/crop/upscaled sizes in the
   worker metadata.
5. Run targeted shape-prior tests and smoke validation.
