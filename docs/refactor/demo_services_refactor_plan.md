# Demo Services Refactor Plan

The service refactor keeps existing demo semantics stable while moving shared
implementation details out of demo-specific runtimes.

## Frozen Contracts

- Demo 2.3 remains the FFS + EdgeTAM dual-GPU semantic point-cloud runtime.
- Demo 3.0 and Demo 3.1 remain online-only RealSense runtimes.
- Demo 3.1 keeps GPU0 for the main RealSense / EdgeTAM / fusion / render path
  and GPU1 for the CoTracker process.
- Demo 3.1 keeps CPU-only latest-wins IPC and does not transfer CUDA tensors
  across processes.
- Demo 3.0 and Demo 3.1 keep FuturePhysTwin dense tracking semantics:
  controller/towel mask cap before query/fusion, object/controller capped union,
  `phystwin_dense`, `auto = min(capped_union_pixels, 5000)`, and torch
  `randperm(seed + camera_idx)`.

## Phase Order

1. Extract profile keys, latest-wins queues, object volume filtering, and render
   profile facade around existing behavior.
2. Move Open3D lifecycle into the render service after the facade has tests.
3. Move semantic fusion after synthetic camera fusion tests exist.
4. Move GPU worker startup after Demo 3.1 CoTracker process status handling is
   stable through a common service.
5. Move tracking overlay lift and depth-ring alignment after CoTracker IPC is
   fully service-owned.
