# Demo 3: RealSense CoTracker Overlay

Demo 3 is a realtime visualization demo, not a FuturePhysTwin data-processing
pipeline. It follows the Demo 2.2 asynchronous three-camera pattern, requires
exactly three RealSense cameras, uses RealSense RGB-D as the only depth source,
uses HF EdgeTAM masks through the shared batch vision encoder path for semantic
object/controller masks, and runs CoTracker3 online as a separate async
tracking overlay stage.

CoTracker output is used for visualization only. FuturePhysTwin post-processing
artifacts such as `track_process_data.pkl`, inverse physics inputs, and final
controller point selection are out of scope.

Non-dry-run execution now adapts the shared three-view runtime: it opens the
three RealSense cameras, uses HF EdgeTAM masks, fuses RealSense-depth semantic
PCD, forces the HF EdgeTAM batch vision encoder, and starts CoTracker3 as a
sidecar latest-wins overlay stage. Rendering does not wait for CoTracker; stale
or missing overlays are skipped. The shared runtime tracking backend is disabled
from Demo 3, so the sidecar is the only CoTracker owner.

Dry-run contract check:

```bash
python demo_v3/realtime_three_view_cotracker3_realsense_overlay.py \
  --dry-run \
  --camera-ids 0,1,2
```

Live contract defaults:

- exactly three RealSense cameras
- `depth_source = realsense`
- `mask_source = hf_edgetam`
- `edgetam_batch_vision_encoder = true`
- `cotracker_backend = cotracker3_online`
- `cotracker_query_count = 128`
- `overlay_max_points_per_camera = 30`
- `render_waited_for_cotracker = false`
- `uses_ffs = false`

Recommended live validation order:

1. `--preset demo3-realsense-mask-only`
2. injected/fake CoTracker overlay in tests
3. real CoTracker with `--cotracker-query-count 30`
4. real CoTracker with the default `--cotracker-query-count 128`
