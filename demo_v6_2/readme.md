conda run -n demo_2_max --no-capture-output python demo_v6_2/main.py 

conda run -n demo_2_max --no-capture-output python demo_v6_2/main.py --input-source live 


--max-chunks 5

## Formal mask and point-cloud pipeline

The camera may capture RealSense RGB-D at 30 FPS, but the formal realtime
product samples the latest input at a fixed 5 FPS. Depth gating, dense
world-space backprojection, and the PhysTwin radius-outlier mask refinement run
only on those formal 5 FPS frames; they do not run on every 30 FPS capture.

Each formal frame has one canonical processed mask:

1. transform the full color-aligned depth image into the calibrated world frame;
2. keep pixels with `0.2 < depth < 1.5` metres;
3. filter object and combined-controller points independently with the fixed
   PhysTwin rule (`radius=0.01 m`, `nb_points=40`);
4. clear rejected 3D points from their source 2D masks;
5. send that same processed mask to tracker validity, runtime PCD construction,
   shape-prior frame 0, and prepared PhysTwin serialization.

The formal path has no point cap, raw-mask fallback, table-Z point deletion, or
2D mask erosion. Missing/invalid camera-to-world calibration and empty processed
object/controller masks fail immediately. Raw EdgeTAM masks are diagnostic only
and are not a second formal product contract.

As in `data_process_origin`, object and controller masks are filtered
independently and overlap is not subtracted from either class. Such overlap is
therefore preserved, but it makes tracker class identity ambiguous; operators
should treat visible overlap as a segmentation-quality warning.
