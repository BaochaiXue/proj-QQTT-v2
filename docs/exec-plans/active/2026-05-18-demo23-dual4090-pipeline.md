# Demo 2.3 Dual-4090 Pipeline

## Goal

Clone the Demo 2.2 public experience into Demo 2.3 and make the primary runtime contract a dual-GPU split:

- GPU0 owns Fast-FoundationStereo TensorRT depth.
- GPU1 owns SAM3.1 initialization and EdgeTAM tracking/masks.
- The parent process owns RealSense capture, same-group join, raw fusion, async filter, and render.

The first version keeps the existing CPU/NumPy `DepthGroup` and `CameraMaskPacket` contracts. It does not introduce GPU-to-GPU tensor transport, CUDA Graphs, or renderer changes.

## Scope

- Add a Demo 2.3 entrypoint under `demo_v2_3/`.
- Add a Demo 2.3 runtime facade and dual-GPU worker contracts under `qqtt/demo/`.
- Extend the shared three-view runtime with a `demo2.3-dual4090-maxfps` preset and `dual-gpu-split` pipeline mode.
- Add smoke tests for CLI translation, dry-run contract, worker pickleability, same-group joins, and Demo 2.2 default stability.

## Defaults

- `ffs_device = cuda:0`
- `edgetam_device = cuda:1`
- `sam31_device = cuda:1`
- `dual_gpu_queue_size = 2`
- `dual_gpu_transport = pickle`
- `dual_gpu_start_method = spawn`
- `dual_gpu_processes = True`
- FFS TensorRT batch size is `3`, using the builderOptimizationLevel=5 batch-3 artifact path.
- Capture FPS is `30` for Demo 2.3, and the default capture-group target follows the resolved camera FPS (`30` by default). The fusion/report target remains `15 FPS` for the current point-cloud path.

## Validation

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m py_compile \
  demo_v2_3/realtime_three_view_dual_gpu_async_filtered_fused_pcd.py \
  qqtt/demo/demo23_runtime.py \
  qqtt/demo/demo23_dual_gpu_workers.py \
  qqtt/demo/three_view_masked_fused_pcd_runtime.py

conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo_v2_3_dual_gpu_smoke
conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py
git diff --check
```

## Real-Hardware Follow-Up

Profile these after smoke validation:

- Demo 2.2 PR2 single-owner batch-vision baseline.
- Demo 2.3 dual-GPU split with `--render-mode none`.
- Demo 2.3 dual-GPU split with point-cloud rendering.

Success means same-group mismatch stays zero, render backpressure stays zero, Demo 2.3 point-cloud FPS exceeds the PR2 point-cloud FPS, and per-GPU sampling shows FFS on GPU0 and EdgeTAM on GPU1.
