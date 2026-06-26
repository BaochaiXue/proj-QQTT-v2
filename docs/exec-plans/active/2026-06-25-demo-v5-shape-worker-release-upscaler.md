# Demo v5 Shape Worker Upscaler Release Plan

Goal: complete Demo v5 fake-live end-to-end by preventing SAM3D worker GPU OOM after image upscaling.

Root cause:
- The managed shape-prior worker keeps the x4 upscaler resident on GPU after producing the upscaled RGB/mask.
- SAM3D inference then runs with both the upscaler and SAM3D model resident on a 24 GB GPU, and failed in `decode_slat` with CUDA OOM.

Steps:
- [x] Add a unit test proving the worker releases the upscaler before SAM3D inference.
- [x] Release the upscaler after materializing the upscaled RGB/mask, without changing the image-upscale path.
- [x] Run focused shape-prior worker tests and py_compile.
- [ ] Re-run Demo v5 full fake-live camera E2E.

Validation:
- RED: `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo32_shape_prior_warmup.ShapePriorWorkerSam3DInputTest.test_worker_releases_upscaler_before_sam3d_inference`
- PASS: `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo32_shape_prior_warmup`
- PASS: `conda run -n demo_2_max --no-capture-output python -m py_compile services/shape_prior_remote/server.py tests/test_demo32_shape_prior_warmup.py`
