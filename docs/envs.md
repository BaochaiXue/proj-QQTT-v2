# Environments

## Shared CUDA 13 Toolkit Policy

- Current shared WSL toolkit: `/usr/local/cuda`, resolved to `/usr/local/cuda-13.2`.
- Current shared `nvcc`: CUDA 13.2.78.
- For future CUDA 13-family extension builds, use the shared toolkit:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=12.0
```

- Do not install another env-local CUDA toolkit or `cuda-nvcc` package in new conda environments unless a specific validation plan explains why the shared toolkit is unusable.
- Do not install `cuda`, `cuda-drivers`, or Linux NVIDIA driver packages inside WSL.
- Existing validated environments such as `edgetam-max` may continue using their recorded env-local CUDA path until they are deliberately rebuilt.

## `ffs-standalone`

- Purpose: validate the external Fast-FoundationStereo repo in isolation
- Current validated Python:
  - `3.10`
- Current validated torch stack:
  - `torch==2.7.0+cu128`
  - `torchvision==0.22.0+cu128`
- Current WSL-validated helper packages:
  - `timm==1.0.26`
  - `scikit-image==0.25.2`
  - `PyTurboJPEG==2.2.0`
  - `gdown==6.0.0`
- Current optional TensorRT / ONNX add-on:
  - `onnx==1.21.0`
  - `tensorrt_cu12_bindings==10.16.1.11`
  - `tensorrt_cu12_libs==10.16.1.11`
- Reason for deviation from upstream README:
  - local RTX 5090 Laptop GPU is `sm_120`
  - upstream `torch==2.6.0+cu124` failed with `no kernel image is available for execution on the device`
- Validation command:
  - `conda run -n ffs-standalone python scripts/harness/verify_ffs_demo.py --ffs_repo ../Fast-FoundationStereo --model_path ../Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth`
  - `conda run -n ffs-standalone python scripts/harness/benchmark_ffs_configs.py --help`
  - `conda run -n ffs-standalone python scripts/harness/verify_ffs_tensorrt_wsl.py`
- Expected use:
  - official FFS demo
  - checkpoint / import sanity checks
  - saved-pair checkpoint / scale / iteration tradeoff benchmarks
  - two-stage ONNX / TensorRT engine export and headless demo validation on WSL

## `ffs-official-trt`

- Purpose: run the official Fast-FoundationStereo two-stage TensorRT route with Triton GWC enabled
- Created from:
  - `conda create -y -n ffs-official-trt --clone ffs-standalone`
- Current validated Python:
  - `3.10.18`
- Current validated torch stack:
  - `torch==2.7.0+cu128`
  - `torchvision==0.22.0+cu128`
- Current validated Triton:
  - `triton==3.4.0`
- Reason for separate environment:
  - `triton==3.3.0` and `3.3.1` failed the official Fast-FoundationStereo GWC kernel with `SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats`
  - `triton==3.4.0` fixed the local official GWC probe while keeping the RTX 5090-compatible `torch==2.7.0+cu128` stack
- Validation commands:
  - `/home/zhangxinjie/miniconda3/envs/ffs-official-trt/bin/python cameras_viewer_FFS.py --help`
  - official GWC and static-sample two-stage TRT validation are recorded in `docs/generated/ffs_official_twostage_triton_env_validation.md`
- Expected use:
  - official-style two-stage TensorRT inference: `feature_runner.engine -> Triton GWC -> post_runner.engine`
  - live/viewer or static replay tests that specifically need official two-stage TRT latency behavior

## `FFS-SAM-RS`

- Purpose: default environment for new QQTT viewer, static replay / TensorRT proxy, and visualization experiments that need RealSense, Fast-FoundationStereo, TensorRT, and SAM 3.1 in the same process.
- Current validated Python:
  - `3.12.13`
- Current validated torch stack:
  - `torch==2.11.0+cu130`
  - CUDA available through `torch.version.cuda == 13.0`
- Current validated runtime packages:
  - `pyrealsense2`
  - `sam3`
  - `atomics==1.0.3`
  - `tensorrt==10.16.1.11`
  - `triton==3.6.0`
  - `open3d==0.19.0`
- Default FFS runtime policy:
  - checkpoint: `../Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth`
  - valid iterations: `4`
  - max disparity: `192`
  - two-stage ONNX/TensorRT artifact: `result/ffs_trt_static_rounds_848x480_pad864_builderopt5_rtx5090_laptop_20260428/engines/model_20-30-48_iters_4_res_480x864/`
  - builder optimization level: `5`
  - input policy: keep real `848x480` images, edge-pad to `864x480`, then unpad outputs
- Performance boundary:
  - the level-5 TensorRT artifact is the current static replay / TensorRT proxy target and has basically reached that proxy goal
  - live PyTorch 3-camera FFS remains not realtime on the RTX 5090 laptop; best recorded `scale=0.5` reached about `22.6` aggregate FFS FPS, or about `7.5` FPS per camera
  - keep static replay / TensorRT proxy claims separate from live PyTorch realtime claims
- Validation note:
  - `docs/generated/sam31_env_validation.md` records SAM 3.1 helper validation.
  - `result/ffs_trt_static_rounds_848x480_pad864_builderopt5_rtx5090_laptop_20260428/report.md` records the current level-5 TRT speed table.
- Expected use:
  - live `cameras_viewer_FFS.py` TensorRT preview
  - live FFS probes with explicit backend labeling
  - static replay / TensorRT proxy benchmarking
  - visualization harnesses that need both SAM 3.1 masks and FFS depth
  - current operator-side experiments unless a task explicitly calls for legacy `qqtt-ffs-compat`, `ffs-official-trt`, or `FFS-max`

## `demo_2_max`

- Purpose: combined local demo environment for workflows that need EdgeTAM,
  Fast-FoundationStereo, RealSense, and SAM 3.1 in one Python process.
- Created from:
  - `conda create -y -n demo_2_max --clone FFS-SAM-RS`
- Reason for clone source:
  - `FFS-SAM-RS` already exists locally and contains the validated FFS,
    RealSense, TensorRT, Open3D, and SAM 3.1 stack
  - the documented `FFS-max-sam31-rs` target is not currently present in the
    local conda environment list
- Current validated Python:
  - `3.12.13`
- Current validated torch stack:
  - `torch==2.11.0+cu130`
  - `torchvision==0.26.0+cu130`
  - CUDA available through `torch.version.cuda == 13.0`
- Current activation-hook policy:
  - `PYTHONPATH=/home/zhangxinjie/EdgeTAM`
  - `CUDA_HOME=/usr/local/cuda`
  - PyTorch `torch/lib` plus shared CUDA 13 libraries are prepended to
    `LD_LIBRARY_PATH`
  - `TORCH_CUDA_ARCH_LIST=12.0`
  - `QQTT_SAM31_CHECKPOINT=/home/zhangxinjie/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt`
- Current validated runtime packages:
  - `pyrealsense2`
  - `sam3==0.1.0`
  - `tensorrt==10.16.1.11`
  - `triton==3.6.0`
  - `open3d==0.19.0`
  - EdgeTAM `sam2._C` from `/home/zhangxinjie/EdgeTAM`
- Current caveat:
  - `python -m pip check` reports the inherited `sam3` metadata requirement
    `numpy<2,>=1.26` while this stack keeps `numpy==2.4.4`
  - runtime validation still succeeded, including SAM 3.1 video predictor
    construction from the local checkpoint
- Validation commands:
  - `conda run --no-capture-output -n demo_2_max python verify_edgetam_max.py`
    from `/home/zhangxinjie/EdgeTAM`
  - `conda run --no-capture-output -n demo_2_max python -m unittest -v tests.test_sam31_mask_helper_smoke`
  - `conda run --no-capture-output -n demo_2_max python cameras_viewer_FFS.py --help`
  - `conda run --no-capture-output -n demo_2_max python scripts/harness/verify_ffs_tensorrt_wsl.py --help`
  - `conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py`
- Validation note:
  - `docs/generated/demo_2_max_env_validation.md`
- Expected use:
  - integrated local demos that need RealSense capture, FFS depth, SAM 3.1
    masks, and EdgeTAM tracking in the same process
  - quick operator-side experiments where separate specialized env switching
    is the main friction
  - not a replacement for isolated benchmark envs when measuring clean FPS for
    a single backend

## `FFS-max-sam31-rs`

- Purpose: cloned `FFS-max` environment for FFS max torch/CUDA/TensorRT experiments that also need QQTT RealSense entrypoints and SAM 3.1 masks
- Created from:
  - `conda create -y -n FFS-max-sam31-rs --clone FFS-max`
- Current validated Python:
  - `3.12.13`
- Current preserved torch / CUDA stack:
  - `torch==2.11.0+cu130`
  - `torchvision==0.26.0`
  - `cuda-toolkit==13.0.2`
  - `nvidia-cuda-runtime==13.0.96`
  - `tensorrt-cu13==10.16.1.11`
  - `triton==3.6.0`
- Current RealSense / QQTT camera add-on:
  - `pyrealsense2==2.56.5.9235`
  - `atomics==1.0.3`
  - `pynput==1.8.1`
  - `threadpoolctl==3.6.0`
  - `numba==0.65.1`
  - `llvmlite==0.47.0`
- Current SAM 3.1 add-on:
  - `sam3==0.1.0` installed from `git+https://github.com/facebookresearch/sam3.git@main`
  - install resolved official commit `c97c893969003d3e6803fd5d679f21e515aef5ce`
  - `QQTT_SAM31_CHECKPOINT=/home/xinjie/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt`
- Current caveat:
  - `sam3` declares `numpy<2,>=1.26`, but this clone intentionally preserves `FFS-max`'s `numpy==2.4.4`
  - runtime validation succeeded despite that metadata mismatch; see `docs/generated/ffs_max_sam31_realsense_env_validation.md`
  - Numba was added with `pip install numba` after a conda dry run showed a larger solver change set; pre-install conda and pip snapshots are in `docs/generated/ffs-max-sam31-rs-pre-numba-20260429-explicit.txt` and `docs/generated/ffs-max-sam31-rs-pre-numba-20260429-pip-freeze.txt`
- Validation commands:
  - `conda run -n FFS-max-sam31-rs python -c "import torch; print(torch.__version__, torch.version.cuda)"`
  - `conda run -n FFS-max-sam31-rs python -c "import pyrealsense2 as rs; print(getattr(rs, '__version__', 'import-ok'))"`
  - `conda run -n FFS-max-sam31-rs python -c "import numba, llvmlite, numpy; print(numba.__version__, llvmlite.__version__, numpy.__version__)"`
  - `conda run -n FFS-max-sam31-rs python -c "from qqtt.env import CameraSystem; print(CameraSystem.__name__)"`
  - `conda run -n FFS-max-sam31-rs python -c "import sam3, sam3.model_builder as mb; print(getattr(sam3, '__version__', 'unknown'), hasattr(mb, 'build_sam3_video_predictor'))"`
  - `conda run -n FFS-max-sam31-rs python -m unittest -v tests.test_sam31_mask_helper_smoke`
  - `conda run -n FFS-max-sam31-rs python scripts/harness/check_all.py`
- Expected use:
  - FFS max-stack experiments that must keep `torch==2.11.0+cu130` and CUDA 13.0
  - QQTT RealSense CLI/runtime imports
  - SAM 3.1 object-mask sidecar generation from aligned cases

## `edgetam-max`

- Purpose: isolated EdgeTAM validation environment for RTX 5090 Laptop WSL2 without touching `SAM21-max`.
- Current validated Python:
  - `3.12.13`
- Current validated torch stack:
  - `torch==2.11.0+cu130`
  - `torchvision==0.26.0+cu130`
  - CUDA available through `torch.version.cuda == 13.0`
- Current validated CUDA toolkit:
  - conda-local `nvcc` CUDA 13.0.88
  - `CUDA_HOME=/home/zhangxinjie/miniconda3/envs/edgetam-max`
  - activation hooks add CUDA 13 and PyTorch `torch/lib` to `LD_LIBRARY_PATH`
- Current caveat:
  - this env was built before the shared WSL CUDA 13 toolkit policy was established
  - keep it as-is while it remains validated; future EdgeTAM rebuilds should start from `/usr/local/cuda`
- Current EdgeTAM source:
  - repo: `/home/zhangxinjie/EdgeTAM`
  - commit: `7711e012a30a2402c4eaab637bdb00a521302c91`
  - checkpoint: `/home/zhangxinjie/EdgeTAM/checkpoints/edgetam.pt`
  - config: `/home/zhangxinjie/EdgeTAM/configs/edgetam.yaml`
- Current runtime packages:
  - `timm==1.0.15`
  - `hydra-core==1.3.2`
  - `iopath==0.1.10`
  - `opencv-python-headless==4.13.0.92`
  - `eva-decord==0.6.1`
- Validation commands:
  - `conda activate edgetam-max`
  - `python /home/zhangxinjie/EdgeTAM/verify_edgetam_max.py`
  - `python -c "import sam2._C; print('sam2._C OK')"`
- Validation note:
  - `docs/generated/edgetam_max_env_validation.md`
- Expected use:
  - EdgeTAM image predictor smoke tests
  - EdgeTAM video predictor smoke tests with initial mask and box prompt
  - follow-up still-object case experiments

## `edgetam-hf-stream`

- Purpose: Hugging Face Transformers EdgeTAMVideo streaming proof environment without modifying the patched official `edgetam-max` environment.
- Created from:
  - `conda create -n edgetam-hf-stream --clone edgetam-max -y`
- Current validated Python:
  - `3.12.13`
- Current preserved torch stack:
  - `torch==2.11.0+cu130`
  - `torchvision==0.26.0+cu130`
  - CUDA available through `torch.version.cuda == 13.0`
- Current HF runtime packages:
  - `transformers==5.7.0`
  - `accelerate==1.13.0`
  - `safetensors==0.7.0`
  - `huggingface_hub==1.13.0`
- Current caveat:
  - this env deliberately inherits `edgetam-max`'s conda-local CUDA 13 stack because it is a cloned proof environment, not a fresh CUDA rebuild
  - HF streaming uses `transformers` `EdgeTamVideoModel` / `Sam2VideoProcessor` / session APIs, not the patched official `/home/zhangxinjie/EdgeTAM` predictor code path
  - the currently usable HF model repo is `yonigozlan/EdgeTAM-hf`; `yonigozlan/edgetam-video-1` returned 404 during validation
  - `Sam2VideoProcessor.init_video_session()` currently returns a `Sam2VideoInferenceSession`; direct `EdgeTamVideoInferenceSession` construction was also validated for the EdgeTAM model path
  - `run_hf_edgetam_streaming_realcase.py` defaults to `--compile-mode vision-reduce-overhead`; pass `--compile-mode none` for eager-control benchmarks
- Validation commands:
  - `conda run --no-capture-output -n edgetam-hf-stream python -m pip check`
  - `conda run --no-capture-output -n edgetam-hf-stream python scripts/harness/verify_hf_edgetam_streaming.py --json-output docs/generated/hf_edgetam_streaming_results.json --markdown-output docs/generated/hf_edgetam_streaming_validation.md`
  - `conda run --no-capture-output -n edgetam-hf-stream python scripts/harness/verify_hf_edgetam_streaming.py --frames 3 --session-init processor --json-output docs/generated/hf_edgetam_streaming_processor_session_results.json --markdown-output docs/generated/hf_edgetam_streaming_processor_session_validation.md`
  - `conda run --no-capture-output -n edgetam-hf-stream python scripts/harness/experiments/run_hf_edgetam_streaming_realcase.py --frames 30 --prompt-modes point,box,mask --output-dir result/hf_edgetam_streaming_realcase --json-output docs/generated/hf_edgetam_streaming_realcase_results.json --markdown-output docs/generated/hf_edgetam_streaming_realcase_benchmark.md --quality-output docs/generated/hf_edgetam_streaming_quality.json --overwrite`
- Expected use:
  - frame-by-frame HF EdgeTAMVideo streaming API proof-of-life
  - local-data real-case streaming FPS and quality benchmarks
  - compare live-session behavior against offline EdgeTAM/SAM-family masks
  - prototype session-style state handling before any migration back to the patched official EdgeTAM backend

## `qqtt-ffs-compat`

- Purpose: run QQTT-side proof-of-life scripts that need both RealSense access and FFS-compatible Python packages
- Current validated Python:
  - `3.10`
- Current validated torch stack:
  - `torch==2.7.0+cu128`
  - `torchvision==0.22.0+cu128`
- Current WSL-validated helper packages:
  - `timm==1.0.26`
  - `scikit-image==0.25.2`
  - `PyTurboJPEG==2.2.0`
  - `gdown==6.0.0`
  - `pyrealsense2==2.56.5.9235`
- Current optional TensorRT add-on:
  - `tensorrt_cu12_bindings==10.16.1.11`
  - `tensorrt_cu12_libs==10.16.1.11`
- Current optional SAM 3 add-on:
  - `sam3==0.1.3`
  - `psutil==7.2.2`
- Current optional visualization add-on:
  - `rerun-sdk==0.31.2`
- Current WSL note:
  - the validated `rerun-sdk==0.31.2` path currently runs with a NumPy 2.x import runtime in this env
  - the current PyPI `sam3` path needed an explicit `psutil` install and an external `bpe_simple_vocab_16e6.txt.gz` vocab file for QQTT's sidecar helper
  - the local external checkpoint `sam3.1_multiplex.pt` did not load as a direct upstream `sam3` video-predictor checkpoint; see `docs/generated/sam31_env_validation.md`
- Validation commands:
  - `conda run -n qqtt-ffs-compat python scripts/harness/probe_d455_ir_pair.py --help`
  - `conda run -n qqtt-ffs-compat python scripts/harness/run_ffs_on_saved_pair.py --help`
  - `conda run -n qqtt-ffs-compat python scripts/harness/reproject_ffs_to_color.py --help`
  - `conda run -n qqtt-ffs-compat python record_data.py --help`
  - `conda run -n qqtt-ffs-compat python data_process/record_data_align.py --help`
  - `conda run -n qqtt-ffs-compat python -c "import tensorrt as trt; print(trt.__version__)"`
  - `conda run -n qqtt-ffs-compat python scripts/harness/visual_compare_rerun.py --help`
  - `conda run -n qqtt-ffs-compat python -m unittest -v tests.test_sam31_mask_helper_smoke`
  - `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py`
  - `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py --full`
- Expected use:
  - D455 raw IR probe
  - saved-pair FFS inference
  - geometry / scale conversion
  - integrated `stereo_ir -> ffs` camera-only workflow
  - optional live TensorRT-backed FFS viewer using prebuilt `864x480` two-stage engines with `848x480 -> 864x480` symmetric pad/unpad handling
  - optional multi-frame Rerun point-cloud diagnostics
  - fast default deterministic validation with optional full validation escalation

## WSL RealSense Raw USB Access

- Purpose: keep RSUSB-backed RealSense access ready in WSL without repeating manual `chmod`
- Current validated Linux-side install command:
  - `sudo bash env_install/install_wsl_realsense_udev.sh`
- Installed rule behavior:
  - matches `SUBSYSTEM=="usb"` and `ENV{DEVTYPE}=="usb_device"`
  - matches Intel D455 `idVendor=8086`, `idProduct=0b5c`
  - sets `GROUP=plugdev` and `MODE=0660`
- Current validated WSL prerequisite:
  - the capture user is in the `plugdev` group
- Current Windows-side prerequisite:
  - devices still need to be bound and attached with `usbipd`
  - for persistent host-side reattach, prefer `usbipd attach --wsl --auto-attach --busid <BUSID>`
- Repo-owned helper assets:
  - `env_install/99-qqtt-realsense-wsl.rules`
  - `env_install/install_wsl_realsense_udev.sh`
