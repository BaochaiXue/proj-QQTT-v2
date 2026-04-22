# Environments

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
  - `conda run -n ffs-standalone python scripts/harness/verify_ffs_demo.py --ffs_repo /home/zhangxinjie/Fast-FoundationStereo --model_path /home/zhangxinjie/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth`
  - `conda run -n ffs-standalone python scripts/harness/benchmark_ffs_configs.py --help`
  - `conda run -n ffs-standalone python scripts/harness/verify_ffs_tensorrt_wsl.py`
- Expected use:
  - official FFS demo
  - checkpoint / import sanity checks
  - saved-pair checkpoint / scale / iteration tradeoff benchmarks
  - two-stage ONNX / TensorRT engine export and headless demo validation on WSL

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
