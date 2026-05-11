# Demo 3 Tracking Backend Stack

- python: 3.12.13
- gpu_name: NVIDIA GeForce RTX 5090 Laptop GPU
- cuda_available: True
- torch_cuda: 13.0
- tensorrt_importable: True
- onnxruntime_importable: True
- onnxruntime_providers: TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider

## Backends

- cotracker3_online: available - torch is importable; CoTracker3 model loads lazily through torch.hub or injected model
- locotrack: unavailable - /home/zhangxinjie/external_tracking_backends/locotrack exists but runtime module/checkpoint is not configured; module locotrack import failed: No module named 'locotrack'
- nvofa: unavailable - NVIDIA Optical Flow SDK repo found but helper/binding is not built
- onnxruntime_cuda: available - CUDAExecutionProvider found
- onnxruntime_tensorrt: available - TensorrtExecutionProvider found
- tapir: unavailable - module tapnet importable; checkpoint not configured; DEMO3_TAPIR_CHECKPOINT=<unset>, TAPIR_CHECKPOINT=<unset>; PyTorch runtime wrapper is not implemented in this dependency-gated probe
- tapnext: unavailable - module tapnet importable; checkpoint not configured; DEMO3_TAPNEXT_CHECKPOINT=<unset>, TAPNEXT_CHECKPOINT=<unset>; PyTorch runtime wrapper is not implemented in this dependency-gated probe
- vpi_lk: unavailable - import vpi failed: No module named 'vpi'
