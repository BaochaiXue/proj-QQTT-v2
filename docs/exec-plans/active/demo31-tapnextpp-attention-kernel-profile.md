# Demo 3.1 TAPNext++ Attention Kernel Profile

## Goal

Confirm whether the current TAPNext++ recurrent update uses efficient attention
kernels on RTX 4090 before doing more ONNX/TensorRT work.

## Questions

- Does recurrent update call `scaled_dot_product_attention`?
- Does it select a flash attention backend?
- Does it fall back to math or memory-efficient attention?
- How much CUDA time is spent in attention compared with linear, einsum,
  matmul, copy, permute, and contiguous work?

## Scope

- Model-only probe, no RealSense, Open3D, IPC, depth lift, or runtime changes.
- Use the current PyTorch TAPNext++ checkpoint path.
- Do not alter the Demo 3.1 live backend or tracking semantics.

## Validation

- Add a harness with `--help` coverage.
- Run a real B=3 q1365/view fp16 profile on `demo_3_1_max`.
- Keep generated evidence under `docs/generated/`.
