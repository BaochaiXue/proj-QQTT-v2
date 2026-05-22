# Demo 3.1 TAPNext++ Attention Kernel Profile

This is a model-only probe. It does not change the live Demo 3.1 backend.

- Case: B=3 q=1365/view total=4095
- Torch: `2.11.0+cu130`, CUDA `13.0`
- `flash_attn` package available: `False`
- PyTorch SDP flags: flash `True`, mem-efficient `True`, math `True`

## Answer

- Uses `scaled_dot_product_attention`: `True`
- Uses flash attention kernel: `True`
- Uses mem-efficient attention kernel: `False`
- Math attention fallback likely: `False`
- Attention primary bottleneck: `False`
- De-duplicated SDPA/flash kernel time: `4.869ms`

## CUDA Time By Category

These category totals are profiler aggregates and are not mutually exclusive when parent `aten::` ops contain child kernels.

| Category | Device time ms |
| --- | ---: |
| elementwise | 64.506 |
| linear | 34.197 |
| copy | 23.110 |
| mm | 20.208 |
| flash_attention | 14.608 |
| einsum | 13.536 |
| scaled_dot_product_attention | 9.739 |
| gelu | 6.972 |
| bmm | 6.175 |
| cat | 3.359 |
| contiguous_clone | 3.325 |
| matmul | 2.105 |
| softmax | 0.020 |
| layout_view | 0.000 |

## Top Device Ops

| Op | Count | Device ms | Self device ms |
| --- | ---: | ---: | ---: |
| `aten::linear` | 204 | 34.197 | 0.000 |
| `aten::einsum` | 72 | 13.536 | 0.000 |
| `aten::addmm` | 90 | 12.063 | 12.063 |
| `aten::mul` | 175 | 10.746 | 10.746 |
| `ampere_fp16_s1688gemm_fp16_256x64_ldg8_relu_f2f_tn` | 72 | 9.149 | 9.149 |
| `aten::copy_` | 454 | 8.276 | 8.276 |
| `aten::add` | 175 | 7.368 | 7.368 |
| `aten::bmm` | 36 | 6.175 | 6.175 |
| `ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_nn` | 12 | 5.800 | 5.800 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1})` | 73 | 5.545 | 5.545 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | 72 | 5.384 | 5.384 |
| `aten::to` | 437 | 4.878 | 0.000 |
| `aten::_to_copy` | 364 | 4.878 | 0.000 |
| `aten::scaled_dot_product_attention` | 12 | 4.869 | 0.000 |
| `aten::_scaled_dot_product_flash_attention` | 12 | 4.869 | 0.000 |
| `aten::_flash_attention_forward` | 12 | 4.869 | 4.869 |
| `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass::half_t, Flash_kernel_traits<64, 128, 128, 4, cutlass::half_t> >, false, false, false, false, false, true, false, false>(pytorch_flash::Flash_fwd_params)` | 12 | 4.869 | 4.869 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>)` | 355 | 4.839 | 4.839 |
| `aten::gelu` | 40 | 3.486 | 3.486 |
| `ampere_fp16_s16816gemm_fp16_128x64_ldg8_relu_f2f_tn` | 12 | 2.790 | 2.790 |

## Interpretation

- The current PyTorch path does select flash attention through SDPA.
- This is PyTorch's flash SDPA kernel, not evidence that the external FlashAttention3 package is installed or used.
- If flash attention is a small fraction of total recurrent time, the next speed work should focus on linear/einsum/state/update kernels rather than only attention selection.
