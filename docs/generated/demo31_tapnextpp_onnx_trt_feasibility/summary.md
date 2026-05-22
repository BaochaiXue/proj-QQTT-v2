# Demo 3.1 TAPNext++ ONNX/TRT Feasibility

This is an isolated model-only probe. It does not modify or enable ONNX/TRT in the live Demo 3.1 TAPNext++ backend.

- Status: `exportable_but_trt_session_not_ready`
- Live runtime changed: `False`
- Recommendation: Keep Demo 3.1 on the existing PyTorch TAPNext++ backend. The plausible next step is a fixed-shape recurrent-cell lowering that rewrites TAPNext++ MLP Einsum into TensorRT-friendly matmul/linear ops, then rebuilds a flat-state ONNX/TRT probe for B=3,q1365 before any runtime integration.

## Runtime Stack

- Torch: `2.11.0+cu130`
- ONNX: `1.21.0`
- ONNX Runtime: `1.26.0`
- TensorRT Python: `10.16.1.11`
- Torch-TensorRT: `missing`
- ORT providers listed: `TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider`

## State Size Estimates

| Case | Total Points | Hidden State | Min State I/O Per Step |
| --- | ---: | ---: | ---: |
| q1365/view target | 4095 | 629.91 MiB | 1.23 GiB |
| q4096/view stress | 12288 | 1.32 GiB | 2.64 GiB |

## Actual Small Probes

- B=1 q=8: state `90.70 MiB`, flat tensors `24`, step-invariance `True`.
  Torch export recurrent state: `ok` nodes `1952`.
  ONNX `flat-state`: `ok`, artifact `936.32 MiB`, nodes `1598`, Einsum `... h i, h i j -> ... h j, ...td,cdD->c...tD`.
  ORT `cuda` session: `ok`, actual providers `CUDAExecutionProvider, CPUExecutionProvider`, elapsed `1637.6ms`.
  ORT `trt` session: `timeout`, actual providers ``, elapsed `20002.1ms`.

## Blockers

- Exported ONNX contains uppercase Einsum equations such as ...td,cdD->c...tD; TensorRT importer rejects these.
- torch_tensorrt is not installed in demo_3_1_max.

## Interpretation

- q1365/view is the 4095-total-point target. q4096/view is a 12288-total-point stress case.
- A deployable engine must keep TAPNext++ quality by carrying the recurrent state as inputs and outputs.
- Constant-state ONNX exports are useful only as an operator translation smoke test; they are not a live tracker.
- The existing PyTorch Demo 3.1 path should stay the default until a flat-state engine is both correct and faster.
