# Demo 3 ONNX/TensorRT Probe

This probe checks whether an already exportable tracking model can be loaded through ONNX Runtime CUDA/TensorRT execution providers.
It does not claim TAPNext or LocoTrack are exportable until a concrete model wrapper or ONNX path is provided.

| Model | Export ONNX | ORT CUDA | ORT TensorRT | Notes |
| --- | --- | --- | --- | --- |
| locotrack | fail | unavailable | unavailable | No exportable model wrapper or ONNX path was provided. |
| tapnext | fail | unavailable | unavailable | No exportable model wrapper or ONNX path was provided. |
