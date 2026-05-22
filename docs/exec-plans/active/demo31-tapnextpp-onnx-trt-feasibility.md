# Demo 3.1 TAPNext++ ONNX/TRT Feasibility

## Goal

Study whether TAPNext++ recurrent inference can be exported or compiled through
ONNX Runtime / TensorRT without changing the successful Demo 3.1 live backend.

## Non-Goals

- Do not wire ONNX/TRT into Demo 3.1 runtime.
- Do not change TAPNext++ checkpoint, image size, query count, layer count, or
  tracking semantics.
- Do not change the default `tapnextpp` adapter behavior.
- Do not commit generated ONNX engines or TensorRT caches.

## Plan

1. Inspect the local `demo_3_1_max` ONNX/TensorRT stack and report missing
   runtime-library details.
2. Probe TAPNext++ state shape and state byte scale for q1365/view and
   q4096/view.
3. Add an isolated model-only feasibility harness that can optionally attempt
   tiny fixed-shape ONNX exports.
4. Record whether the path is:
   - impossible at export time,
   - exportable but blocked by runtime libraries,
   - exportable but blocked by TensorRT importer/operators,
   - or ready for a real fixed-shape engine benchmark.
5. Keep verification deterministic and lightweight by default.

## Acceptance

- The new probe is opt-in and does not affect live Demo 3.1.
- Default probe writes JSON and Markdown without building large ONNX artifacts.
- Heavy export/session attempts require explicit flags.
- The report distinguishes q1365/view as the 4095-total target and q4096/view
  as the 12288-total stress case.
