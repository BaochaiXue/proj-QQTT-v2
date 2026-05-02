from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


SUSPICIOUS_OPS = {
    "OneHot",
    "NonZero",
    "GatherND",
    "Scatter",
    "ScatterND",
    "Where",
    "Range",
    "TopK",
    "RoiAlign",
}


def _tensor_type_name(elem_type: int) -> str:
    import onnx

    return onnx.TensorProto.DataType.Name(int(elem_type))


def _shape_dims(value_info: Any) -> list[int | str]:
    dims: list[int | str] = []
    tensor_type = value_info.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            dims.append(str(dim.dim_param))
        else:
            dims.append("?")
    return dims


def _value_info(value_info: Any) -> dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    return {
        "name": value_info.name,
        "dtype": _tensor_type_name(tensor_type.elem_type),
        "shape": _shape_dims(value_info),
    }


def inspect_model(path: Path, *, check: bool) -> dict[str, Any]:
    import onnx

    model = onnx.load(path, load_external_data=False)
    checker_status = "not_run"
    checker_error = None
    if check:
        try:
            onnx.checker.check_model(str(path))
            checker_status = "ok"
        except Exception as exc:  # pragma: no cover - depends on external ONNX files
            checker_status = "failed"
            checker_error = str(exc)

    op_hist = Counter(node.op_type for node in model.graph.node)
    suspicious = {op: op_hist[op] for op in sorted(SUSPICIOUS_OPS) if op in op_hist}
    int64_initializers = [
        init.name
        for init in model.graph.initializer
        if int(init.data_type) == int(onnx.TensorProto.INT64)
    ]
    external_initializers = [
        init.name
        for init in model.graph.initializer
        if any(entry.key == "location" for entry in init.external_data)
    ]

    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "checker_status": checker_status,
        "checker_error": checker_error,
        "ir_version": int(model.ir_version),
        "opsets": [
            {"domain": opset.domain, "version": int(opset.version)}
            for opset in model.opset_import
        ],
        "inputs": [_value_info(value) for value in model.graph.input],
        "outputs": [_value_info(value) for value in model.graph.output],
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "external_data": bool(external_initializers),
        "external_initializer_count": len(external_initializers),
        "int64_initializer_count": len(int64_initializers),
        "op_histogram": dict(sorted(op_hist.items())),
        "suspicious_ops": suspicious,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect EdgeTAM ONNX component graphs.")
    parser.add_argument("--onnx", type=Path, action="append", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--skip-checker", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = [inspect_model(path.resolve(), check=not args.skip_checker) for path in args.onnx]
    payload = {"models": records}
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for record in records:
        print(f"ONNX: {record['path']}")
        print(f"  checker: {record['checker_status']}")
        print(f"  inputs: {[item['name'] for item in record['inputs']]}")
        print(f"  outputs: {[item['name'] for item in record['outputs']]}")
        print(f"  suspicious_ops: {record['suspicious_ops']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
