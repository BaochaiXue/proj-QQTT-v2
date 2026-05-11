#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUTPUT_ROOT = ROOT / "data" / "experiments" / "demo3_onnx_trt_probe"
DEFAULT_JSON = ROOT / "docs" / "generated" / "demo3_onnx_trt_probe.json"
DEFAULT_MD = ROOT / "docs" / "generated" / "demo3_onnx_trt_probe.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe ONNX Runtime CUDA/TensorRT EP support for Demo 3 tracking models.")
    parser.add_argument("--models", default="locotrack,tapnext")
    parser.add_argument("--onnx-path", type=Path, default=None)
    parser.add_argument("--fixed-shape", action="store_true")
    parser.add_argument("--num-query-points", type=int, default=256)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--trt-fp16", action="store_true")
    parser.add_argument("--trt-engine-cache", type=Path, default=ROOT / "data" / "cache" / "demo3_tracking_trt")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    return parser.parse_args(argv)


def _parse_models(spec: str) -> list[str]:
    return [item.strip() for item in str(spec).split(",") if item.strip()]


def _write_markdown(path: Path, results: list[dict]) -> None:
    lines = ["# Demo 3 ONNX/TensorRT Probe", ""]
    for item in results:
        lines.append(f"- {item['model']}: export={item['export_onnx']} cuda={item['onnxruntime_cuda']} trt={item['onnxruntime_tensorrt']} notes={item['quality_notes']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_probe(args: argparse.Namespace) -> dict:
    from qqtt.tracking.onnx_trt.export_probe import run_export_probe

    args.output_root.mkdir(parents=True, exist_ok=True)
    results = [
        run_export_probe(
            model_name=model_name,
            onnx_path=args.onnx_path,
            engine_cache_path=args.trt_engine_cache,
            trt_fp16=bool(args.trt_fp16),
        )
        for model_name in _parse_models(args.models)
    ]
    payload = {
        "fixed_shape": bool(args.fixed_shape),
        "num_query_points": int(args.num_query_points),
        "height": int(args.height),
        "width": int(args.width),
        "trt_fp16": bool(args.trt_fp16),
        "trt_engine_cache": str(args.trt_engine_cache),
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    _write_markdown(args.output_md, results)
    (args.output_root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_probe(args)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
