#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "docs/generated/demo31_tapnextpp_model_only"

SUMMARY_COLUMNS = (
    "batch_size",
    "query_count_per_view",
    "total_query_count",
    "image_size",
    "autocast_dtype",
    "compile",
    "fast_postprocess",
    "first_update_ms",
    "first_update_model_ms",
    "first_update_wall_with_pack_ms",
    "recurrent_update_ms_p50",
    "recurrent_update_ms_p95",
    "preprocess_ms_p50",
    "preprocess_ms_p95",
    "output_extract_ms_p50",
    "output_shape_inspect_ms_p50",
    "slice_latest_on_gpu_ms_p50",
    "gpu_wait_before_cpu_copy_ms_p50",
    "tracks_to_cpu_ms_p50",
    "visibility_to_cpu_ms_p50",
    "numpy_conversion_ms_p50",
    "tracks_normalize_shape_ms_p50",
    "visibility_normalize_shape_ms_p50",
    "normalize_shape_ms_p50",
    "xy_to_yx_ms_p50",
    "scale_xy_to_original_ms_p50",
    "scale_to_original_ms_p50",
    "total_postprocess_ms_p50",
    "postprocess_ms_p50",
    "postprocess_ms_p95",
    "result_pack_ms_p50",
    "postprocess_with_pack_ms_p50",
    "cuda_event_ms_p50",
    "cuda_event_ms_p95",
    "wall_ms_p50",
    "wall_ms_p95",
    "wall_with_pack_ms_p50",
    "tracks_raw_shape",
    "visible_raw_shape",
    "tracks_latest_shape",
    "visible_latest_shape",
    "tracks_raw_dtype",
    "visible_raw_dtype",
    "tracks_raw_device",
    "visible_raw_device",
    "tracks_cpu_bytes_p50",
    "visible_cpu_bytes_p50",
    "visibility_cpu_bytes_p50",
    "measured_wall_fps",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize TAPNext++ model-only benchmark JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--profile-json", type=Path, action="append", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser


def _profile_files(args: argparse.Namespace) -> list[Path]:
    if args.profile_json:
        return [Path(path) for path in args.profile_json]
    ignored = {"manifest.json", "raw_rows.json", "summary.json"}
    return sorted(
        path
        for path in Path(args.input_dir).glob("*.json")
        if path.name not in ignored and not path.name.startswith("summary")
    )


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _load_row(path: Path) -> dict[str, Any]:
    payload = _as_mapping(json.loads(path.read_text(encoding="utf-8")))
    return dict(payload)


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return "x".join(str(item) for item in value)
    return str(value)


def render_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Demo 3.1 TAPNext++ Model-Only Benchmark Summary",
        "",
        "This excludes RealSense, masks, Open3D, IPC, lift, and render. `recurrent_update_ms_*` is the adapter-reported TAPNext++ model update time; the postprocess columns split output extraction, target-device synchronization before CPU copy, GPU-to-CPU transfer, shape normalization, scaling, and result packing.",
        "",
    ]
    if not rows:
        lines.append("No model-only JSON files found.")
        return "\n".join(lines) + "\n"
    lines.append("| " + " | ".join(SUMMARY_COLUMNS) + " |")
    lines.append("| " + " | ".join("---" for _ in SUMMARY_COLUMNS) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(column, "")) for column in SUMMARY_COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = [_load_row(path) for path in _profile_files(args)]
    rows.sort(
        key=lambda row: (
            int(row.get("batch_size", 0) or 0),
            int(row.get("query_count_per_view", 0) or 0),
            str(row.get("autocast_dtype", "")),
            bool(row.get("compile", False)),
            bool(row.get("fast_postprocess", True)),
        )
    )
    output_json = Path(args.output_json) if args.output_json else Path(args.input_dir) / "summary.json"
    output_md = Path(args.output_md) if args.output_md else Path(args.input_dir) / "summary.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(rows)
    output_md.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
