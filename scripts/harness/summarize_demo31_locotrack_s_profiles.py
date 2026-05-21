#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "docs/generated/demo31_locotrack_s_rendered_profile"

SUMMARY_COLUMNS = (
    "execution_mode",
    "query_count_per_camera",
    "window_frames",
    "rendered_fps",
    "tracker_publish_fps",
    "tracker_model_ms_p50",
    "tracker_model_ms_p95",
    "tracker_e2e_ms_p50",
    "tracker_e2e_ms_p95",
    "input_drop_count",
    "result_drop_count",
    "stale_overlay_count",
    "lift_cache_miss_count",
    "gpu0_mem_used_gb",
    "gpu1_mem_used_gb",
    "locotrack_batch_size",
    "locotrack_query_chunk_size",
    "invalid_lift_rate",
    "surface_snap_accept",
    "surface_snap_reject",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize rendered Demo 3.1 LocoTrack-S profile JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--profile-json", type=Path, action="append", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _get_number(*values: Any, default: float = 0.0) -> float:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)


def _get_int(*values: Any, default: int = 0) -> int:
    return int(round(_get_number(*values, default=float(default))))


def _sum_mapping_values(value: Any) -> int:
    if not isinstance(value, Mapping):
        return 0
    total = 0
    for item in value.values():
        try:
            total += int(item)
        except (TypeError, ValueError):
            continue
    return total


def _profile_files(args: argparse.Namespace) -> list[Path]:
    if args.profile_json:
        return [Path(path) for path in args.profile_json]
    input_dir = Path(args.input_dir)
    return sorted(path for path in input_dir.glob("*.json") if path.name not in {"manifest.json", "failures.json", "summary.json"})


def summarize_profile(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = _as_mapping(payload.get("contract"))
    summary = _as_mapping(payload.get("summary"))
    snapshot = _as_mapping(payload.get("cotracker_process_snapshot"))
    output_endpoint = _as_mapping(snapshot.get("output_endpoint"))
    tracker_stats = _as_mapping(snapshot.get("worker"))
    execution_mode = str(contract.get("tracking_backend_execution_mode", summary.get("tracking_backend_execution_mode", "")))
    query_count = contract.get("tracking_query_count_requested", summary.get("tracking_query_count_requested", "0"))
    try:
        query_count_int = int(query_count)
    except (TypeError, ValueError):
        query_count_int = 0
    accepted = _sum_mapping_values(summary.get("tracker_marker_accepted_by_camera"))
    rejected = _sum_mapping_values(summary.get("tracker_marker_rejected_by_camera"))
    result_drop_count = _get_int(
        summary.get("tracker_result_drop_count"),
        summary.get("cotracker_result_drop_count"),
        output_endpoint.get("drop_count"),
        output_endpoint.get("replace_count"),
    )
    return {
        "profile_json": str(path),
        "execution_mode": execution_mode,
        "query_count_per_camera": int(query_count_int),
        "window_frames": _get_int(contract.get("locotrack_window_frames"), summary.get("locotrack_window_frames")),
        "rendered_fps": _get_number(summary.get("rendered_fps"), summary.get("render_fps")),
        "tracker_publish_fps": _get_number(summary.get("tracker_publish_fps"), summary.get("cotracker_publish_fps")),
        "tracker_model_ms_p50": _get_number(summary.get("tracker_model_ms_median"), summary.get("cotracker_model_ms_median")),
        "tracker_model_ms_p95": _get_number(summary.get("tracker_model_ms_p95"), summary.get("cotracker_model_ms_p95")),
        "tracker_e2e_ms_p50": _get_number(summary.get("tracker_e2e_ms_median"), summary.get("cotracker_e2e_ms_median")),
        "tracker_e2e_ms_p95": _get_number(summary.get("tracker_e2e_ms_p95"), summary.get("cotracker_e2e_ms_p95")),
        "input_drop_count": _get_int(summary.get("tracker_input_drop_count"), summary.get("cotracker_input_drop_count")),
        "result_drop_count": int(result_drop_count),
        "stale_overlay_count": _get_int(summary.get("stale_overlay_count"), tracker_stats.get("stale_overlay_count")),
        "lift_cache_miss_count": _get_int(summary.get("tracking_result_without_lift_input_count")),
        "gpu0_mem_used_gb": _get_number(summary.get("gpu0_mem_used_gb")),
        "gpu1_mem_used_gb": _get_number(summary.get("gpu1_mem_used_gb")),
        "locotrack_batch_size": _get_int(
            summary.get("locotrack_batch_size"),
            summary.get("tracking_backend_batch_size"),
            contract.get("tracking_backend_batch_size"),
        ),
        "locotrack_query_chunk_size": _get_int(
            summary.get("locotrack_query_chunk_size"),
            contract.get("locotrack_query_chunk_size"),
        ),
        "invalid_lift_rate": _get_number(summary.get("invalid_lift_rate")),
        "surface_snap_accept": int(accepted),
        "surface_snap_reject": int(rejected),
    }


def render_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# Demo 3.1 LocoTrack-S Rendered Profile Summary", ""]
    if not rows:
        lines.append("No profile JSON files found.")
        return "\n".join(lines) + "\n"
    lines.append("| " + " | ".join(SUMMARY_COLUMNS) + " |")
    lines.append("| " + " | ".join("---" for _ in SUMMARY_COLUMNS) + " |")
    for row in rows:
        rendered: list[str] = []
        for column in SUMMARY_COLUMNS:
            value = row.get(column, "")
            if isinstance(value, float):
                rendered.append(f"{value:.3f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = [summarize_profile(path) for path in _profile_files(args)]
    rows.sort(key=lambda row: (str(row["execution_mode"]), int(row["query_count_per_camera"]), int(row["window_frames"])))
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
