#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "docs/generated/demo31_tapnextpp_rendered_profile"

SUMMARY_COLUMNS = (
    "execution_mode",
    "query_count_per_camera",
    "total_query_count_across_views",
    "target_class",
    "rendered_fps",
    "rendered_groups_after_warmup",
    "valid_rendered_profile",
    "tracker_publish_fps",
    "tracker_group_wall_ms_p50",
    "tracker_group_wall_ms_p95",
    "tracker_model_ms_sum_per_group_p50",
    "tracker_model_ms_sum_per_group_p95",
    "tracker_model_ms_max_per_group_p50",
    "tracker_model_ms_max_per_group_p95",
    "per_camera_model_ms_p50_by_camera",
    "model_calls_per_group",
    "model_instances_expected",
    "model_instances_actual",
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
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize rendered Demo 3.1 TAPNext++ profile JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
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


def _nested_get(payload: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cursor: Any = payload
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def _gpu_mem_gb(shared_payload: Mapping[str, Any], device_index: int) -> float:
    by_device = _as_mapping(_as_mapping(shared_payload.get("gpu_sampling")).get("summary_by_device_after_warmup"))
    device_summary = _as_mapping(by_device.get(str(int(device_index))))
    memory_mb = _get_number(_nested_get(device_summary, ("metrics", "memory_used_mb", "median"), 0.0))
    return float(memory_mb / 1024.0)


def _profile_files(args: argparse.Namespace) -> list[Path]:
    if args.profile_json:
        return [Path(path) for path in args.profile_json]
    input_dir = Path(args.input_dir)
    ignored = {
        "manifest.json",
        "manifest_q1365.json",
        "failures.json",
        "failures_q1365.json",
        "summary.json",
        "summary.md",
    }
    return sorted(
        path
        for path in input_dir.glob("*.json")
        if path.name not in ignored
        and not path.name.startswith("summary")
        and not path.name.endswith("_shared_runtime.json")
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_shared_runtime_payload(path: Path, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    embedded = _as_mapping(payload.get("shared_runtime_profile_payload"))
    if embedded:
        return embedded
    shared_path_value = payload.get("shared_runtime_profile")
    candidates: list[Path] = []
    if shared_path_value:
        candidates.append(Path(str(shared_path_value)))
    candidates.append(path.with_name(f"{path.stem}_shared_runtime.json"))
    for candidate in candidates:
        candidate = candidate if candidate.is_absolute() else (ROOT / candidate)
        if candidate.is_file():
            return _as_mapping(_load_json(candidate))
    return {}


def _query_count_from_path(path: Path) -> int:
    match = re.search(r"_q(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def _execution_mode_from_path(path: Path) -> str:
    if path.name.startswith("batch_views"):
        return "batch-views"
    if path.name.startswith("serial"):
        return "serial"
    return ""


def _target_class(query_count_per_camera: int, total_query_count: int) -> str:
    if query_count_per_camera == 1365 or 3800 <= total_query_count <= 4300:
        return "~4000_total_target"
    if query_count_per_camera == 4096 or total_query_count == 12288:
        return "stress_12288_total"
    return "sweep"


def summarize_profile(path: Path) -> dict[str, Any]:
    payload = _as_mapping(_load_json(path))
    contract = _as_mapping(payload.get("contract"))
    summary = _as_mapping(payload.get("summary"))
    snapshot = _as_mapping(payload.get("cotracker_process_snapshot"))
    output_endpoint = _as_mapping(snapshot.get("output_endpoint"))
    worker = _as_mapping(snapshot.get("worker"))
    shared_payload = _load_shared_runtime_payload(path, payload)
    warm = _as_mapping(shared_payload.get("summary_after_warmup"))

    execution_mode = str(
        contract.get(
            "tracking_backend_execution_mode",
            summary.get("tracking_backend_execution_mode", _execution_mode_from_path(path)),
        )
    )
    query_count_raw = contract.get(
        "tracking_query_count_requested",
        summary.get("query_count_per_camera", summary.get("tracking_query_count_requested", _query_count_from_path(path))),
    )
    try:
        query_count_per_camera = int(query_count_raw)
    except (TypeError, ValueError):
        query_count_per_camera = _query_count_from_path(path)
    total_query_count = _get_int(
        summary.get("total_query_count_across_views"),
        snapshot.get("total_query_count_across_views"),
        worker.get("total_query_count_across_views"),
        default=query_count_per_camera * 3,
    )
    rendered_groups = _get_int(
        summary.get("rendered_groups_after_warmup"),
        warm.get("rendered_groups"),
        warm.get("group_count"),
    )
    result_drop_count = _get_int(
        summary.get("tracker_result_drop_count"),
        summary.get("cotracker_result_drop_count"),
        output_endpoint.get("drop_count"),
        output_endpoint.get("replace_count"),
    )
    gpu0_mem_gb = _gpu_mem_gb(shared_payload, 0)
    gpu1_mem_gb = _gpu_mem_gb(shared_payload, 1)
    accepted = _sum_mapping_values(summary.get("tracker_marker_accepted_by_camera"))
    rejected = _sum_mapping_values(summary.get("tracker_marker_rejected_by_camera"))
    row = {
        "profile_json": str(path),
        "execution_mode": execution_mode,
        "tracking_backend_execution_mode": execution_mode,
        "query_count_per_camera": int(query_count_per_camera),
        "total_query_count_across_views": int(total_query_count),
        "target_class": _target_class(int(query_count_per_camera), int(total_query_count)),
        "rendered_fps": _get_number(warm.get("render_fps"), summary.get("rendered_fps"), summary.get("render_fps")),
        "rendered_groups_after_warmup": int(rendered_groups),
        "valid_rendered_profile": bool(rendered_groups > 0),
        "tracker_publish_fps": _get_number(summary.get("tracker_publish_fps"), summary.get("cotracker_publish_fps")),
        "tracker_group_wall_ms_p50": _get_number(
            summary.get("tracker_group_wall_ms_p50"),
            snapshot.get("tracker_group_wall_ms_p50"),
            worker.get("tracker_group_wall_ms_p50"),
        ),
        "tracker_group_wall_ms_p95": _get_number(
            summary.get("tracker_group_wall_ms_p95"),
            snapshot.get("tracker_group_wall_ms_p95"),
            worker.get("tracker_group_wall_ms_p95"),
        ),
        "tracker_model_ms_sum_per_group_p50": _get_number(
            summary.get("tracker_model_ms_sum_per_group_p50"),
            snapshot.get("tracker_model_ms_sum_per_group_p50"),
            worker.get("tracker_model_ms_sum_per_group_p50"),
        ),
        "tracker_model_ms_sum_per_group_p95": _get_number(
            summary.get("tracker_model_ms_sum_per_group_p95"),
            snapshot.get("tracker_model_ms_sum_per_group_p95"),
            worker.get("tracker_model_ms_sum_per_group_p95"),
        ),
        "tracker_model_ms_max_per_group_p50": _get_number(
            summary.get("tracker_model_ms_max_per_group_p50"),
            snapshot.get("tracker_model_ms_max_per_group_p50"),
            worker.get("tracker_model_ms_max_per_group_p50"),
        ),
        "tracker_model_ms_max_per_group_p95": _get_number(
            summary.get("tracker_model_ms_max_per_group_p95"),
            snapshot.get("tracker_model_ms_max_per_group_p95"),
            worker.get("tracker_model_ms_max_per_group_p95"),
        ),
        "per_camera_model_ms_p50_by_camera": summary.get(
            "per_camera_model_ms_p50_by_camera",
            snapshot.get("per_camera_model_ms_p50_by_camera", worker.get("per_camera_model_ms_p50_by_camera", {})),
        ),
        "per_camera_model_ms_p95_by_camera": summary.get(
            "per_camera_model_ms_p95_by_camera",
            snapshot.get("per_camera_model_ms_p95_by_camera", worker.get("per_camera_model_ms_p95_by_camera", {})),
        ),
        "model_calls_per_group": _get_int(
            summary.get("model_calls_per_group"),
            snapshot.get("model_calls_per_group"),
            worker.get("model_calls_per_group"),
        ),
        "model_instances_expected": _get_int(
            summary.get("model_instances_expected"),
            snapshot.get("model_instances_expected"),
            worker.get("model_instances_expected"),
        ),
        "model_instances_actual": _get_int(
            summary.get("model_instances_actual"),
            snapshot.get("model_instances_actual"),
            worker.get("model_instances_actual"),
        ),
        "tracker_batch_size": _get_int(
            summary.get("tracking_backend_batch_size"),
            summary.get("cotracker_batch_size"),
            contract.get("tracking_backend_batch_size"),
        ),
        "tracker_model_ms_p50": _get_number(summary.get("tracker_model_ms_median"), summary.get("cotracker_model_ms_median")),
        "tracker_model_ms_p95": _get_number(summary.get("tracker_model_ms_p95"), summary.get("cotracker_model_ms_p95")),
        "tracker_e2e_ms_p50": _get_number(summary.get("tracker_e2e_ms_median"), summary.get("cotracker_e2e_ms_median")),
        "tracker_e2e_ms_p95": _get_number(summary.get("tracker_e2e_ms_p95"), summary.get("cotracker_e2e_ms_p95")),
        "input_drop_count": _get_int(summary.get("tracker_input_drop_count"), summary.get("cotracker_input_drop_count")),
        "result_drop_count": int(result_drop_count),
        "stale_overlay_count": _get_int(summary.get("stale_overlay_count"), worker.get("stale_overlay_count")),
        "lift_cache_miss_count": _get_int(summary.get("tracking_result_without_lift_input_count")),
        "gpu0_mem_used_gb": gpu0_mem_gb if gpu0_mem_gb > 0.0 else _get_number(summary.get("gpu0_mem_used_gb")),
        "gpu1_mem_used_gb": gpu1_mem_gb if gpu1_mem_gb > 0.0 else _get_number(summary.get("gpu1_mem_used_gb")),
        "surface_snap_accept": int(accepted),
        "surface_snap_reject": int(rejected),
    }
    return row


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, Mapping):
        return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    return str(value)


def render_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Demo 3.1 TAPNext++ Rendered Profile Summary",
        "",
        "- `q1365/view` is the real ~4000-total target: `1365 * 3 = 4095` points.",
        "- `q4096/view` is a stress test: `4096 * 3 = 12288` points.",
        "- 45 FPS requires recurrent tracker latency at or below `22.2 ms`.",
        "- Treat a rendered profile as valid only when `rendered_groups_after_warmup > 0`.",
        "",
    ]
    if not rows:
        lines.append("No profile JSON files found.")
        return "\n".join(lines) + "\n"
    lines.append("| " + " | ".join(SUMMARY_COLUMNS) + " |")
    lines.append("| " + " | ".join("---" for _ in SUMMARY_COLUMNS) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(column, "")) for column in SUMMARY_COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = [summarize_profile(path) for path in _profile_files(args)]
    rows.sort(key=lambda row: (str(row["execution_mode"]), int(row["query_count_per_camera"])))
    output_json = Path(args.output_json) if args.output_json else Path(args.input_dir) / "summary_tapnextpp_live.json"
    output_md = Path(args.output_md) if args.output_md else Path(args.input_dir) / "summary_tapnextpp_live.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(rows)
    output_md.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
