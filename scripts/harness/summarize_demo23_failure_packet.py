from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _latest_file(pattern: str) -> Path | None:
    matches = sorted(ROOT.glob(pattern), key=lambda path: path.stat().st_mtime)
    return matches[-1] if matches else None


def _latest_file_from_patterns(patterns: tuple[str, ...]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(ROOT.glob(pattern))
    existing = sorted({path for path in matches if path.is_file()}, key=lambda path: path.stat().st_mtime)
    return existing[-1] if existing else None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _get_path(data: Mapping[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _stats_median(stats: Mapping[str, Any]) -> float:
    return _number(stats.get("median"))


def _safe_contract(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    contract = profile.get("contract")
    if isinstance(contract, Mapping):
        return contract
    return profile


def _summarize_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    contract = _safe_contract(profile)
    warm = _as_mapping(profile.get("summary_after_warmup"))
    full = _as_mapping(profile.get("summary_full_run"))
    active = warm or full
    dual = _as_mapping(active.get("dual_gpu"))
    period = _as_mapping(active.get("period_ms"))
    metrics = _as_mapping(active.get("metrics"))
    ffs_contract = _as_mapping(contract.get("ffs_contract"))
    render_mode = str(contract.get("render_mode", "unknown"))
    fusion_fps = _number(active.get("fusion_fps"))
    render_fps = _number(active.get("render_fps"))
    target_fps = _number(profile.get("target_fps", contract.get("fusion_target_fps")))
    return {
        "source_present": bool(profile),
        "demo_version": profile.get("demo_version", contract.get("demo_version")),
        "pipeline": profile.get("pipeline", _get_path(contract, ("gpu_pipeline", "mode"))),
        "render_mode": render_mode,
        "target_fps": target_fps,
        "fps": {
            "capture_group": _number(active.get("capture_group_fps")),
            "raw_fusion": _number(active.get("raw_fusion_fps")),
            "filter_output": _number(active.get("filter_output_fps")),
            "fusion": fusion_fps,
            "render": render_fps,
            "complete_group_ratio": _number(active.get("complete_group_ratio")),
        },
        "period_ms": {
            "capture_group_median": _stats_median(_as_mapping(period.get("capture_group_period_ms"))),
            "stage_join_median": _stats_median(_as_mapping(period.get("stage_join_publish_period_ms"))),
            "display_packet_median": _stats_median(_as_mapping(period.get("display_packet_publish_period_ms"))),
        },
        "worker_ms": {
            "ffs_median": _stats_median(_as_mapping(dual.get("ffs_worker_period_ms"))),
            "edgetam_median": _stats_median(_as_mapping(dual.get("edgetam_worker_period_ms"))),
            "join_latency_median": _stats_median(_as_mapping(dual.get("join_latency_ms"))),
            "raw_fusion_median": _stats_median(_as_mapping(metrics.get("raw_fusion_total_ms"))),
            "filter_median": _stats_median(_as_mapping(metrics.get("filter_total_ms"))),
        },
        "dual_gpu": {
            "capture_dispatch_fps": _number(dual.get("capture_dispatch_fps")),
            "depth_publish_fps": _number(dual.get("depth_publish_fps")),
            "mask_publish_fps": _number(dual.get("mask_publish_fps")),
            "join_publish_fps": _number(dual.get("join_publish_fps")),
            "depth_ready_before_mask_ratio": _number(dual.get("depth_ready_before_mask_ratio")),
            "mean_depth_wait_after_mask_ms": _number(dual.get("mean_depth_wait_after_mask_ms")),
            "mean_mask_wait_after_depth_ms": _number(dual.get("mean_mask_wait_after_depth_ms")),
            "ffs_queue_drops": _integer(dual.get("ffs_queue_drops")),
            "edgetam_queue_drops": _integer(dual.get("edgetam_queue_drops")),
            "stale_depth_drops": _integer(dual.get("stale_depth_drops")),
            "stale_mask_drops": _integer(dual.get("stale_mask_drops")),
            "ready_join_count": _integer(dual.get("ready_join_count")),
        },
        "ffs_contract": {
            "trt_batch_size": _integer(ffs_contract.get("trt_batch_size")),
            "builderOptimizationLevel": _integer(ffs_contract.get("builderOptimizationLevel")),
            "trt_model_dir": str(ffs_contract.get("trt_model_dir", "")),
            "batch3_isolated_artifact": bool(ffs_contract.get("batch3_isolated_artifact")),
        },
        "risk_inputs": {
            "render_mode_none_deficit_misleading": render_mode == "none" and render_fps == 0.0 and fusion_fps > 0.0,
            "queue_drop_total": _integer(dual.get("ffs_queue_drops")) + _integer(dual.get("edgetam_queue_drops")),
            "stale_drop_total": _integer(dual.get("stale_depth_drops")) + _integer(dual.get("stale_mask_drops")),
        },
    }


def _summarize_runtime_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    final = _as_mapping(summary.get("final"))
    temporal = _as_mapping(summary.get("temporal_grouping"))
    dual = _as_mapping(final.get("dual_gpu"))
    return {
        "source_present": bool(summary),
        "fatal": summary.get("fatal_error", summary.get("fatal")),
        "latest_group_id": final.get("latest_group_id"),
        "object_points": final.get("object_points"),
        "controller_points": final.get("controller_points"),
        "fps": {
            "capture_group": _number(final.get("capture_group_fps")),
            "raw_fusion": _number(final.get("raw_fusion_fps")),
            "filter_output": _number(final.get("filter_output_fps")),
            "fusion": _number(final.get("fusion_fps")),
            "render": _number(final.get("render_fps")),
        },
        "dual_gpu": {
            "same_group_mismatch": _integer(summary.get("dual_gpu_same_group_mismatch")),
            "ready_join_count": _integer(dual.get("ready_join_count")),
            "depth_groups_received": _integer(dual.get("depth_groups_received")),
            "mask_groups_received": _integer(dual.get("mask_groups_received")),
            "ffs_queue_drops": _integer(dual.get("ffs_queue_drops")),
            "edgetam_queue_drops": _integer(dual.get("edgetam_queue_drops")),
        },
        "temporal": {
            "policy": temporal.get("policy"),
            "timestamp_source": temporal.get("timestamp_source"),
            "skew_ms_median": _number(temporal.get("skew_ms_median")),
            "skew_ms_p95": _number(temporal.get("skew_ms_p95")),
            "skew_ms_max": _number(temporal.get("skew_ms_max")),
            "max_capture_skew_ms": _number(temporal.get("max_capture_skew_ms")),
            "groups_dropped_skew": _integer(temporal.get("groups_dropped_skew")),
            "groups_dropped_no_candidate": _integer(temporal.get("groups_dropped_no_candidate")),
            "stale_frames_pruned": _integer(temporal.get("stale_frames_pruned")),
        },
        "debug_fusion_calibration_report": summary.get("debug_fusion_calibration_report"),
    }


def _summarize_calibration_report(report: Mapping[str, Any]) -> dict[str, Any]:
    transforms = _as_mapping(report.get("transforms"))
    return {
        "source_present": bool(report),
        "mapping_mode": report.get("mapping_mode"),
        "debug_identity_c2w": bool(report.get("debug_identity_c2w")),
        "debug_invert_c2w": bool(report.get("debug_invert_c2w")),
        "runtime_serial_numbers": list(report.get("runtime_serial_numbers", []) or []),
        "calibration_reference_serials": list(report.get("calibration_reference_serials", []) or []),
        "pairwise_center_distances_m": dict(_as_mapping(report.get("pairwise_center_distances_m"))),
        "rotation_det": {
            key: _number(_as_mapping(value).get("rotation_det"))
            for key, value in transforms.items()
            if isinstance(value, Mapping)
        },
        "orthonormal_error_fro": {
            key: _number(_as_mapping(value).get("orthonormal_error_fro"))
            for key, value in transforms.items()
            if isinstance(value, Mapping)
        },
    }


def _summarize_calibration_preflight(report: Mapping[str, Any]) -> dict[str, Any]:
    frames = [frame for frame in report.get("frames", []) if isinstance(frame, Mapping)]
    weak = []
    for frame in frames:
        corner_count = _integer(frame.get("charuco_corner_count"))
        passes_corners = bool(frame.get("passes_corner_threshold", corner_count >= _integer(report.get("min_charuco_corners"), 35)))
        passes_error = bool(frame.get("passes_error_threshold", True))
        if not passes_corners or not passes_error:
            weak.append(
                {
                    "camera_idx": frame.get("camera_idx"),
                    "serial": frame.get("serial"),
                    "charuco_corner_count": corner_count,
                    "reprojection_error": frame.get("reprojection_error"),
                    "passes_corner_threshold": passes_corners,
                    "passes_error_threshold": passes_error,
                }
            )
    return {
        "source_present": bool(report),
        "created_at_utc": report.get("created_at_utc"),
        "serial_numbers": list(report.get("serial_numbers", []) or []),
        "board": report.get("board"),
        "min_charuco_corners": _integer(report.get("min_charuco_corners"), 35),
        "weak_frames": weak,
        "frames": [
            {
                "camera_idx": frame.get("camera_idx"),
                "serial": frame.get("serial"),
                "aruco_marker_count": _integer(frame.get("aruco_marker_count")),
                "charuco_corner_count": _integer(frame.get("charuco_corner_count")),
                "reprojection_error": frame.get("reprojection_error"),
                "passes_corner_threshold": frame.get("passes_corner_threshold"),
                "passes_error_threshold": frame.get("passes_error_threshold"),
            }
            for frame in frames
        ],
    }


def _risk_flags(packet: Mapping[str, Any]) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    profile = _as_mapping(packet.get("profile"))
    runtime = _as_mapping(packet.get("runtime_summary"))
    calibration = _as_mapping(packet.get("calibration_report"))
    preflight = _as_mapping(packet.get("calibration_preflight"))
    ffs = _as_mapping(profile.get("ffs_contract"))
    temporal = _as_mapping(runtime.get("temporal"))
    profile_risks = _as_mapping(profile.get("risk_inputs"))

    if not bool(profile.get("source_present")):
        flags.append({"severity": "medium", "code": "missing_profile", "message": "No Demo 2.3 profile JSON was loaded."})
    if bool(profile.get("source_present")) and (
        _integer(ffs.get("trt_batch_size")) != 3 or _integer(ffs.get("builderOptimizationLevel")) != 5
    ):
        flags.append(
            {
                "severity": "high",
                "code": "ffs_contract_not_batch3_opt5",
                "message": "FFS contract is not batch=3 builderOptimizationLevel=5.",
            }
        )
    if bool(calibration.get("debug_identity_c2w")) or bool(calibration.get("debug_invert_c2w")):
        flags.append(
            {
                "severity": "high",
                "code": "debug_c2w_override",
                "message": "Calibration report used identity or inverted c2w debug transforms.",
            }
        )
    if _as_mapping(preflight).get("weak_frames"):
        flags.append(
            {
                "severity": "high",
                "code": "weak_calibration_preflight",
                "message": "Latest calibration preflight has cameras failing corner or reprojection thresholds.",
            }
        )
    if _number(temporal.get("skew_ms_p95")) > 33.4 or _number(temporal.get("skew_ms_max")) > _number(temporal.get("max_capture_skew_ms"), 66.7):
        flags.append(
            {
                "severity": "medium",
                "code": "temporal_skew_pressure",
                "message": "Capture temporal skew is high enough to cause moving-object ghosting.",
            }
        )
    if _integer(runtime.get("dual_gpu", {}).get("same_group_mismatch")) > 0:
        flags.append({"severity": "high", "code": "same_group_mismatch", "message": "Runtime summary reported same-group mismatch."})
    if _integer(profile_risks.get("queue_drop_total")) > 0 or _integer(profile_risks.get("stale_drop_total")) > 0:
        flags.append(
            {
                "severity": "medium",
                "code": "latest_only_drop_pressure",
                "message": "Latest-only worker or join queues dropped groups; expect jumps even without mismatched joins.",
            }
        )
    if bool(profile_risks.get("render_mode_none_deficit_misleading")):
        flags.append(
            {
                "severity": "low",
                "code": "no_render_deficit_metric",
                "message": "For render-mode none, render_fps=0 should not be interpreted as pipeline FPS failure.",
            }
        )
    return flags


def build_failure_packet(
    *,
    profile_json: Path | None,
    summary_json: Path | None,
    calibration_report: Path | None,
    calibration_preflight: Path | None,
) -> dict[str, Any]:
    packet = {
        "inputs": {
            "profile_json": None if profile_json is None else str(profile_json),
            "summary_json": None if summary_json is None else str(summary_json),
            "calibration_report": None if calibration_report is None else str(calibration_report),
            "calibration_preflight": None if calibration_preflight is None else str(calibration_preflight),
        },
        "profile": _summarize_profile(_load_json(profile_json)),
        "runtime_summary": _summarize_runtime_summary(_load_json(summary_json)),
        "calibration_report": _summarize_calibration_report(_load_json(calibration_report)),
        "calibration_preflight": _summarize_calibration_preflight(_load_json(calibration_preflight)),
    }
    packet["risk_flags"] = _risk_flags(packet)
    return packet


def render_markdown(packet: Mapping[str, Any]) -> str:
    profile = _as_mapping(packet.get("profile"))
    runtime = _as_mapping(packet.get("runtime_summary"))
    lines = [
        "# Demo 2.3 Failure Packet",
        "",
        "## Inputs",
    ]
    for key, value in _as_mapping(packet.get("inputs")).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Contract",
            f"- pipeline: `{profile.get('pipeline')}`",
            f"- render mode: `{profile.get('render_mode')}`",
            f"- target FPS: `{profile.get('target_fps')}`",
            f"- FFS batch: `{_get_path(profile, ('ffs_contract', 'trt_batch_size'))}`",
            f"- FFS builderOptimizationLevel: `{_get_path(profile, ('ffs_contract', 'builderOptimizationLevel'))}`",
            f"- FFS TRT dir: `{_get_path(profile, ('ffs_contract', 'trt_model_dir'))}`",
            "",
            "## Throughput",
        ]
    )
    for key, value in _as_mapping(profile.get("fps")).items():
        lines.append(f"- `{key}` FPS: `{value}`")
    lines.extend(["", "## Runtime Summary"])
    lines.append(f"- fatal: `{runtime.get('fatal')}`")
    lines.append(f"- latest group: `{runtime.get('latest_group_id')}`")
    lines.append(f"- object/controller points: `{runtime.get('object_points')}` / `{runtime.get('controller_points')}`")
    lines.extend(["", "## Risks"])
    risks = list(packet.get("risk_flags", []) or [])
    if not risks:
        lines.append("- No risk flags from loaded artifacts.")
    else:
        for item in risks:
            if isinstance(item, Mapping):
                lines.append(f"- `{item.get('severity')}` `{item.get('code')}`: {item.get('message')}")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an agent-legible Demo 2.3 FPS/fused-PCD failure packet.")
    parser.add_argument(
        "--profile-json",
        type=Path,
        default=None,
        help="Demo 2.3 profile JSON. Defaults to the fixed no-render profile path, then latest demo23 profile.",
    )
    parser.add_argument("--summary-json", type=Path, default=None, help="Runtime summary JSON. Defaults to latest result/demo2_1_three_view_fused_pcd summary.")
    parser.add_argument("--calibration-report", type=Path, default=None, help="Calibration debug report. Defaults to latest docs/generated/debug_fusion report.")
    parser.add_argument("--calibration-preflight", type=Path, default=None, help="Calibration detection report. Defaults to latest docs/generated/calibration_debug report.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profile_json = args.profile_json
    if profile_json is None:
        fixed_profile = ROOT / "docs/generated/demo23_dual4090_no_render_profile.json"
        profile_json = fixed_profile if fixed_profile.is_file() else _latest_file_from_patterns(
            (
                "docs/generated/*demo23*profile*.json",
                "docs/generated/**/*demo23*profile*.json",
                "result/**/*demo23*profile*.json",
            )
        )
    summary_json = args.summary_json or _latest_file("result/demo2_1_three_view_fused_pcd/session_*_summary.json")
    calibration_report = args.calibration_report or _latest_file("docs/generated/debug_fusion/*/calibration_report.json")
    calibration_preflight = args.calibration_preflight or _latest_file("docs/generated/calibration_debug/*/detection_report.json")
    packet = build_failure_packet(
        profile_json=profile_json,
        summary_json=summary_json,
        calibration_report=calibration_report,
        calibration_preflight=calibration_preflight,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(packet), encoding="utf-8")
    if args.output_json is None and args.output_md is None:
        print(json.dumps(packet, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
