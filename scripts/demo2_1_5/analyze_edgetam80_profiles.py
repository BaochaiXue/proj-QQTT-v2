#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np


def nested_get(record: dict[str, Any], path: Sequence[str], default: Any = None) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def stats(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=np.float64)
    if arr.size == 0:
        return {"sample_count": 0, "median": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    median = float(np.median(arr))
    return {
        "sample_count": int(arr.size),
        "median": median,
        "p50": median,
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def compute_stage_wall_from_groups(payload: dict[str, Any]) -> dict[str, float]:
    values: list[float] = []
    camera_ids = [int(item) for item in nested_get(payload, ("gpu_pipeline", "camera_ids"), [0, 1, 2])]
    for record in payload.get("per_group", []) or []:
        gpu_owner_wall = nested_get(record, ("gpu_owner", "edgetam_stage_wall_ms"))
        if gpu_owner_wall is not None:
            values.append(float(gpu_owner_wall))
            continue
        cams = [nested_get(record, ("edgetam", f"cam{camera_idx}")) for camera_idx in camera_ids]
        if not all(isinstance(item, dict) for item in cams):
            continue
        starts = [float(item.get("job_start_s", item.get("publish_s", 0.0)) or 0.0) for item in cams]
        publishes = [float(item.get("publish_s", 0.0) or 0.0) for item in cams]
        if min(starts, default=0.0) <= 0.0 or max(publishes, default=0.0) <= 0.0:
            continue
        values.append(float((max(publishes) - min(starts)) * 1000.0))
    return stats(values)


def loader_profiles(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    loaders = nested_get(payload, ("init_profile", "edgetam", "loaders"), {}) or {}
    return {str(key): value for key, value in loaders.items() if isinstance(value, dict)}


def compiled_summary(payload: dict[str, Any]) -> tuple[int, dict[str, bool], list[str], dict[str, str]]:
    loaders = loader_profiles(payload)
    per_cam: dict[str, bool] = {}
    names: set[str] = set()
    types: dict[str, str] = {}
    total = 0
    for key, profile in loaders.items():
        compiled_names = [str(item) for item in profile.get("compiled_module_names", []) or []]
        count = int(profile.get("compiled_module_count", len(compiled_names)) or 0)
        if key.startswith("cam"):
            per_cam[key] = count > 0
        total += count
        names.update(compiled_names)
        for target, typ in (profile.get("compiled_module_types", {}) or {}).items():
            types[str(target)] = str(typ)
    return total, per_cam, sorted(names), types


def profile_mode(payload: dict[str, Any], path: Path) -> str:
    mode = str(payload.get("compile_mode") or "")
    if mode:
        return mode
    stem = path.stem
    return stem.replace("demo215_edgetam80_", "").replace("_towel_profile", "")


def profile_variant(payload: dict[str, Any], path: Path, metrics: dict[str, Any]) -> str:
    loaders = loader_profiles(payload)
    gpu_mode = str(nested_get(payload, ("gpu_pipeline", "mode"), "") or "")
    batch_stats = metrics.get("edgetam_batch_vision_total_ms") or {}
    batch_samples = int(batch_stats.get("sample_count", 0) or 0) if isinstance(batch_stats, dict) else 0
    if batch_samples > 0 or "batchvision" in path.stem:
        return "batch-vision-shared-model"
    if any(key.startswith("cam") for key in loaders) or gpu_mode == "staged":
        return "replicated-3-worker"
    return gpu_mode or "unknown"


def summarize_profile(path: Path, *, target_stage_wall_p50_ms: float) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    warm = payload.get("summary_after_warmup", {}) or {}
    metrics = warm.get("metrics", {}) or {}
    stage = metrics.get("edgetam_stage_wall_ms") or compute_stage_wall_from_groups(payload)
    stage_sample_count = int(stage.get("sample_count", 0) or 0)
    if stage_sample_count <= 0 and float(stage.get("median", 0.0) or 0.0) > 0.0:
        stage_sample_count = int(warm.get("complete_mask_groups", 1) or 1)
    compiled_count, per_cam_compiled, compiled_names, compiled_types = compiled_summary(payload)
    gpu_metrics = nested_get(payload, ("gpu_sampling", "summary_after_warmup", "metrics"), {}) or {}
    gpu_util = gpu_metrics.get("gpu_util_pct", {}) or {}
    result = {
        "path": str(path),
        "mode": profile_mode(payload, path),
        "variant": profile_variant(payload, path, metrics),
        "compile_scope": str(payload.get("edgetam_compile_scope", "auto")),
        "graph_output_policy": payload.get("edgetam_graph_output_policy", {}),
        "compiled_module_count": int(compiled_count),
        "compiled_module_names": compiled_names,
        "compiled_module_types": compiled_types,
        "per_camera_compiled": per_cam_compiled,
        "stage_wall_ms": stage,
        "valid_stage_samples": bool(stage_sample_count > 0),
        "pass_80ms": bool(stage_sample_count > 0 and float(stage.get("median", 0.0)) < float(target_stage_wall_p50_ms)),
        "complete_mask_group_fps": float(warm.get("complete_mask_group_fps", 0.0) or 0.0),
        "complete_mask_groups": int(warm.get("complete_mask_groups", 0) or 0),
        "gpu_util": {
            "median": float(gpu_util.get("median", 0.0) or 0.0),
            "p90": float(gpu_util.get("p90", 0.0) or 0.0),
            "max": float(gpu_util.get("max", 0.0) or 0.0),
        },
        "cam_model_ms": {
            f"cam{idx}": metrics.get(f"edgetam_cam{idx}_model_ms", {})
            for idx in range(3)
        },
        "cpu_preprocess_ms": metrics.get("edgetam_preprocess_ms", {}),
        "mask_postprocess_ms": metrics.get("edgetam_postprocess_ms", {}),
        "mask_to_cpu_ms": metrics.get("edgetam_mask_to_cpu_ms", {}),
        "parallel_efficiency": metrics.get("edgetam_parallel_efficiency", {}),
        "batch_vision_model_ms": metrics.get("edgetam_batch_vision_model_ms", {}),
        "batch_vision_total_ms": metrics.get("edgetam_batch_vision_total_ms", {}),
        "fatal_error": payload.get("fatal_error"),
    }
    return result


def expand_profiles(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(item) for item in glob.glob(pattern)]
        if matches:
            paths.extend(matches)
        else:
            paths.append(Path(pattern))
    unique = sorted({path.resolve() for path in paths})
    return [path for path in unique if path.exists()]


def markdown_report(rows: list[dict[str, Any]], *, target_stage_wall_p50_ms: float) -> str:
    replicated_pass_count = sum(
        1 for row in rows if row["pass_80ms"] and row.get("variant") == "replicated-3-worker"
    )
    batch_pass_count = sum(
        1 for row in rows if row["pass_80ms"] and row.get("variant") == "batch-vision-shared-model"
    )
    lines = [
        "# Demo 2.1.5 EdgeTAM 80ms Compile Report",
        "",
        f"Target: `edgetam_stage_wall_ms p50 < {target_stage_wall_p50_ms:.2f} ms`.",
        "",
        f"Strict replicated 3-worker pass count: `{replicated_pass_count}`.",
        f"Batch-vision shared-model pass count: `{batch_pass_count}`.",
        "",
        "| variant | mode | compiled | graph policy | stage p50 | p90 | p95 | p99 | mask group FPS | GPU med/p90/max | pass |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        stage = row["stage_wall_ms"]
        gpu = row["gpu_util"]
        policy = row.get("graph_output_policy", {})
        policy_label = policy.get("effective", policy) if isinstance(policy, dict) else policy
        if row.get("valid_stage_samples", True):
            p50 = f"{float(stage.get('median', 0.0) or 0.0):.2f}"
            p90 = f"{float(stage.get('p90', 0.0) or 0.0):.2f}"
            p95 = f"{float(stage.get('p95', 0.0) or 0.0):.2f}"
            p99 = f"{float(stage.get('p99', 0.0) or 0.0):.2f}"
        else:
            p50 = p90 = p95 = p99 = "n/a"
        lines.append(
            "| {variant} | {mode} | {compiled} | `{policy}` | {p50} | {p90} | {p95} | {p99} | {fps:.2f} | {gmed:.0f}/{gp90:.0f}/{gmax:.0f} | {passed} |".format(
                variant=row.get("variant", "unknown"),
                mode=row["mode"],
                compiled=row["compiled_module_count"],
                policy=policy_label,
                p50=p50,
                p90=p90,
                p95=p95,
                p99=p99,
                fps=float(row.get("complete_mask_group_fps", 0.0) or 0.0),
                gmed=float(gpu.get("median", 0.0) or 0.0),
                gp90=float(gpu.get("p90", 0.0) or 0.0),
                gmax=float(gpu.get("max", 0.0) or 0.0),
                passed="yes" if row["pass_80ms"] else "no",
            )
        )
    lines.extend(["", "## Blockers", ""])
    failing = [row for row in rows if not row["pass_80ms"]]
    if not failing:
        lines.append("All supplied profiles pass the p50 gate.")
    else:
        for row in failing:
            stage = row["stage_wall_ms"]
            if not row.get("valid_stage_samples", True):
                lines.append(
                    f"- `{row.get('variant', 'unknown')} / {row['mode']}`: no valid stage samples; "
                    f"complete mask group FPS `{float(row.get('complete_mask_group_fps', 0.0) or 0.0):.2f}`."
                )
            else:
                lines.append(
                    f"- `{row.get('variant', 'unknown')} / {row['mode']}`: "
                    f"p50 `{float(stage.get('median', 0.0) or 0.0):.2f} ms`, "
                    f"complete mask group FPS `{float(row.get('complete_mask_group_fps', 0.0) or 0.0):.2f}`."
                )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--target-stage-wall-p50-ms", type=float, default=80.0)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--fail-if-no-pass", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = expand_profiles(args.profiles)
    if args.baseline:
        baseline = Path(args.baseline)
        if baseline.exists() and baseline.resolve() not in {path.resolve() for path in paths}:
            paths.insert(0, baseline)
    if not paths:
        print("No profile JSON files found.", file=sys.stderr)
        return 2
    rows = [summarize_profile(path, target_stage_wall_p50_ms=args.target_stage_wall_p50_ms) for path in paths]
    payload = {
        "target_stage_wall_p50_ms": float(args.target_stage_wall_p50_ms),
        "profiles": rows,
        "pass_count": int(sum(1 for row in rows if row["pass_80ms"])),
        "replicated_3_worker_pass_count": int(
            sum(1 for row in rows if row["pass_80ms"] and row.get("variant") == "replicated-3-worker")
        ),
        "batch_vision_shared_model_pass_count": int(
            sum(1 for row in rows if row["pass_80ms"] and row.get("variant") == "batch-vision-shared-model")
        ),
    }
    md = markdown_report(rows, target_stage_wall_p50_ms=args.target_stage_wall_p50_ms)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
    else:
        print(md)
    if args.fail_if_no_pass and payload["pass_count"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
