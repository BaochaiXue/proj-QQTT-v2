#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qqtt.demo import demo22_runtime as runtime  # noqa: E402


DEFAULT_JSON = ROOT / "docs/generated/demo215_hf_edgetam_gpu_underutilization_mitigation.json"
DEFAULT_MD = ROOT / "docs/generated/demo215_hf_edgetam_gpu_underutilization_mitigation.md"


def _nested_get(record: dict[str, Any], path: Sequence[str]) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _profile_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    warm = payload.get("summary_after_warmup", {}) or {}
    metrics = warm.get("metrics", {}) or {}
    gpu = ((payload.get("gpu_sampling", {}) or {}).get("summary_after_warmup", {}) or {}).get("metrics", {}) or {}
    contract = payload.get("contract", {}) or {}
    edgetam_model = metrics.get("edgetam_model_ms") or {}
    if not edgetam_model:
        cam_medians = [
            float(_nested_get(metrics, (f"edgetam_cam{idx}_model_ms", "median")) or 0.0)
            for idx in range(3)
        ]
        cam_p90 = [
            float(_nested_get(metrics, (f"edgetam_cam{idx}_model_ms", "p90")) or 0.0)
            for idx in range(3)
        ]
        nonzero_medians = [value for value in cam_medians if value > 0.0]
        nonzero_p90 = [value for value in cam_p90 if value > 0.0]
        edgetam_model = {
            "median": sum(nonzero_medians) / len(nonzero_medians) if nonzero_medians else 0.0,
            "p90": sum(nonzero_p90) / len(nonzero_p90) if nonzero_p90 else 0.0,
        }
    return {
        "path": str(path),
        "preset": payload.get("preset"),
        "depth_source": payload.get("depth_source"),
        "compile_mode": payload.get("compile_mode") or contract.get("compile_mode"),
        "dtype": payload.get("dtype") or contract.get("dtype"),
        "mask_postprocess": payload.get("mask_postprocess") or contract.get("mask_postprocess"),
        "render_fps": float(warm.get("render_fps", 0.0) or 0.0),
        "fusion_fps": float(warm.get("fusion_fps", 0.0) or 0.0),
        "complete_group_ratio": float(warm.get("complete_group_ratio", 0.0) or 0.0),
        "edgetam_model_ms_p50": float(edgetam_model.get("median", 0.0) or 0.0),
        "edgetam_model_ms_p90": float(edgetam_model.get("p90", 0.0) or 0.0),
        "edgetam_total_ms_p90": float(_nested_get(metrics, ("edgetam_total_ms", "p90")) or 0.0),
        "edgetam_mask_to_cpu_ms_p90": float(_nested_get(metrics, ("edgetam_mask_to_cpu_ms", "p90")) or 0.0),
        "gpu_util_pct_p50": float(_nested_get(gpu, ("gpu_util_pct", "median")) or 0.0),
        "gpu_util_pct_p95": float(_nested_get(gpu, ("gpu_util_pct", "p95")) or 0.0),
    }


def _command_matrix() -> list[dict[str, Any]]:
    base = [
        "conda", "run", "--no-capture-output", "-n", "demo_2_max", "python",
        "demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py",
    ]
    common_profile = [
        "--duration-s", "90",
        "--warmup-s", "45",
        "--render-mode", "none",
        "--gpu-sampling",
        "--gpu-sampling-interval-s", "0.2",
        "--profile-cuda-events",
        "--profile-edgetam-stages",
    ]
    rows: list[dict[str, Any]] = []
    for compile_mode in (
        runtime.COMPILE_MODE_NONE,
        runtime.COMPILE_MODE_VISION_REDUCE_OVERHEAD,
        runtime.COMPILE_MODE_COMPONENTS_REDUCE_OVERHEAD,
    ):
        for dtype in ("bfloat16", "float16", "float32"):
            rows.append(
                {
                    "name": f"mask_only_{compile_mode}_{dtype}",
                    "backend": "hf",
                    "depth_source": "none",
                    "compile_mode": compile_mode,
                    "dtype": dtype,
                    "command": [
                        *base,
                        "--mask-only-debug",
                        "--compile-mode", compile_mode,
                        "--dtype", dtype,
                        "--mask-postprocess", runtime.MASK_POSTPROCESS_CUDA_INLINE,
                        *common_profile,
                    ],
                }
            )
    for mode_name, preset_flag in (
        ("live_fast_native", "--live-fast-native"),
        ("live_quality_ffs", "--live-quality-ffs"),
    ):
        rows.append(
            {
                "name": mode_name,
                "backend": "hf",
                "depth_source": "realsense" if mode_name == "live_fast_native" else "ffs",
                "compile_mode": runtime.COMPILE_MODE_VISION_REDUCE_OVERHEAD,
                "dtype": "bfloat16",
                "command": [
                    *base,
                    preset_flag,
                    "--compile-mode", runtime.COMPILE_MODE_VISION_REDUCE_OVERHEAD,
                    "--dtype", "bfloat16",
                    "--mask-postprocess", runtime.MASK_POSTPROCESS_CUDA_INLINE,
                    *common_profile,
                ],
            }
        )
    return rows


def build_report(profile_paths: Sequence[Path]) -> dict[str, Any]:
    summaries = [_profile_summary(path) for path in profile_paths if path.is_file()]
    return {
        "title": "Demo 2.1.5 HF EdgeTAM GPU underutilization mitigation",
        "generated_by": "scripts/harness/experiments/run_demo215_hf_edgetam_mitigation_matrix.py",
        "target_machine": "WSL Ubuntu RTX 5090 Laptop",
        "principle": "Optimize p50/p90 latency and end-to-end p90; GPU utilization is diagnostic, not the primary KPI.",
        "defaults_changed": False,
        "implemented_flags": {
            "profile_edgetam_stages": True,
            "profile_nsys_markers": True,
            "profile_sync": True,
            "mask_postprocess_cuda_inline": True,
            "dtype_float32_ablation": True,
            "compile_submodule_modes": True,
            "live_fast_native_preset": True,
            "live_quality_ffs_preset": True,
            "mask_only_debug_preset": True,
        },
        "command_matrix": _command_matrix(),
        "profile_summaries": summaries,
        "decision_rules": {
            "hf_keep_default": "Keep HF default unless another backend improves model_ms p90 and e2e p90 with mask parity.",
            "cuda_inline": "Promote only if mask parity passes and p90 improves.",
            "component_compile": "Treat as experimental until warmup and graph-break behavior are stable.",
            "ffs_quality": "Keep FFS quality mode separate from native fast mode.",
        },
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Demo 2.1.5 HF EdgeTAM GPU Underutilization Mitigation",
        "",
        f"- target machine: `{payload['target_machine']}`",
        f"- defaults changed: `{payload['defaults_changed']}`",
        f"- principle: {payload['principle']}",
        "",
        "## Implemented Flags",
        "",
    ]
    for key, value in payload["implemented_flags"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Benchmark Matrix", "", "| name | depth | compile | dtype | command |", "| --- | --- | --- | --- | --- |"])
    for row in payload["command_matrix"]:
        command = " ".join(str(part) for part in row["command"])
        lines.append(
            f"| `{row['name']}` | `{row['depth_source']}` | `{row['compile_mode']}` | `{row['dtype']}` | `{command}` |"
        )
    lines.extend(["", "## Existing Profile Summaries", ""])
    summaries = payload.get("profile_summaries", [])
    if not summaries:
        lines.append("No profile JSON files were provided; run one or more matrix commands and regenerate this report with `--profile-json`.")
    else:
        lines.extend([
            "| profile | depth | compile | dtype | EdgeTAM p50 | EdgeTAM p90 | fusion FPS | GPU p50/p95 |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ])
        for item in summaries:
            lines.append(
                f"| `{item['path']}` | `{item.get('depth_source')}` | `{item.get('compile_mode')}` | `{item.get('dtype')}` | "
                f"`{item['edgetam_model_ms_p50']:.2f}` | `{item['edgetam_model_ms_p90']:.2f}` | "
                f"`{item['fusion_fps']:.2f}` | `{item['gpu_util_pct_p50']:.1f}/{item['gpu_util_pct_p95']:.1f}` |"
            )
    lines.extend(["", "## Decision Rules", ""])
    for key, value in payload["decision_rules"].items():
        lines.append(f"- `{key}`: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-json", action="append", default=[], help="Existing profile JSON to summarize.")
    parser.add_argument("--json-output", default=str(DEFAULT_JSON))
    parser.add_argument("--md-output", default=str(DEFAULT_MD))
    args = parser.parse_args(argv)

    payload = build_report([Path(path) for path in args.profile_json])
    json_path = Path(args.json_output)
    md_path = Path(args.md_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(payload, md_path)
    print(f"[demo215-mitigation] json={json_path}", flush=True)
    print(f"[demo215-mitigation] md={md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
