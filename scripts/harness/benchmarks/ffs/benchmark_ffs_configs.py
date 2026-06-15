from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class BenchmarkFrameInput:
    frame_idx: int
    camera_idx: int
    left_path: str
    right_path: str
    left_image: Any
    right_image: Any
    K_ir_left: Any
    baseline_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Fast-FoundationStereo config tradeoffs on saved aligned stereo pairs. "
            "Measures warmup-adjusted latency/FPS and agreement relative to a reference config. "
            "This is offline PyTorch screening, not live 3-camera realtime validation."
        )
    )
    parser.add_argument("--aligned_root", type=Path, default=ROOT / "data")
    parser.add_argument("--case_ref", required=True, help="Aligned case ref or relative path under aligned_root.")
    parser.add_argument("--camera_idx", type=int, default=0)
    parser.add_argument("--frame_idx", type=int, nargs="+", default=[0])
    parser.add_argument("--ffs_repo", type=Path, required=True)
    parser.add_argument(
        "--model_path",
        type=Path,
        action="append",
        required=True,
        help="Repeat this flag to benchmark multiple checkpoints.",
    )
    parser.add_argument("--scale", type=float, nargs="+", default=[1.0])
    parser.add_argument("--valid_iters", type=int, nargs="+", default=[8])
    parser.add_argument("--max_disp", type=int, nargs="+", default=[192])
    parser.add_argument("--warmup_runs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--target_fps", type=float, nargs="+", default=[15.0, 25.0, 30.0])
    parser.add_argument("--reference_config_id", default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    return parser.parse_args()


def _load_inputs(
    *,
    case_dir: Path,
    camera_idx: int,
    frame_indices: list[int],
) -> tuple[dict[str, Any], list[BenchmarkFrameInput]]:
    import cv2
    import numpy as np
    from data_process.visualization.io_case import load_case_metadata

    metadata = load_case_metadata(case_dir)
    serials = metadata["serial_numbers"]
    if int(camera_idx) < 0 or int(camera_idx) >= len(serials):
        raise IndexError(f"camera_idx={camera_idx} out of range for {len(serials)} cameras.")

    if "K_ir_left" not in metadata or "ir_baseline_m" not in metadata:
        raise ValueError(
            f"Case {case_dir} is missing FFS geometry keys required for benchmarking."
        )

    frame_count = int(metadata["frame_num"])
    inputs: list[BenchmarkFrameInput] = []
    for frame_idx in frame_indices:
        if int(frame_idx) < 0 or int(frame_idx) >= frame_count:
            raise IndexError(f"frame_idx={frame_idx} out of range for frame_count={frame_count}.")
        left_path = case_dir / "ir_left" / str(camera_idx) / f"{frame_idx}.png"
        right_path = case_dir / "ir_right" / str(camera_idx) / f"{frame_idx}.png"
        if not left_path.exists() or not right_path.exists():
            raise FileNotFoundError(
                f"Missing stereo pair for frame_idx={frame_idx}, camera_idx={camera_idx}: "
                f"{left_path} / {right_path}"
            )
        left_image = cv2.imread(str(left_path), cv2.IMREAD_UNCHANGED)
        right_image = cv2.imread(str(right_path), cv2.IMREAD_UNCHANGED)
        if left_image is None or right_image is None:
            raise RuntimeError(f"Failed to load stereo pair for frame_idx={frame_idx}.")
        inputs.append(
            BenchmarkFrameInput(
                frame_idx=int(frame_idx),
                camera_idx=int(camera_idx),
                left_path=str(left_path),
                right_path=str(right_path),
                left_image=np.asarray(left_image),
                right_image=np.asarray(right_image),
                K_ir_left=np.asarray(metadata["K_ir_left"][camera_idx], dtype=np.float32),
                baseline_m=float(metadata["ir_baseline_m"][camera_idx]),
            )
        )
    return metadata, inputs


def _run_config(
    *,
    config: Any,
    ffs_repo: Path,
    inputs: list[BenchmarkFrameInput],
    warmup_runs: int,
    repeats: int,
) -> tuple[dict[str, Any], dict[int, Any]]:
    from data_process.depth_backends import FastFoundationStereoRunner
    from data_process.depth_backends.benchmarking import summarize_latency_samples_ms

    if not inputs:
        raise ValueError("Benchmark inputs must not be empty.")

    init_start_s = time.perf_counter()
    runner = FastFoundationStereoRunner(
        ffs_repo=ffs_repo,
        model_path=config.model_path,
        scale=config.scale,
        valid_iters=config.valid_iters,
        max_disp=config.max_disp,
    )
    init_time_s = float(time.perf_counter() - init_start_s)
    torch = runner.torch
    device_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    warmup_latencies_ms: list[float] = []
    measured_latencies_ms: list[float] = []
    outputs_by_frame: dict[int, Any] = {}
    total_runs = max(0, int(warmup_runs)) + max(1, int(repeats)) * len(inputs)
    for run_idx in range(total_runs):
        frame_input = inputs[run_idx % len(inputs)]
        torch.cuda.synchronize()
        start_s = time.perf_counter()
        output = runner.run_pair(
            frame_input.left_image,
            frame_input.right_image,
            K_ir_left=frame_input.K_ir_left,
            baseline_m=frame_input.baseline_m,
        )
        torch.cuda.synchronize()
        latency_ms = float((time.perf_counter() - start_s) * 1000.0)
        if run_idx < int(warmup_runs):
            warmup_latencies_ms.append(latency_ms)
            continue
        measured_latencies_ms.append(latency_ms)
        outputs_by_frame[int(frame_input.frame_idx)] = output["depth_ir_left_m"]

    memory_allocated_mb = float(torch.cuda.memory_allocated() / (1024.0 * 1024.0))
    peak_memory_allocated_mb = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    latency_summary = summarize_latency_samples_ms(measured_latencies_ms)
    result = {
        "config": config.to_dict(),
        "device_name": device_name,
        "init_time_s": init_time_s,
        "warmup_runs": int(warmup_runs),
        "repeats": int(repeats),
        "warmup_latency_ms": warmup_latencies_ms,
        "latency_ms": measured_latencies_ms,
        "latency_summary": latency_summary,
        "memory_allocated_mb": memory_allocated_mb,
        "peak_memory_allocated_mb": peak_memory_allocated_mb,
    }
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    return result, outputs_by_frame


def _select_reference_config(configs: list[Any], reference_config_id: str | None) -> Any:
    if reference_config_id is None:
        return configs[0]
    for config in configs:
        if str(config.config_id) == str(reference_config_id):
            return config
    raise ValueError(
        f"Could not find --reference_config_id={reference_config_id!r} within "
        f"{[config.config_id for config in configs]}"
    )


def _render_report(
    *,
    summary: dict[str, Any],
) -> str:
    benchmark = summary["benchmark"]
    selection = summary["selection"]
    lines = [
        "# FFS Benchmark Report",
        "",
        f"- Timestamp (UTC): `{benchmark['timestamp_utc']}`",
        f"- Case: `{benchmark['case_dir']}`",
        f"- Camera / frames: `cam{benchmark['camera_idx']}` / `{benchmark['frame_indices']}`",
        f"- GPU: `{benchmark['device_name']}`",
        f"- PyTorch: `{benchmark['torch_version']}`",
        f"- Reference config: `{summary['reference_config']['config_id']}`",
        "",
        "## Tradeoff Summary",
        "",
    ]
    fastest = selection["fastest_overall"]
    if fastest is not None:
        lines.append(
            f"- Fastest overall: `{fastest['config_id']}` "
            f"({fastest['latency_mean_ms']:.1f} ms, {fastest['fps_from_mean']:.1f} FPS)"
        )
    most_reference_like = selection["most_reference_like"]
    if most_reference_like is not None:
        lines.append(
            f"- Most reference-like: `{most_reference_like['config_id']}` "
            f"({most_reference_like['latency_mean_ms']:.1f} ms, {most_reference_like['fps_from_mean']:.1f} FPS, "
            f"median abs diff {most_reference_like['median_abs_depth_diff_m'] * 1000.0:.2f} mm)"
        )
    for target_fps, payload in selection["targets"].items():
        if payload is None:
            lines.append(f"- No config reached `{target_fps}` FPS.")
        else:
            lines.append(
                f"- Best config meeting `{target_fps}` FPS: `{payload['config_id']}` "
                f"({payload['latency_mean_ms']:.1f} ms, {payload['fps_from_mean']:.1f} FPS, "
                f"median abs diff {payload['median_abs_depth_diff_m'] * 1000.0:.2f} mm)"
            )
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Config | Mean ms | FPS | P90 ms | Peak MB | Ref MAD mm | Ref P90 mm | Overlap |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in summary["results"]:
        latency = result["latency_summary"]
        reference = result["reference_metrics"]
        lines.append(
            "| "
            f"`{result['config']['config_id']}` | "
            f"{latency['latency_mean_ms']:.1f} | "
            f"{latency['fps_from_mean']:.1f} | "
            f"{latency['latency_p90_ms']:.1f} | "
            f"{result['peak_memory_allocated_mb']:.0f} | "
            f"{reference['median_abs_depth_diff_m'] * 1000.0:.2f} | "
            f"{reference['p90_abs_depth_diff_m'] * 1000.0:.2f} | "
            f"{reference['overlap_valid_ratio'] * 100.0:.1f}% |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    from data_process.depth_backends.benchmarking import (
        build_tradeoff_summary,
        compute_reference_depth_metrics,
        expand_benchmark_configs,
    )
    from data_process.visualization.io_case import resolve_case_dir

    case_dir = resolve_case_dir(
        aligned_root=Path(args.aligned_root).resolve(),
        case_ref=str(args.case_ref),
    )
    metadata, inputs = _load_inputs(
        case_dir=case_dir,
        camera_idx=int(args.camera_idx),
        frame_indices=[int(frame_idx) for frame_idx in args.frame_idx],
    )
    configs = expand_benchmark_configs(
        model_paths=[Path(model_path).resolve() for model_path in args.model_path],
        scales=[float(scale) for scale in args.scale],
        valid_iters_values=[int(value) for value in args.valid_iters],
        max_disp_values=[int(value) for value in args.max_disp],
    )
    reference_config = _select_reference_config(configs, args.reference_config_id)
    if configs[0].config_id != reference_config.config_id:
        ordered_configs = [reference_config] + [
            config for config in configs if config.config_id != reference_config.config_id
        ]
        configs = ordered_configs

    args.out_dir.mkdir(parents=True, exist_ok=True)

    reference_depth_by_frame: dict[int, Any] | None = None
    results: list[dict[str, Any]] = []
    device_name = None
    torch_version = None
    for config in configs:
        print(f"[bench] {config.config_id}", flush=True)
        result, outputs_by_frame = _run_config(
            config=config,
            ffs_repo=Path(args.ffs_repo).resolve(),
            inputs=inputs,
            warmup_runs=int(args.warmup_runs),
            repeats=int(args.repeats),
        )
        if device_name is None:
            device_name = str(result["device_name"])
        if torch_version is None:
            try:
                import torch

                torch_version = str(torch.__version__)
            except Exception:
                torch_version = "unknown"
        if reference_depth_by_frame is None:
            reference_depth_by_frame = outputs_by_frame

        reference_metrics_per_frame = []
        for frame_input in inputs:
            reference_depth = reference_depth_by_frame[int(frame_input.frame_idx)]
            candidate_depth = outputs_by_frame[int(frame_input.frame_idx)]
            reference_metrics_per_frame.append(
                compute_reference_depth_metrics(reference_depth, candidate_depth)
            )
        reference_metrics = {
            key: float(
                sum(float(item[key]) for item in reference_metrics_per_frame)
                / max(1, len(reference_metrics_per_frame))
            )
            for key in reference_metrics_per_frame[0].keys()
        }
        result["reference_metrics"] = reference_metrics
        results.append(result)

    results.sort(key=lambda item: float(item["latency_summary"]["latency_mean_ms"]))
    summary = {
        "benchmark": {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "case_dir": str(case_dir),
            "camera_idx": int(args.camera_idx),
            "frame_indices": [int(frame_input.frame_idx) for frame_input in inputs],
            "frame_shape_hw": list(inputs[0].left_image.shape[:2]),
            "fps": int(metadata.get("fps", 0)),
            "ffs_repo": str(Path(args.ffs_repo).resolve()),
            "target_fps": [float(value) for value in args.target_fps],
            "warmup_runs": int(args.warmup_runs),
            "repeats": int(args.repeats),
            "python_version": platform.python_version(),
            "torch_version": torch_version or "unknown",
            "device_name": device_name or "unknown",
            "hostname": platform.node(),
            "pid": int(os.getpid()),
        },
        "reference_config": reference_config.to_dict(),
        "selection": build_tradeoff_summary(
            results,
            target_fps_values=[float(value) for value in args.target_fps],
        ),
        "results": results,
    }

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "report.md").write_text(
        _render_report(summary=summary),
        encoding="utf-8",
    )
    print(f"[done] wrote {args.out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
