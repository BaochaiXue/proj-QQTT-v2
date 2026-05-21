#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from qqtt.tracking.backends.tapnextpp_adapter import TAPNextPPAdapter


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "docs/generated/demo31_tapnextpp_model_only"
SUMMARY_SCRIPT = ROOT / "scripts/harness/summarize_demo31_tapnextpp_model_only.py"


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def _csv_strings(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _parse_image_size(value: str) -> tuple[int, int]:
    raw = str(value).strip().lower().replace("x", ",")
    parts = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if len(parts) == 1:
        return (parts[0], parts[0])
    if len(parts) == 2:
        return (parts[0], parts[1])
    raise argparse.ArgumentTypeError("--image-size must be H,W, HxW, or a square size.")


def _csv_bools(value: str) -> tuple[bool, ...]:
    aliases = {"1": True, "true": True, "yes": True, "on": True, "0": False, "false": False, "no": False, "off": False}
    parsed: list[bool] = []
    for part in str(value).split(","):
        item = part.strip().lower()
        if not item:
            continue
        if item not in aliases:
            raise argparse.ArgumentTypeError(f"Unsupported bool value {part!r}")
        parsed.append(bool(aliases[item]))
    return tuple(parsed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark TAPNext++ adapter model-only recurrent updates without RealSense or Open3D.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tapnet-repo-dir", default="external/tapnet")
    parser.add_argument("--tapnextpp-checkpoint", default="checkpoints/tapnextpp/tapnextpp_ckpt.pt")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-sizes", type=_csv_ints, default=(1, 3))
    parser.add_argument("--query-counts", type=_csv_ints, default=(512, 1024, 1365, 2048, 4096))
    parser.add_argument("--image-size", type=_parse_image_size, default=(256, 256))
    parser.add_argument("--autocast-dtypes", type=_csv_strings, default=("fp16", "bf16"))
    parser.add_argument("--compile-modes", type=_csv_bools, default=(False,))
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--measured-frames", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    return parser


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray([float(item) for item in values], dtype=np.float64)
    if arr.size == 0:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "p50": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _points(query_count: int, image_size: tuple[int, int]) -> np.ndarray:
    height, width = int(image_size[0]), int(image_size[1])
    cols = int(np.ceil(np.sqrt(float(query_count) * float(width) / float(max(height, 1)))))
    rows = int(np.ceil(float(query_count) / float(max(cols, 1))))
    ys = np.linspace(4, max(height - 5, 4), max(rows, 1), dtype=np.float32)
    xs = np.linspace(4, max(width - 5, 4), max(cols, 1), dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    points = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)[: int(query_count)]
    return np.ascontiguousarray(points, dtype=np.float32)


def _frame(rng: np.random.Generator, image_size: tuple[int, int], frame_idx: int) -> np.ndarray:
    height, width = int(image_size[0]), int(image_size[1])
    base = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    base[..., 0] = (base[..., 0].astype(np.uint16) + int(frame_idx)) % 255
    return np.ascontiguousarray(base, dtype=np.uint8)


def _stats_from_result(result: Any) -> Mapping[str, Any]:
    stats = getattr(result, "stats", {})
    return stats if isinstance(stats, Mapping) else {}


def _run_update(adapter: TAPNextPPAdapter, *, batch_size: int, frame: np.ndarray) -> Mapping[str, Any]:
    if int(batch_size) == 1:
        result = adapter.update(frame)
        return _stats_from_result(result)
    results = adapter.update_batch({idx: frame for idx in range(int(batch_size))})
    first_key = min(results)
    return _stats_from_result(results[first_key])


def _run_case(
    *,
    args: argparse.Namespace,
    batch_size: int,
    query_count: int,
    autocast_dtype: str,
    compile_model: bool,
) -> dict[str, Any]:
    image_size = tuple(int(item) for item in args.image_size)
    rng = np.random.default_rng(int(args.seed))
    points = _points(int(query_count), image_size)
    adapter = TAPNextPPAdapter(
        device=str(args.device),
        repo_dir=str(args.tapnet_repo_dir),
        checkpoint=str(args.tapnextpp_checkpoint),
        image_size=image_size,
        autocast_dtype=str(autocast_dtype),
        compile_model=bool(compile_model),
    )
    if int(batch_size) == 1:
        adapter.initialize([], points)
    else:
        adapter.initialize_batch({idx: points for idx in range(int(batch_size))})

    first_stats = _run_update(adapter, batch_size=int(batch_size), frame=_frame(rng, image_size, 0))
    for idx in range(int(args.warmup_frames)):
        _run_update(adapter, batch_size=int(batch_size), frame=_frame(rng, image_size, idx + 1))

    measured: dict[str, list[float]] = {
        "recurrent_update_ms": [],
        "preprocess_ms": [],
        "postprocess_ms": [],
        "cuda_event_ms": [],
        "wall_ms": [],
    }
    started_s = time.perf_counter()
    for idx in range(int(args.measured_frames)):
        stats = _run_update(
            adapter,
            batch_size=int(batch_size),
            frame=_frame(rng, image_size, idx + 1 + int(args.warmup_frames)),
        )
        measured["recurrent_update_ms"].append(float(stats.get("model_run_ms", 0.0) or 0.0))
        measured["preprocess_ms"].append(float(stats.get("preprocess_ms", 0.0) or 0.0))
        measured["postprocess_ms"].append(float(stats.get("postprocess_ms", 0.0) or 0.0))
        measured["cuda_event_ms"].append(float(stats.get("cuda_event_ms", stats.get("model_run_ms", 0.0)) or 0.0))
        measured["wall_ms"].append(float(stats.get("wall_ms", 0.0) or 0.0))
    total_measured_s = float(time.perf_counter() - started_s)

    row: dict[str, Any] = {
        "backend": "tapnextpp",
        "batch_size": int(batch_size),
        "query_count_per_view": int(query_count),
        "total_query_count": int(query_count) * int(batch_size),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "autocast_dtype": str(autocast_dtype),
        "compile": bool(compile_model),
        "warmup_frames": int(args.warmup_frames),
        "measured_frames": int(args.measured_frames),
        "device": str(args.device),
        "first_update_ms": float(first_stats.get("wall_ms", 0.0) or 0.0),
        "first_update_model_ms": float(first_stats.get("model_run_ms", 0.0) or 0.0),
        "measured_wall_fps": (
            float(args.measured_frames) / total_measured_s if total_measured_s > 0.0 else 0.0
        ),
    }
    for name, values in measured.items():
        stats = _percentiles(values)
        row[f"{name}_p50"] = float(stats["p50"])
        row[f"{name}_p95"] = float(stats["p95"])
        row[f"{name}_max"] = float(stats["max"])
    return row


def _case_path(output_dir: Path, row: Mapping[str, Any]) -> Path:
    h, w = row.get("image_size", [0, 0])
    compile_tag = "compile_on" if bool(row.get("compile")) else "compile_off"
    return output_dir / (
        f"tapnextpp_model_only_b{int(row['batch_size'])}"
        f"_q{int(row['query_count_per_view'])}_{int(h)}x{int(w)}"
        f"_{row['autocast_dtype']}_{compile_tag}.json"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = [
        {
            "batch_size": int(batch_size),
            "query_count_per_view": int(query_count),
            "autocast_dtype": str(dtype),
            "compile": bool(compile_model),
        }
        for batch_size in tuple(args.batch_sizes)
        for query_count in tuple(args.query_counts)
        for dtype in tuple(args.autocast_dtypes)
        for compile_model in tuple(args.compile_modes)
    ]
    manifest = {
        "backend": "tapnextpp",
        "benchmark": "model_only",
        "image_size": list(args.image_size),
        "warmup_frames": int(args.warmup_frames),
        "measured_frames": int(args.measured_frames),
        "matrix": matrix,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    rows: list[dict[str, Any]] = []
    for entry in matrix:
        print(
            "[tapnextpp-model-only] "
            f"B={entry['batch_size']} q={entry['query_count_per_view']} "
            f"dtype={entry['autocast_dtype']} compile={entry['compile']}"
        )
        row = _run_case(
            args=args,
            batch_size=int(entry["batch_size"]),
            query_count=int(entry["query_count_per_view"]),
            autocast_dtype=str(entry["autocast_dtype"]),
            compile_model=bool(entry["compile"]),
        )
        rows.append(row)
        _case_path(output_dir, row).write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    (output_dir / "raw_rows.json").write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not bool(args.skip_summary):
        import subprocess

        completed = subprocess.run(
            [
                sys.executable,
                str(SUMMARY_SCRIPT),
                "--input-dir",
                str(output_dir),
                "--output-json",
                str(output_dir / "summary.json"),
                "--output-md",
                str(output_dir / "summary.md"),
            ],
            cwd=str(ROOT),
            check=False,
        )
        return int(completed.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
