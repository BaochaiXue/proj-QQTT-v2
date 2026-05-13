#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_v2_2.render_fastpath import (
    RENDER_BACKENDS,
    RenderMicroProfileRecord,
    RenderMicroProfiler,
    write_render_profile_summary,
)


def _load_packet(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "points" in data and "colors" in data:
        return np.asarray(data["points"], dtype=np.float32), np.asarray(data["colors"], dtype=np.uint8)
    points_parts: list[np.ndarray] = []
    colors_parts: list[np.ndarray] = []
    for prefix in ("object", "controller"):
        point_key = f"{prefix}_points"
        color_key = f"{prefix}_colors"
        if point_key in data and color_key in data:
            points_parts.append(np.asarray(data[point_key], dtype=np.float32))
            colors_parts.append(np.asarray(data[color_key], dtype=np.uint8))
    if points_parts and colors_parts:
        return np.concatenate(points_parts, axis=0), np.concatenate(colors_parts, axis=0)
    raise ValueError(f"{path} does not contain points/colors arrays")


def _load_packets(packet_dir: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    if not packet_dir.exists():
        return []
    packets: list[tuple[np.ndarray, np.ndarray]] = []
    for path in sorted(packet_dir.glob("*.npz")):
        packets.append(_load_packet(path))
    return packets


def _synthetic_packets(count: int = 100, points_per_packet: int = 60_000) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(22)
    packets: list[tuple[np.ndarray, np.ndarray]] = []
    for idx in range(count):
        n = points_per_packet + (idx % 5) * 512
        points = rng.normal(0.0, 0.25, size=(n, 3)).astype(np.float32)
        points[:, 2] += 0.8
        colors = rng.integers(0, 255, size=(n, 3), dtype=np.uint8)
        packets.append((points, colors))
    return packets


def _benchmark_backend(
    *,
    backend: str,
    packets: Sequence[tuple[np.ndarray, np.ndarray]],
    duration_s: float,
    target_fps: float,
) -> dict:
    profiler = RenderMicroProfiler()
    interval_s = 1.0 / max(float(target_fps), 1.0)
    deadline_s = time.perf_counter() + max(float(duration_s), 0.0)
    idx = 0
    displayed = 0
    while time.perf_counter() < deadline_s or displayed == 0:
        points, colors = packets[idx % len(packets)]
        start_s = time.perf_counter()
        format_start_s = time.perf_counter()
        points_f32 = np.ascontiguousarray(points, dtype=np.float32)
        colors_f32 = np.empty(colors.shape, dtype=np.float32)
        np.multiply(colors, np.float32(1.0 / 255.0), out=colors_f32, casting="unsafe")
        cpu_format_ms = (time.perf_counter() - format_start_s) * 1000.0
        total_ms = (time.perf_counter() - start_s) * 1000.0
        profiler.record(
            RenderMicroProfileRecord(
                render_packet_id=idx,
                points_count=int(points_f32.shape[0]),
                colors_count=int(colors_f32.shape[0]),
                cpu_format_ms=float(cpu_format_ms),
                render_total_ms=float(total_ms),
                backend=backend,
                backend_effective="headless-format-only",
                extra={"quality_same_points": True},
            )
        )
        displayed += 1
        idx += 1
        sleep_s = interval_s - (time.perf_counter() - start_s)
        if sleep_s > 0:
            time.sleep(sleep_s)
    summary = profiler.summary()
    summary["backend"] = backend
    summary["headless_format_only"] = True
    summary["render_fps"] = float(displayed / max(float(duration_s), 1e-6))
    summary["quality_same_points"] = True
    return {"backend": backend, "summary": summary, "records": profiler.records()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Headless Demo 2.2 render packet replay microbenchmark.")
    parser.add_argument("--packet-dir", type=Path, default=Path("result/demo22_render_packets_controller_object_100"))
    parser.add_argument("--render-backends", nargs="+", choices=RENDER_BACKENDS, default=list(RENDER_BACKENDS))
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--output-md", type=Path, default=Path("docs/generated/demo22_render_backend_microbenchmark.md"))
    parser.add_argument("--output-json", type=Path, default=Path("docs/generated/demo22_render_backend_microbenchmark.json"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    packets = _load_packets(args.packet_dir)
    source = str(args.packet_dir)
    if not packets:
        packets = _synthetic_packets()
        source = "synthetic"
    results = [
        _benchmark_backend(
            backend=str(backend),
            packets=packets,
            duration_s=float(args.duration_s),
            target_fps=float(args.target_fps),
        )
        for backend in args.render_backends
    ]
    records = [record for result in results for record in result["records"]]
    payload = write_render_profile_summary(
        records=records,
        output_json=args.output_json,
        output_md=args.output_md,
        title="Demo 2.2 render backend headless microbenchmark",
    )
    payload["source"] = source
    payload["backend_results"] = results
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"source": source, "output_json": str(args.output_json), "output_md": str(args.output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
