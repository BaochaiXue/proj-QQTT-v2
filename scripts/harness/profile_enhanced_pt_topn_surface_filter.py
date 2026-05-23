#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "generated" / "enhanced_pt_topn_surface_filter_profile"


def _make_components(*, point_count: int, component_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    counts = [point_count // component_count] * component_count
    for idx in range(point_count % component_count):
        counts[idx] += 1
    points: list[np.ndarray] = []
    for idx, count in enumerate(counts):
        center = np.array([idx * 0.08, 0.0, 1.0], dtype=np.float32)
        jitter = rng.normal(loc=0.0, scale=0.0015, size=(int(count), 3)).astype(np.float32)
        points.append(center[None, :] + jitter)
    return np.ascontiguousarray(np.concatenate(points, axis=0), dtype=np.float32)


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _bench_case(
    *,
    label: str,
    keep_top_n_components: int,
    point_count: int,
    component_count: int,
    repeats: int,
    reuse: bool,
) -> dict[str, Any]:
    from qqtt.demo.pcd_postprocess import COMPONENT_SELECTION_LARGEST_N_PLUS_GAP
    from qqtt.demo.semantic_surface_filter import filter_semantic_surface_points

    points = _make_components(point_count=point_count, component_count=component_count, seed=point_count + component_count)
    cached = None
    if reuse:
        cached = filter_semantic_surface_points(
            points_world=points,
            colors=None,
            enabled=True,
            radius_m=0.01,
            nb_points=1,
            component_voxel_size_m=0.01,
            keep_near_main_gap_m=0.0,
            keep_top_n_components=int(keep_top_n_components),
            component_selection_policy=COMPONENT_SELECTION_LARGEST_N_PLUS_GAP,
            min_component_points=32,
            min_component_ratio=0.0,
        )
    totals: list[float] = []
    radius: list[float] = []
    voxel: list[float] = []
    selection: list[float] = []
    survivor_count = 0
    removed_count = 0
    for _ in range(int(repeats)):
        started_s = time.perf_counter()
        if reuse and cached is not None:
            result = cached
        else:
            result = filter_semantic_surface_points(
                points_world=points,
                colors=None,
                enabled=True,
                radius_m=0.01,
                nb_points=1,
                component_voxel_size_m=0.01,
                keep_near_main_gap_m=0.0,
                keep_top_n_components=int(keep_top_n_components),
                component_selection_policy=COMPONENT_SELECTION_LARGEST_N_PLUS_GAP,
                min_component_points=32,
                min_component_ratio=0.0,
            )
            if reuse:
                cached = result
        elapsed_ms = float((time.perf_counter() - started_s) * 1000.0)
        stats = result.stats
        totals.append(elapsed_ms if reuse and cached is result else float(stats.get("total_ms", elapsed_ms)))
        radius.append(0.0 if reuse and cached is result else float(stats.get("radius_filter_ms", 0.0)))
        voxel.append(0.0 if reuse and cached is result else float(stats.get("voxel_component_ms", 0.0)))
        selection.append(0.0 if reuse and cached is result else float(stats.get("component_selection_ms", 0.0)))
        survivor_count = int(len(result.survivor_indices))
        removed_count = int(point_count - survivor_count)
    return {
        "label": str(label),
        "keep_top_n_components": int(keep_top_n_components),
        "point_count": int(point_count),
        "component_count": int(component_count),
        "repeats": int(repeats),
        "reused_filter_result": bool(reuse),
        "radius_filter_ms": _median(radius),
        "voxel_component_ms": _median(voxel),
        "component_selection_ms": _median(selection),
        "total_ms": _median(totals),
        "survivor_count": int(survivor_count),
        "removed_count": int(removed_count),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark enhanced PT top-N 3D semantic surface filtering.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for label, keep_n in (("object", 1), ("controller", 2)):
        for point_count in (1_000, 5_000, 20_000, 50_000):
            for component_count in (1, 2, 5, 20):
                for reuse in (False, True):
                    rows.append(
                        _bench_case(
                            label=label,
                            keep_top_n_components=keep_n,
                            point_count=point_count,
                            component_count=component_count,
                            repeats=max(1, int(args.repeats)),
                            reuse=reuse,
                        )
                    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"cases": rows}
    (output_dir / "profile.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Enhanced PT Top-N Surface Filter Profile",
        "",
        "| label | N | points | components | reuse | radius ms | voxel ms | selection ms | total ms | survivors | removed |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {keep_top_n_components} | {point_count} | {component_count} | {reused_filter_result} | "
            "{radius_filter_ms:.3f} | {voxel_component_ms:.3f} | {component_selection_ms:.3f} | "
            "{total_ms:.3f} | {survivor_count} | {removed_count} |".format(**row)
        )
    (output_dir / "profile.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
