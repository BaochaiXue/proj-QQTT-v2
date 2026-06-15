#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_process.visualization.experiments.sam21_pcd_overlay_panel import (
    DEFAULT_DEPTH_SCALE_OVERRIDE_M_PER_UNIT,
    render_fused_pcd_overlay_2x3_gif,
)
from data_process.visualization.io_artifacts import write_json


CASE_KEY = "sloth_set_2_motion_ffs"
CASE_LABEL = "Sloth Set 2 Motion FFS"
VARIANT_KEY = "hf_edgetam_streaming_mask"
VARIANT_LABEL = "HF EdgeTAM streaming"
DEFAULT_RESULT_ROOT = ROOT / "result/sloth_set_2_motion_ffs_hf_edgetam_streaming_pcd_xor"
DEFAULT_CASE_DIR = ROOT / "data/different_types/sloth_set_2_motion_ffs"
DEFAULT_SAM31_MASK_ROOT = DEFAULT_RESULT_ROOT / "sam31_masks"
DEFAULT_EDGETAM_MASK_ROOT = DEFAULT_RESULT_ROOT / "hf_edgetam_streaming/masks/sloth_set_2_motion_ffs/mask"
DEFAULT_STREAMING_RESULTS_JSON = ROOT / "docs/generated/sloth_set_2_motion_ffs_hf_edgetam_streaming_results.json"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULT_ROOT / "pcd_xor"
DEFAULT_DOC_MD = ROOT / "docs/generated/sloth_set_2_motion_ffs_hf_edgetam_streaming_pcd_xor_benchmark.md"
DEFAULT_DOC_JSON = ROOT / "docs/generated/sloth_set_2_motion_ffs_hf_edgetam_streaming_pcd_xor_results.json"
DEFAULT_OUTPUT_NAME = "sloth_set_2_motion_ffs_hf_edgetam_streaming_pcd_xor"


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _latency_value(job: Mapping[str, Any]) -> float:
    subsequent = job.get("subsequent_frame_latency_ms", {})
    if isinstance(subsequent, Mapping) and subsequent.get("median") is not None:
        return float(subsequent["median"])
    frames = [float(item.get("frame_total_ms", 0.0)) for item in job.get("frames", [])]
    if len(frames) > 1:
        frames = frames[1:]
    return float(sum(frames) / max(1, len(frames))) if frames else 0.0


def load_streaming_timing_records(path: str | Path, *, variant_key: str = VARIANT_KEY) -> list[dict[str, Any]]:
    result_path = _resolve_path(path)
    if not result_path.is_file():
        return []
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for job in payload.get("jobs", []):
        if job.get("status") != "pass" or str(job.get("prompt_mode")) != "mask":
            continue
        records.append(
            {
                "checkpoint_key": str(variant_key),
                "case_key": str(job.get("case_key", "")),
                "camera_idx": int(job.get("camera_idx", 0)),
                "inference_ms_per_frame": _latency_value(job),
                "fps": float(job.get("end_to_end_streaming_fps", 0.0)),
            }
        )
    return records


def write_sloth_set2_report(markdown_path: str | Path, summary: Mapping[str, Any]) -> None:
    aggregate = summary.get("aggregate", {}).get(VARIANT_KEY, {})
    timing = summary.get("timing_summary", {}).get(VARIANT_KEY, {})
    lines = [
        "# Sloth Set 2 HF EdgeTAM Streaming PCD XOR",
        "",
        "## Output",
        "",
        f"- GIF: `{summary.get('gif_path')}`",
        f"- first frame: `{summary.get('first_frame_path')}`",
        f"- first-frame PLY dir: `{summary.get('first_frame_ply_dir')}`",
        f"- case: `{summary.get('case_key')}`",
        f"- frames: `{summary.get('frames')}`",
        "",
        "## Render Contract",
        "",
        "- reference: SAM3.1 masks generated from local case frames",
        "- candidate: HF EdgeTAMVideo streaming masks initialized by SAM3.1 frame-0 mask prompt",
        "- EdgeTAM input contract: one PNG frame at a time; no MP4, full video path, or offline video-folder input",
        "- rows: HF EdgeTAM streaming",
        "- columns: cam0/cam1/cam2 original camera pinhole views",
        "- fused PCD: three masked camera RGB point clouds fused before rendering",
        "- overlap: RGB point color",
        "- red: SAM3.1-only points",
        "- cyan: EdgeTAM-only points",
        "",
        "## Aggregate Metrics",
        "",
        "| variant | mean 2D IoU | min 2D IoU | mean raw pIoU | mean post pIoU | mean output pts | mean ms/f | mean FPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {VARIANT_KEY} | "
            f"{float(aggregate.get('mask_iou', {}).get('mean', 0.0)):.4f} | "
            f"{float(aggregate.get('mask_iou', {}).get('min', 0.0)):.4f} | "
            f"{float(aggregate.get('raw_point_iou', {}).get('mean', 0.0)):.4f} | "
            f"{float(aggregate.get('postprocess_point_iou', {}).get('mean', 0.0)):.4f} | "
            f"{float(aggregate.get('output_point_count', {}).get('mean', 0.0)):.0f} | "
            f"{float(timing.get('mean_inference_ms_per_frame', 0.0)):.2f} | "
            f"{float(timing.get('mean_fps', 0.0)):.2f} |"
        ),
        "",
    ]
    path = _resolve_path(markdown_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    timing_records = load_streaming_timing_records(args.streaming_results_json)
    summary = render_fused_pcd_overlay_2x3_gif(
        root=ROOT,
        case_dir=args.case_dir,
        output_dir=args.output_dir,
        case_key=CASE_KEY,
        case_label=CASE_LABEL,
        text_prompt=args.text_prompt,
        sam31_mask_root=args.sam31_mask_root,
        variant_roots={VARIANT_KEY: args.edgetam_mask_root},
        variants=((VARIANT_KEY, VARIANT_LABEL),),
        timing_records=timing_records,
        frames=args.frames,
        gif_fps=int(args.gif_fps),
        tile_width=int(args.tile_width),
        tile_height=int(args.tile_height),
        row_label_width=int(args.row_label_width),
        depth_scale_override_m_per_unit=float(args.depth_scale_override_m_per_unit),
        max_points_per_camera=args.max_points_per_camera,
        max_points_per_render=args.max_points_per_render,
        output_name=DEFAULT_OUTPUT_NAME,
    )
    summary["streaming_contract"] = {
        "frame_by_frame_streaming": True,
        "offline_video_input_used": False,
        "frame_source": "png_loop",
        "edge_tam_prompt": "sam31_frame0_mask",
    }
    summary["streaming_results_json"] = str(_resolve_path(args.streaming_results_json))

    json_path = _resolve_path(args.doc_json)
    write_json(json_path, summary)
    md_path = _resolve_path(args.doc_md)
    write_sloth_set2_report(md_path, summary)
    write_json(
        _resolve_path(args.output_dir) / "summary.json",
        {**summary, "docs": {"benchmark_md": str(md_path), "benchmark_json": str(json_path)}},
    )
    summary["docs"] = {"benchmark_md": str(md_path), "benchmark_json": str(json_path)}
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Sloth Set 2 HF EdgeTAM streaming fused-PCD XOR GIF against SAM3.1."
    )
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--sam31-mask-root", type=Path, default=DEFAULT_SAM31_MASK_ROOT)
    parser.add_argument("--edgetam-mask-root", type=Path, default=DEFAULT_EDGETAM_MASK_ROOT)
    parser.add_argument("--text-prompt", default="stuffed animal")
    parser.add_argument("--streaming-results-json", type=Path, default=DEFAULT_STREAMING_RESULTS_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames", type=int, help="Limit frames for debug runs. Omit for all frames.")
    parser.add_argument("--gif-fps", type=int, default=6)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--tile-height", type=int, default=180)
    parser.add_argument("--row-label-width", type=int, default=170)
    parser.add_argument("--depth-scale-override-m-per-unit", type=float, default=DEFAULT_DEPTH_SCALE_OVERRIDE_M_PER_UNIT)
    parser.add_argument("--max-points-per-camera", type=int)
    parser.add_argument("--max-points-per-render", type=int, default=100_000)
    parser.add_argument("--doc-md", type=Path, default=DEFAULT_DOC_MD)
    parser.add_argument("--doc-json", type=Path, default=DEFAULT_DOC_JSON)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(f"[sloth-set2-pcd-xor] gif: {summary['gif_path']}", flush=True)
    print(f"[sloth-set2-pcd-xor] first frame: {summary['first_frame_path']}", flush=True)
    print(f"[sloth-set2-pcd-xor] report: {summary['docs']['benchmark_md']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
