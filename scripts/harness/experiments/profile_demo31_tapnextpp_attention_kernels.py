#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT_DIR = ROOT / "docs/generated/demo31_tapnextpp_attention_kernels"


def _parse_image_size(value: str | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, str):
        raw = value.strip().lower().replace("x", ",")
        parts = [int(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        parts = [int(item) for item in value]
    if len(parts) == 1:
        return (parts[0], parts[0])
    if len(parts) == 2:
        return (parts[0], parts[1])
    raise argparse.ArgumentTypeError("--image-size must be H,W, HxW, or a square size.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile Demo 3.1 TAPNext++ recurrent attention/kernel path without changing runtime behavior.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tapnet-repo-dir", type=Path, default=ROOT / "external/tapnet")
    parser.add_argument("--tapnextpp-checkpoint", type=Path, default=ROOT / "checkpoints/tapnextpp/tapnextpp_ckpt.pt")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--query-count", type=int, default=1365)
    parser.add_argument("--image-size", type=_parse_image_size, default=(256, 256))
    parser.add_argument("--autocast-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument("--profile-updates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--write-chrome-trace", action="store_true")
    parser.add_argument("--skip-model-load", action="store_true", help="Only report import/backend availability.")
    return parser


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _points(query_count: int, image_size: tuple[int, int]) -> np.ndarray:
    height, width = int(image_size[0]), int(image_size[1])
    cols = int(np.ceil(np.sqrt(float(query_count) * float(width) / float(max(height, 1)))))
    rows = int(np.ceil(float(query_count) / float(max(cols, 1))))
    ys = np.linspace(4, max(height - 5, 4), max(rows, 1), dtype=np.float32)
    xs = np.linspace(4, max(width - 5, 4), max(cols, 1), dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.ascontiguousarray(np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)[:query_count], dtype=np.float32)


def _frame(rng: np.random.Generator, image_size: tuple[int, int]) -> np.ndarray:
    return np.ascontiguousarray(rng.integers(0, 255, size=(image_size[0], image_size[1], 3), dtype=np.uint8))


def _metric(event: Any, names: Iterable[str]) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def _event_row(event: Any) -> dict[str, Any]:
    device_total_us = _metric(event, ("device_time_total", "cuda_time_total"))
    self_device_us = _metric(event, ("self_device_time_total", "self_cuda_time_total"))
    cpu_total_us = _metric(event, ("cpu_time_total",))
    self_cpu_us = _metric(event, ("self_cpu_time_total",))
    return {
        "key": str(event.key),
        "count": int(getattr(event, "count", 0) or 0),
        "device_time_total_ms": float(device_total_us / 1000.0),
        "self_device_time_total_ms": float(self_device_us / 1000.0),
        "cpu_time_total_ms": float(cpu_total_us / 1000.0),
        "self_cpu_time_total_ms": float(self_cpu_us / 1000.0),
    }


def _classify_event(key: str) -> tuple[str, ...]:
    name = key.lower()
    tags: list[str] = []
    if "scaled_dot_product" in name:
        tags.append("scaled_dot_product_attention")
    if "flash" in name or "fmha" in name:
        tags.append("flash_attention")
    if "efficient_attention" in name or "mem_efficient" in name:
        tags.append("mem_efficient_attention")
    if "softmax" in name and "attention" not in tags:
        tags.append("softmax")
    if "einsum" in name:
        tags.append("einsum")
    if "bmm" in name:
        tags.append("bmm")
    if "matmul" in name:
        tags.append("matmul")
    if name.endswith("mm") or "aten::mm" in name or "addmm" in name:
        tags.append("mm")
    if "linear" in name:
        tags.append("linear")
    if "permute" in name or "transpose" in name or "rearrange" in name:
        tags.append("layout_view")
    if "contiguous" in name or "clone" in name:
        tags.append("contiguous_clone")
    if "copy" in name:
        tags.append("copy")
    if "cat" in name:
        tags.append("cat")
    if "gelu" in name:
        tags.append("gelu")
    if "elementwise" in name or "aten::mul" in name or "aten::add" in name or "aten::sigmoid" in name:
        tags.append("elementwise")
    return tuple(tags)


def _summarize_events(events: Sequence[Any]) -> dict[str, Any]:
    rows = [_event_row(event) for event in events]
    total_self_device_ms = float(sum(row["self_device_time_total_ms"] for row in rows))
    tag_totals: dict[str, float] = {}
    tag_counts: Counter[str] = Counter()
    for row in rows:
        for tag in _classify_event(row["key"]):
            tag_totals[tag] = tag_totals.get(tag, 0.0) + float(row["device_time_total_ms"])
            tag_counts[tag] += int(row["count"])
    top_rows = sorted(rows, key=lambda row: row["device_time_total_ms"], reverse=True)[:40]
    filtered_rows = [
        row
        for row in sorted(rows, key=lambda item: item["device_time_total_ms"], reverse=True)
        if _classify_event(row["key"])
    ][:80]
    sdpa_total_ms = float(
        sum(row["device_time_total_ms"] for row in rows if row["key"] == "aten::scaled_dot_product_attention")
    )
    flash_kernel_self_ms = float(
        sum(
            row["self_device_time_total_ms"]
            for row in rows
            if "flash_fwd_kernel" in row["key"].lower() or "fmha" in row["key"].lower()
        )
    )
    return {
        "total_self_device_ms": total_self_device_ms,
        "tag_device_time_total_ms": {key: float(value) for key, value in sorted(tag_totals.items())},
        "tag_counts": dict(tag_counts),
        "top_device_ops": top_rows,
        "filtered_ops": filtered_rows,
        "sdpa_device_time_total_ms": sdpa_total_ms,
        "flash_kernel_self_device_time_ms": flash_kernel_self_ms,
        "flash_attention_detected": any("flash_attention" in _classify_event(row["key"]) for row in rows),
        "scaled_dot_product_attention_detected": any(
            "scaled_dot_product_attention" in _classify_event(row["key"]) for row in rows
        ),
        "mem_efficient_attention_detected": any("mem_efficient_attention" in _classify_event(row["key"]) for row in rows),
        "math_attention_likely": bool(
            not any("flash_attention" in _classify_event(row["key"]) for row in rows)
            and any(row["key"].lower().endswith("bmm") or "aten::bmm" in row["key"].lower() for row in rows)
        ),
    }


def _stack_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "flash_attn_package_available": _module_available("flash_attn"),
        "torch_sdp_flash_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "torch_sdp_mem_efficient_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "torch_sdp_math_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
    }
    if torch.cuda.is_available():
        info["device_count"] = int(torch.cuda.device_count())
        info["devices"] = [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())]
    return info


def _build_adapter(args: argparse.Namespace):
    from qqtt.tracking.backends.tapnextpp_adapter import TAPNextPPAdapter

    return TAPNextPPAdapter(
        device=str(args.device),
        repo_dir=str(args.tapnet_repo_dir),
        checkpoint=str(args.tapnextpp_checkpoint),
        image_size=tuple(int(item) for item in args.image_size),
        autocast_dtype=str(args.autocast_dtype),
        fast_postprocess=True,
    )


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_size = tuple(int(item) for item in args.image_size)
    payload: dict[str, Any] = {
        "profile": "demo31_tapnextpp_attention_kernels",
        "live_runtime_changed": False,
        "device": str(args.device),
        "batch_size": int(args.batch_size),
        "query_count_per_view": int(args.query_count),
        "total_query_count": int(args.batch_size) * int(args.query_count),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "autocast_dtype": str(args.autocast_dtype),
        "stack": _stack_info(),
    }
    if bool(args.skip_model_load):
        payload["status"] = "stack_only"
        return payload

    adapter = _build_adapter(args)
    model = adapter._load_model()
    rng = np.random.default_rng(int(args.seed))
    frame = _frame(rng, image_size)
    frames = [frame for _ in range(int(args.batch_size))]
    video, source_shape = adapter._frames_to_video_tensor(frames, camera_ids=tuple(range(int(args.batch_size))))
    points = _points(int(args.query_count), image_size)
    query = adapter._queries_yx_to_tyx_tensor(points, source_shape_hw=source_shape)[None].repeat(
        int(args.batch_size), 1, 1
    ).contiguous()

    with torch.no_grad(), adapter._autocast_context():
        first = model(video=video, query_points=query)
    state = first[3]
    adapter._sync_cuda_if_needed()

    for _ in range(int(args.warmup_updates)):
        with torch.no_grad(), adapter._autocast_context():
            out = model(video=video, state=state)
        state = out[3]
        adapter._sync_cuda_if_needed()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
        acc_events=True,
    ) as prof:
        for _ in range(int(args.profile_updates)):
            with torch.no_grad(), adapter._autocast_context():
                out = model(video=video, state=state)
            state = out[3]
        adapter._sync_cuda_if_needed()

    events = list(prof.key_averages())
    summary = _summarize_events(events)
    payload.update(
        {
            "status": "ok",
            "warmup_updates": int(args.warmup_updates),
            "profile_updates": int(args.profile_updates),
            "first_tracks_shape": [int(item) for item in tuple(first[0].shape)],
            "recurrent_tracks_shape": [int(item) for item in tuple(out[0].shape)],
            "visible_logits_shape": [int(item) for item in tuple(out[2].shape)],
            "profiler": summary,
            "interpretation": _interpret(summary),
        }
    )
    if bool(args.write_chrome_trace):
        trace_path = output_dir / "trace.json"
        prof.export_chrome_trace(str(trace_path))
        payload["chrome_trace"] = str(trace_path)
    return payload


def _pct(value: float, total: float) -> float:
    return float(value / total * 100.0) if total > 0.0 else 0.0


def _interpret(summary: Mapping[str, Any]) -> dict[str, Any]:
    total = float(summary.get("total_self_device_ms", 0.0) or 0.0)
    tags = summary.get("tag_device_time_total_ms", {})
    sdpa_ms = float(summary.get("sdpa_device_time_total_ms", 0.0) or 0.0)
    flash_ms = float(summary.get("flash_kernel_self_device_time_ms", 0.0) or 0.0)
    if flash_ms <= 0.0:
        flash_ms = float(tags.get("flash_attention", 0.0) or 0.0)
    if sdpa_ms <= 0.0:
        sdpa_ms = float(tags.get("scaled_dot_product_attention", 0.0) or 0.0)
    einsum_ms = float(tags.get("einsum", 0.0) or 0.0)
    linear_ms = float(tags.get("linear", 0.0) or 0.0)
    copy_ms = float(tags.get("copy", 0.0) or 0.0)
    return {
        "uses_scaled_dot_product_attention": bool(summary.get("scaled_dot_product_attention_detected")),
        "uses_flash_attention_kernel": bool(summary.get("flash_attention_detected")),
        "uses_mem_efficient_attention_kernel": bool(summary.get("mem_efficient_attention_detected")),
        "math_attention_fallback_likely": bool(summary.get("math_attention_likely")),
        "flash_attention_device_time_ms": flash_ms,
        "flash_attention_pct_of_self_device_time": _pct(flash_ms, total),
        "scaled_dot_product_attention_device_time_ms": sdpa_ms,
        "scaled_dot_product_attention_pct_of_self_device_time": _pct(sdpa_ms, total),
        "einsum_device_time_ms": einsum_ms,
        "linear_device_time_ms": linear_ms,
        "copy_device_time_ms": copy_ms,
        "attention_is_primary_bottleneck": bool(flash_ms > 0.0 and _pct(flash_ms, total) >= 35.0),
    }


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    interp = payload.get("interpretation", {})
    profiler = payload.get("profiler", {})
    stack = payload.get("stack", {})
    tags = profiler.get("tag_device_time_total_ms", {})
    lines = [
        "# Demo 3.1 TAPNext++ Attention Kernel Profile",
        "",
        "This is a model-only probe. It does not change the live Demo 3.1 backend.",
        "",
        f"- Case: B={payload.get('batch_size')} q={payload.get('query_count_per_view')}/view total={payload.get('total_query_count')}",
        f"- Torch: `{stack.get('torch')}`, CUDA `{stack.get('cuda')}`",
        f"- `flash_attn` package available: `{stack.get('flash_attn_package_available')}`",
        f"- PyTorch SDP flags: flash `{stack.get('torch_sdp_flash_enabled')}`, mem-efficient `{stack.get('torch_sdp_mem_efficient_enabled')}`, math `{stack.get('torch_sdp_math_enabled')}`",
        "",
        "## Answer",
        "",
        f"- Uses `scaled_dot_product_attention`: `{interp.get('uses_scaled_dot_product_attention')}`",
        f"- Uses flash attention kernel: `{interp.get('uses_flash_attention_kernel')}`",
        f"- Uses mem-efficient attention kernel: `{interp.get('uses_mem_efficient_attention_kernel')}`",
        f"- Math attention fallback likely: `{interp.get('math_attention_fallback_likely')}`",
        f"- Attention primary bottleneck: `{interp.get('attention_is_primary_bottleneck')}`",
        f"- De-duplicated SDPA/flash kernel time: `{float(interp.get('scaled_dot_product_attention_device_time_ms', 0.0)):.3f}ms`",
        "",
        "## CUDA Time By Category",
        "",
        "These category totals are profiler aggregates and are not mutually exclusive when parent `aten::` ops contain child kernels.",
        "",
        "| Category | Device time ms |",
        "| --- | ---: |",
    ]
    for key, value in sorted(tags.items(), key=lambda item: float(item[1]), reverse=True):
        lines.append(f"| {key} | {float(value):.3f} |")
    lines.extend(["", "## Top Device Ops", "", "| Op | Count | Device ms | Self device ms |", "| --- | ---: | ---: | ---: |"])
    for row in profiler.get("top_device_ops", [])[:20]:
        lines.append(
            f"| `{row['key']}` | {int(row['count'])} | {float(row['device_time_total_ms']):.3f} | {float(row['self_device_time_total_ms']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The current PyTorch path does select flash attention through SDPA.",
            "- This is PyTorch's flash SDPA kernel, not evidence that the external FlashAttention3 package is installed or used.",
            "- If flash attention is a small fraction of total recurrent time, the next speed work should focus on linear/einsum/state/update kernels rather than only attention selection.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = run_profile(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "summary.json"
    output_md = output_dir / "summary.md"
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    print(json.dumps(payload.get("interpretation", {"status": payload.get("status")}), indent=2, sort_keys=True))
    print(output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
