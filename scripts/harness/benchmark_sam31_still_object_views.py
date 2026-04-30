from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.harness.object_case_registry import get_raw_object_capture_spec
from scripts.harness.sam31_mask_helper import (
    ColorSource,
    _collect_frame_segments,
    _load_runtime_deps,
    _merge_initial_frame_segments,
    _prepare_camera_output,
    _prepare_session_frames,
    _select_output_indices,
    build_mask_output_path,
    build_sam31_video_predictor,
    discover_color_sources,
    parse_text_prompts,
)


DEFAULT_OBJECT_SET = "still_object"
DEFAULT_ROUND_ID = "round1"
DEFAULT_TEXT_PROMPT = "stuffed animal"
DEFAULT_FRAME_COUNT = 30
DEFAULT_CAMERA_IDS = (0, 1, 2)


@dataclass(frozen=True)
class CameraBenchmarkTiming:
    camera_idx: int
    source_path: str
    frame_count: int
    frame_tokens: list[str]
    tracked_object_count: int
    saved_frame_count: int
    frame_prep_seconds: float
    predictor_build_seconds: float
    start_session_seconds: float
    prompt_seconds: float
    propagate_seconds: float
    mask_write_seconds: float
    close_seconds: float
    segment_seconds: float
    total_seconds: float
    segment_ms_per_frame: float
    total_ms_per_frame: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark SAM 3.1 segmentation speed on 30 RGB frames from each "
            "still-object camera view."
        )
    )
    parser.add_argument("--base_path", type=Path, default=ROOT / "data_collect")
    parser.add_argument(
        "--case_root",
        type=Path,
        default=None,
        help="Explicit raw case root. Defaults to object_case_registry still_object/round1.",
    )
    parser.add_argument("--object_set", type=str, default=DEFAULT_OBJECT_SET)
    parser.add_argument("--round_id", type=str, default=DEFAULT_ROUND_ID)
    parser.add_argument("--camera_ids", nargs="+", type=int, default=list(DEFAULT_CAMERA_IDS))
    parser.add_argument("--frame_count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--source_mode", choices=("frames", "auto"), default="frames")
    parser.add_argument("--text_prompt", type=str, default=DEFAULT_TEXT_PROMPT)
    parser.add_argument("--ann_frame_index", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to write timing summary and optional masks. Defaults under data/experiments/.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional external SAM 3.1 checkpoint path. Otherwise use QQTT_SAM31_CHECKPOINT or HF cache.",
    )
    parser.add_argument("--session_root", type=Path, default=None)
    parser.add_argument("--keep_session_frames", action="store_true")
    parser.add_argument("--write_masks", action="store_true", help="Write masks in addition to timing segmentation.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing optional mask outputs.")
    parser.add_argument("--async_loading_frames", action="store_true")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--max_num_objects", type=int, default=16)
    return parser.parse_args(argv)


def _resolve_case_root(args: argparse.Namespace) -> tuple[Path, str]:
    if args.case_root is not None:
        case_root = args.case_root.resolve()
        return case_root, case_root.name
    spec = get_raw_object_capture_spec(object_set=args.object_set, round_id=args.round_id)
    case_root = (args.base_path / spec.raw_case_name).resolve()
    return case_root, spec.raw_case_name


def _default_output_dir(raw_case_name: str) -> Path:
    return ROOT / "data" / "experiments" / f"sam31_still_object_view_benchmark_{raw_case_name}"


def _limited_frame_source(source: ColorSource, *, frame_count: int) -> ColorSource:
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}.")
    if source.mode != "frames" or not source.frame_paths:
        raise ValueError(
            "This benchmark requires frame-directory RGB sources so it can time exactly "
            f"{frame_count} frames per camera. Pass --source_mode frames."
        )
    if len(source.frame_paths) < frame_count:
        raise ValueError(
            f"Camera {source.camera_idx} has only {len(source.frame_paths)} RGB frames under "
            f"{source.path}; expected at least {frame_count}."
        )
    return ColorSource(
        camera_idx=source.camera_idx,
        mode=source.mode,
        path=source.path,
        frame_paths=list(source.frame_paths[:frame_count]),
    )


def _sync_cuda(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def _seconds_for(callable_obj, *, torch_module: Any) -> tuple[float, Any]:
    _sync_cuda(torch_module)
    start = time.perf_counter()
    result = callable_obj()
    _sync_cuda(torch_module)
    return time.perf_counter() - start, result


def _write_masks(
    *,
    output_dir: Path,
    camera_idx: int,
    frame_token_by_index: dict[int, str],
    video_segments: dict[int, dict[int, Any]],
) -> tuple[float, int]:
    cv2, np, torch_module, _, _, _ = _load_runtime_deps()

    def _write() -> int:
        saved_frame_count = 0
        for frame_idx, masks in sorted(video_segments.items()):
            frame_token = frame_token_by_index.get(int(frame_idx), str(int(frame_idx)))
            if masks:
                saved_frame_count += 1
            for obj_id, mask in masks.items():
                output_path = build_mask_output_path(
                    output_dir,
                    camera_idx=camera_idx,
                    obj_id=obj_id,
                    frame_token=frame_token,
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), np.asarray(mask, dtype=np.uint8) * 255)
        return saved_frame_count

    return _seconds_for(_write, torch_module=torch_module)


def _run_camera_benchmark(
    *,
    source: ColorSource,
    output_dir: Path,
    text_prompt: str,
    checkpoint_path: Path | None,
    ann_frame_index: int,
    session_root: Path | None,
    keep_session_frames: bool,
    write_masks: bool,
    overwrite: bool,
    async_loading_frames: bool,
    compile_model: bool,
    max_num_objects: int,
) -> CameraBenchmarkTiming:
    _, np, torch_module, _, _, _ = _load_runtime_deps()
    prompts = parse_text_prompts(text_prompt)
    if not prompts:
        raise ValueError("text_prompt must contain at least one non-empty prompt.")

    if write_masks:
        _prepare_camera_output(output_dir, camera_idx=source.camera_idx, overwrite=overwrite)

    session_base_dir = (
        session_root.resolve()
        if session_root is not None
        else output_dir / "_sam31_benchmark_session_frames"
    )
    session_base_dir.mkdir(parents=True, exist_ok=True)
    session_dir = Path(tempfile.mkdtemp(prefix=f"cam{int(source.camera_idx)}_", dir=str(session_base_dir)))

    frame_prep_seconds, frame_token_by_index = _seconds_for(
        lambda: _prepare_session_frames(source, session_dir=session_dir),
        torch_module=torch_module,
    )

    predictor_build_seconds, predictor_result = _seconds_for(
        lambda: build_sam31_video_predictor(
            checkpoint_path=checkpoint_path,
            async_loading_frames=async_loading_frames,
            compile_model=compile_model,
            max_num_objects=max_num_objects,
        ),
        torch_module=torch_module,
    )
    predictor, _ = predictor_result

    session_id = None
    selected_obj_to_label: dict[int, str] = {}
    initial_frame_segments: dict[int, Any] = {}
    start_session_seconds = 0.0
    prompt_seconds = 0.0
    propagate_seconds = 0.0
    close_seconds = 0.0
    video_segments: dict[int, dict[int, Any]] = {}

    try:
        start_session_seconds, session_response = _seconds_for(
            lambda: predictor.handle_request(
                {"type": "start_session", "resource_path": str(session_dir)}
            ),
            torch_module=torch_module,
        )
        session_id = session_response["session_id"]

        def _add_prompts() -> None:
            for prompt_idx, prompt in enumerate(prompts):
                response = predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": int(ann_frame_index),
                        "text": prompt,
                    }
                )
                outputs = response["outputs"]
                candidate_ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
                if candidate_ids.size == 0:
                    continue

                keep_all_instances = len(prompts) == 1 or prompt_idx > 0
                selected_indices = _select_output_indices(outputs, keep_all_instances=keep_all_instances)
                selected_ids = {int(candidate_ids[idx]) for idx in selected_indices}

                for candidate_id in candidate_ids.tolist():
                    candidate_id = int(candidate_id)
                    if candidate_id in selected_ids:
                        selected_obj_to_label[candidate_id] = prompt
                        continue
                    predictor.handle_request(
                        {
                            "type": "remove_object",
                            "session_id": session_id,
                            "frame_index": int(ann_frame_index),
                            "obj_id": candidate_id,
                        }
                    )

                initial_frame_segments.update(
                    _collect_frame_segments(outputs, allowed_obj_ids=selected_ids)
                )

        prompt_seconds, _ = _seconds_for(_add_prompts, torch_module=torch_module)
        if not selected_obj_to_label:
            raise RuntimeError(
                f"SAM 3.1 did not register any object for prompt `{text_prompt}` "
                f"for camera {source.camera_idx}."
            )

        tracked_obj_ids = set(selected_obj_to_label)

        def _propagate() -> dict[int, dict[int, Any]]:
            segments: dict[int, dict[int, Any]] = {}
            for response in predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "start_frame_index": int(ann_frame_index),
                    "propagation_direction": "forward",
                }
            ):
                frame_idx = int(response["frame_index"])
                segments[frame_idx] = _collect_frame_segments(
                    response["outputs"],
                    allowed_obj_ids=tracked_obj_ids,
                )
            _merge_initial_frame_segments(
                segments,
                ann_frame_index=int(ann_frame_index),
                initial_frame_segments=initial_frame_segments,
            )
            return segments

        propagate_seconds, video_segments = _seconds_for(_propagate, torch_module=torch_module)
    finally:
        def _close() -> None:
            if session_id is not None:
                predictor.handle_request(
                    {
                        "type": "close_session",
                        "session_id": session_id,
                        "run_gc_collect": True,
                    }
                )
            if hasattr(predictor, "shutdown"):
                predictor.shutdown()

        close_seconds, _ = _seconds_for(_close, torch_module=torch_module)

    if write_masks:
        mask_write_seconds, saved_frame_count = _write_masks(
            output_dir=output_dir,
            camera_idx=source.camera_idx,
            frame_token_by_index=frame_token_by_index,
            video_segments=video_segments,
        )
    else:
        mask_write_seconds = 0.0
        saved_frame_count = sum(1 for masks in video_segments.values() if masks)

    if not keep_session_frames:
        shutil.rmtree(session_dir, ignore_errors=True)
        if session_root is None and session_base_dir.exists() and not any(session_base_dir.iterdir()):
            session_base_dir.rmdir()

    frame_count = len(frame_token_by_index)
    segment_seconds = prompt_seconds + propagate_seconds
    total_seconds = (
        frame_prep_seconds
        + predictor_build_seconds
        + start_session_seconds
        + prompt_seconds
        + propagate_seconds
        + mask_write_seconds
        + close_seconds
    )
    return CameraBenchmarkTiming(
        camera_idx=int(source.camera_idx),
        source_path=str(source.path.resolve()),
        frame_count=int(frame_count),
        frame_tokens=[frame_token_by_index[idx] for idx in sorted(frame_token_by_index)],
        tracked_object_count=int(len(selected_obj_to_label)),
        saved_frame_count=int(saved_frame_count),
        frame_prep_seconds=float(frame_prep_seconds),
        predictor_build_seconds=float(predictor_build_seconds),
        start_session_seconds=float(start_session_seconds),
        prompt_seconds=float(prompt_seconds),
        propagate_seconds=float(propagate_seconds),
        mask_write_seconds=float(mask_write_seconds),
        close_seconds=float(close_seconds),
        segment_seconds=float(segment_seconds),
        total_seconds=float(total_seconds),
        segment_ms_per_frame=float(segment_seconds * 1000.0 / max(1, frame_count)),
        total_ms_per_frame=float(total_seconds * 1000.0 / max(1, frame_count)),
    )


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def build_average_summary(camera_timings: Sequence[CameraBenchmarkTiming]) -> dict[str, float]:
    return {
        "camera_count": float(len(camera_timings)),
        "segment_seconds_per_camera_mean": _mean([item.segment_seconds for item in camera_timings]),
        "segment_ms_per_frame_mean": _mean([item.segment_ms_per_frame for item in camera_timings]),
        "total_seconds_per_camera_mean": _mean([item.total_seconds for item in camera_timings]),
        "total_ms_per_frame_mean": _mean([item.total_ms_per_frame for item in camera_timings]),
        "predictor_build_seconds_mean": _mean([item.predictor_build_seconds for item in camera_timings]),
        "propagate_seconds_mean": _mean([item.propagate_seconds for item in camera_timings]),
    }


def _print_summary(camera_timings: Sequence[CameraBenchmarkTiming], averages: dict[str, float]) -> None:
    print("camera | frames | prompt_s | propagate_s | segment_ms/frame | total_s | total_ms/frame")
    for item in camera_timings:
        print(
            f"{item.camera_idx:>6} | {item.frame_count:>6} | "
            f"{item.prompt_seconds:>8.3f} | {item.propagate_seconds:>11.3f} | "
            f"{item.segment_ms_per_frame:>16.2f} | {item.total_seconds:>7.3f} | "
            f"{item.total_ms_per_frame:>14.2f}"
        )
    print(
        "average segment: "
        f"{averages['segment_ms_per_frame_mean']:.2f} ms/frame, "
        f"{averages['segment_seconds_per_camera_mean']:.3f} s/camera"
    )
    print(
        "average total: "
        f"{averages['total_ms_per_frame_mean']:.2f} ms/frame, "
        f"{averages['total_seconds_per_camera_mean']:.3f} s/camera"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    case_root, raw_case_name = _resolve_case_root(args)
    if not case_root.is_dir():
        raise FileNotFoundError(f"Missing raw case root: {case_root}")

    output_dir = args.output_dir.resolve() if args.output_dir else _default_output_dir(raw_case_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = discover_color_sources(case_root, camera_ids=args.camera_ids, source_mode=args.source_mode)
    camera_timings: list[CameraBenchmarkTiming] = []
    for camera_idx in [int(item) for item in args.camera_ids]:
        limited_source = _limited_frame_source(sources[camera_idx], frame_count=int(args.frame_count))
        print(f"[sam31-bench] camera={camera_idx} frames={args.frame_count} source={limited_source.path}")
        timing = _run_camera_benchmark(
            source=limited_source,
            output_dir=output_dir,
            text_prompt=args.text_prompt,
            checkpoint_path=args.checkpoint,
            ann_frame_index=args.ann_frame_index,
            session_root=args.session_root,
            keep_session_frames=args.keep_session_frames,
            write_masks=args.write_masks,
            overwrite=args.overwrite,
            async_loading_frames=args.async_loading_frames,
            compile_model=args.compile_model,
            max_num_objects=args.max_num_objects,
        )
        camera_timings.append(timing)

    averages = build_average_summary(camera_timings)
    result = {
        "case_root": str(case_root),
        "raw_case_name": raw_case_name,
        "object_set": args.object_set,
        "round_id": args.round_id,
        "camera_ids": [int(item) for item in args.camera_ids],
        "frame_count_requested": int(args.frame_count),
        "text_prompt": args.text_prompt,
        "parsed_prompts": parse_text_prompts(args.text_prompt),
        "ann_frame_index": int(args.ann_frame_index),
        "write_masks": bool(args.write_masks),
        "camera_timings": [asdict(item) for item in camera_timings],
        "averages": averages,
    }
    summary_path = output_dir / "sam31_still_object_view_benchmark.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    _print_summary(camera_timings, averages)
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
