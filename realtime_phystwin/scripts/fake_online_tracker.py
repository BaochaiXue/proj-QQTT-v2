#!/usr/bin/env python3
"""Replay an existing final_data.pkl as online tracker chunks.

This script is a fake producer for online-training development. It reads the
offline processed `final_data.pkl`, writes consecutive frame chunks, and updates
an atomic manifest after each chunk is committed.
"""

import argparse
import json
import os
import pickle
import shutil
import time
from pathlib import Path


TIME_KEYS = (
    "object_points",
    "object_colors",
    "object_visibilities",
    "object_motions_valid",
    "controller_points",
    "asap_object_points_filled",
    "asap_surface_points",
    "asap_interior_points",
)

STATIC_KEYS = (
    "surface_points",
    "interior_points",
)


def atomic_pickle_dump(obj, path):
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def atomic_json_dump(obj, path):
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def infer_frame_count(data):
    for key in ("object_points", "controller_points"):
        if key in data:
            return int(data[key].shape[0])
    raise KeyError("Cannot infer frame count: missing object_points/controller_points")


def validate_time_arrays(data, frame_count):
    required = (
        "object_points",
        "object_colors",
        "object_visibilities",
        "object_motions_valid",
        "controller_points",
    )
    for key in required:
        if key not in data:
            raise KeyError(f"final_data.pkl missing required key: {key}")
        if int(data[key].shape[0]) != frame_count:
            raise ValueError(
                f"{key} has {data[key].shape[0]} frames, expected {frame_count}"
            )

    for key in ("asap_object_points_filled", "asap_surface_points", "asap_interior_points"):
        if key in data and data[key] is not None and int(data[key].shape[0]) != frame_count:
            raise ValueError(
                f"{key} has {data[key].shape[0]} frames, expected {frame_count}"
            )


def take_source_frames(value, source_frame_indices):
    try:
        return value[source_frame_indices]
    except TypeError:
        return [value[int(idx)] for idx in source_frame_indices]


def build_chunk(
    data,
    case_name,
    chunk_id,
    start_frame,
    end_frame,
    include_static,
    source_frame_indices=None,
):
    if source_frame_indices is None:
        source_frame_indices = list(range(int(start_frame), int(end_frame)))
    source_frame_indices = [int(idx) for idx in source_frame_indices]

    chunk = {
        "case_name": case_name,
        "chunk_id": int(chunk_id),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "source_frame_indices": source_frame_indices,
    }

    for key in TIME_KEYS:
        value = data.get(key)
        if value is not None:
            chunk[key] = take_source_frames(value, source_frame_indices)

    if include_static:
        for key in STATIC_KEYS:
            value = data.get(key)
            if value is not None:
                chunk[key] = value

    return chunk


def write_manifest(
    output_dir,
    case_name,
    status,
    chunk_size,
    frame_count,
    latest_chunk,
    latest_frame,
    version,
    extra=None,
):
    manifest = {
        "case_name": case_name,
        "status": status,
        "chunk_size": int(chunk_size),
        "num_frames_total": int(frame_count),
        "latest_committed_chunk": int(latest_chunk),
        "latest_committed_frame": int(latest_frame),
        "version": int(version),
    }
    if extra is not None:
        manifest.update(extra)
    atomic_json_dump(manifest, Path(output_dir) / "manifest.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay final_data.pkl into online tracker chunks."
    )
    parser.add_argument(
        "case",
        nargs="?",
        default=None,
        help="Case name. With --base_path, reads BASE_PATH/CASE/final_data.pkl.",
    )
    parser.add_argument(
        "--base_path",
        default="data/different_types",
        help="Offline data base path used with case_name.",
    )
    parser.add_argument(
        "--online_base_path",
        default="online_data",
        help="Online output base path used with case_name.",
    )
    parser.add_argument("--input", default=None, help="Path to final_data.pkl")
    parser.add_argument("--output", default=None, help="Output online data dir")
    parser.add_argument(
        "--manifest_case_name",
        default=None,
        help="Override case_name written into chunk metadata and manifest.",
    )
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--sleep_sec", type=float, default=0.5)
    parser.add_argument(
        "--frame_step",
        type=int,
        default=1,
        help=(
            "Replay every Nth source frame as one online frame. "
            "For example, --frame_step 5 emits source frames 0,5,10,..."
        ),
    )
    parser.add_argument(
        "--step_on_key",
        action="store_true",
        help="Wait for Enter before committing the next chunk.",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="First frame to replay, inclusive.",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=None,
        help="Last frame to replay, exclusive. Defaults to all frames.",
    )
    parser.add_argument(
        "--include_static_in_chunks",
        action="store_true",
        help="Also write surface_points/interior_points in every chunk.",
    )
    parser.add_argument(
        "--clear_output",
        action="store_true",
        help="Remove the existing output directory before replaying.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive")
    if args.sleep_sec < 0:
        raise ValueError("--sleep_sec must be non-negative")
    if args.frame_step <= 0:
        raise ValueError("--frame_step must be positive")

    metadata_case_name = args.manifest_case_name or args.case

    if args.input is None:
        if args.case is None:
            raise ValueError("Provide a case_name or --input")
        input_path = Path(args.base_path) / args.case / "final_data.pkl"
    else:
        input_path = Path(args.input)

    if args.output is None:
        if metadata_case_name is None:
            metadata_case_name = input_path.parent.name
        output_dir = Path(args.online_base_path) / metadata_case_name
    else:
        output_dir = Path(args.output)

    case_name = metadata_case_name or input_path.parent.name
    chunks_dir = output_dir / "chunks"

    with open(input_path, "rb") as f:
        data = pickle.load(f)

    frame_count = infer_frame_count(data)
    validate_time_arrays(data, frame_count)

    source_start_frame = max(0, int(args.start_frame))
    source_end_frame = (
        frame_count if args.end_frame is None else min(frame_count, int(args.end_frame))
    )
    if source_start_frame >= source_end_frame:
        raise ValueError(
            "Invalid replay range: "
            f"start_frame={source_start_frame}, end_frame={source_end_frame}"
        )
    source_frame_indices = list(
        range(source_start_frame, source_end_frame, int(args.frame_step))
    )
    if len(source_frame_indices) == 0:
        raise ValueError("No source frames selected for replay")
    online_frame_count = len(source_frame_indices)
    manifest_extra = {
        "source_num_frames_total": int(frame_count),
        "source_start_frame": int(source_start_frame),
        "source_end_frame": int(source_end_frame),
        "source_frame_step": int(args.frame_step),
        "online_num_frames_total": int(online_frame_count),
    }

    if args.clear_output and output_dir.exists():
        shutil.rmtree(output_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    write_manifest(
        output_dir=output_dir,
        case_name=case_name,
        status="recording",
        chunk_size=args.chunk_size,
        frame_count=online_frame_count,
        latest_chunk=-1,
        latest_frame=0,
        version=0,
        extra=manifest_extra,
    )

    chunk_id = 0
    version = 1
    for chunk_start in range(0, online_frame_count, args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, online_frame_count)
        chunk_source_indices = source_frame_indices[chunk_start:chunk_end]
        chunk = build_chunk(
            data=data,
            case_name=case_name,
            chunk_id=chunk_id,
            start_frame=chunk_start,
            end_frame=chunk_end,
            include_static=args.include_static_in_chunks,
            source_frame_indices=chunk_source_indices,
        )
        chunk_path = chunks_dir / f"chunk_{chunk_id:06d}.pkl"
        atomic_pickle_dump(chunk, chunk_path)

        write_manifest(
            output_dir=output_dir,
            case_name=case_name,
            status="recording",
            chunk_size=args.chunk_size,
            frame_count=online_frame_count,
            latest_chunk=chunk_id,
            latest_frame=chunk_end,
            version=version,
            extra=manifest_extra,
        )
        print(
            f"[fake-tracker] committed chunk {chunk_id:06d}: "
            f"online frames [{chunk_start}, {chunk_end}), "
            f"source frames [{chunk_source_indices[0]}, {chunk_source_indices[-1]}]"
        )
        chunk_id += 1
        version += 1
        if args.step_on_key and chunk_end < online_frame_count:
            input("[fake-tracker] press Enter to commit the next chunk...")
        elif args.sleep_sec > 0 and chunk_end < online_frame_count:
            time.sleep(args.sleep_sec)

    write_manifest(
        output_dir=output_dir,
        case_name=case_name,
        status="finished",
        chunk_size=args.chunk_size,
        frame_count=online_frame_count,
        latest_chunk=chunk_id - 1,
        latest_frame=online_frame_count,
        version=version,
        extra=manifest_extra,
    )
    print(
        f"[fake-tracker] finished: {chunk_id} chunks, "
        f"online frames [0, {online_frame_count}), "
        f"source frames [{source_frame_indices[0]}, {source_frame_indices[-1]}], "
        f"frame_step={args.frame_step}"
    )


if __name__ == "__main__":
    main()
