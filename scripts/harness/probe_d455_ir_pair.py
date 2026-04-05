from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe one D455 and save a raw IR stereo sample.")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--capture_color", type=int, choices=(0, 1), default=1)
    parser.add_argument("--emitter", choices=("on", "off", "auto"), default="auto")
    parser.add_argument("--num_frames", type=int, default=3)
    parser.add_argument("--warmup_frames", type=int, default=30)
    return parser.parse_args()


def intrinsics_to_matrix(intrinsics) -> list[list[float]]:
    return [
        [float(intrinsics.fx), 0.0, float(intrinsics.ppx)],
        [0.0, float(intrinsics.fy), float(intrinsics.ppy)],
        [0.0, 0.0, 1.0],
    ]


def extrinsics_to_matrix(extrinsics) -> list[list[float]]:
    rotation = list(map(float, extrinsics.rotation))
    translation = list(map(float, extrinsics.translation))
    return [
        [rotation[0], rotation[1], rotation[2], translation[0]],
        [rotation[3], rotation[4], rotation[5], translation[1]],
        [rotation[6], rotation[7], rotation[8], translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def translation_norm(extrinsics) -> float:
    tx, ty, tz = map(float, extrinsics.translation)
    return float((tx * tx + ty * ty + tz * tz) ** 0.5)


def try_capture(
    *,
    serial: str,
    out_dir: Path,
    width: int,
    height: int,
    fps: int,
    capture_color: bool,
    emitter: str,
    num_frames: int,
    warmup_frames: int,
) -> dict[str, object]:
    import cv2
    import numpy as np
    import pyrealsense2 as rs

    from scripts.harness.ffs_geometry import write_ffs_intrinsic_file

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
    config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
    if capture_color:
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipeline.start(config)
    try:
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        if emitter != "auto" and depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if emitter == "on" else 0.0)
        emitter_actual = None
        if depth_sensor.supports(rs.option.emitter_enabled):
            emitter_actual = float(depth_sensor.get_option(rs.option.emitter_enabled))
        depth_scale = None
        try:
            depth_scale = float(depth_sensor.get_depth_scale())
        except Exception:
            depth_scale = None

        raw_frames = None
        for _ in range(max(warmup_frames, 0)):
            raw_frames = pipeline.wait_for_frames(3000)

        saved_frames = []
        for _ in range(max(num_frames, 1)):
            raw_frames = pipeline.wait_for_frames(3000)
            ir_left = raw_frames.get_infrared_frame(1)
            ir_right = raw_frames.get_infrared_frame(2)
            color = raw_frames.get_color_frame() if capture_color else None
            if not ir_left or not ir_right or (capture_color and not color):
                continue
            frame_record: dict[str, object] = {
                "ir_left": np.asanyarray(ir_left.get_data()).copy(),
                "ir_right": np.asanyarray(ir_right.get_data()).copy(),
            }
            if capture_color and color:
                frame_record["color"] = np.asanyarray(color.get_data()).copy()
            saved_frames.append(frame_record)

        if not saved_frames:
            raise RuntimeError("No valid frames captured during probe")

        chosen = saved_frames[-1]
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "ir_left.png"), chosen["ir_left"])
        cv2.imwrite(str(out_dir / "ir_right.png"), chosen["ir_right"])

        ir_left_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        ir_right_profile = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        ir_left_intrinsics = ir_left_profile.get_intrinsics()
        ir_right_intrinsics = ir_right_profile.get_intrinsics()
        ir_left_to_right = ir_left_profile.get_extrinsics_to(ir_right_profile)

        metadata: dict[str, object] = {
            "schema_version": "d455_ffs_probe_v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "serial": serial,
            "model_name": device.get_info(rs.camera_info.name),
            "product_line": device.get_info(rs.camera_info.product_line),
            "width": width,
            "height": height,
            "fps": fps,
            "emitter_request": emitter,
            "emitter_actual": emitter_actual,
            "K_ir_left": intrinsics_to_matrix(ir_left_intrinsics),
            "K_ir_right": intrinsics_to_matrix(ir_right_intrinsics),
            "T_ir_left_to_right": extrinsics_to_matrix(ir_left_to_right),
            "ir_baseline_m": translation_norm(ir_left_to_right),
            "depth_scale_m_per_unit": depth_scale,
            "stream_names": ["ir_left", "ir_right"] + (["color"] if capture_color else []),
        }

        np.savetxt(out_dir / "K_ir_left.txt", np.asarray(metadata["K_ir_left"], dtype=np.float32), fmt="%.8f")
        write_ffs_intrinsic_file(
            out_dir / "K_ir_left_ffs.txt",
            np.asarray(metadata["K_ir_left"], dtype=np.float32),
            float(metadata["ir_baseline_m"]),
        )

        preview_tiles = [
            cv2.cvtColor(chosen["ir_left"], cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(chosen["ir_right"], cv2.COLOR_GRAY2BGR),
        ]

        if capture_color and "color" in chosen:
            cv2.imwrite(str(out_dir / "color.png"), chosen["color"])
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            color_intrinsics = color_profile.get_intrinsics()
            ir_left_to_color = ir_left_profile.get_extrinsics_to(color_profile)
            metadata["K_color"] = intrinsics_to_matrix(color_intrinsics)
            metadata["T_ir_left_to_color"] = extrinsics_to_matrix(ir_left_to_color)
            np.savetxt(
                out_dir / "K_color.txt",
                np.asarray(metadata["K_color"], dtype=np.float32),
                fmt="%.8f",
            )
            preview_tiles.append(chosen["color"])

        preview = np.hstack(preview_tiles)
        cv2.imwrite(str(out_dir / "preview_contact_sheet.png"), preview)
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata
    finally:
        pipeline.stop()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    attempt_specs = []
    requested_width = args.width
    requested_height = args.height
    capture_color = bool(args.capture_color)

    attempt_specs.append((requested_width, requested_height, capture_color))
    if (requested_width, requested_height) != (640, 480):
        attempt_specs.append((640, 480, capture_color))
    if capture_color:
        attempt_specs.append((848, 480, False))

    seen = set()
    ordered_attempts = []
    for spec in attempt_specs:
        if spec not in seen:
            ordered_attempts.append(spec)
            seen.add(spec)

    errors: list[str] = []
    for width, height, use_color in ordered_attempts:
        try:
            metadata = try_capture(
                serial=args.serial,
                out_dir=out_dir,
                width=width,
                height=height,
                fps=args.fps,
                capture_color=use_color,
                emitter=args.emitter,
                num_frames=args.num_frames,
                warmup_frames=args.warmup_frames,
            )
            metadata["capture_color"] = use_color
            metadata["probe_profile_used"] = {
                "width": width,
                "height": height,
                "fps": args.fps,
                "capture_color": use_color,
            }
            (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"Saved D455 proof-of-life sample to {out_dir}")
            return 0
        except Exception as exc:
            errors.append(
                f"{width}x{height}@{args.fps} capture_color={int(use_color)} -> {type(exc).__name__}: {exc}"
            )

    joined = "\n".join(errors)
    raise RuntimeError(f"Failed to capture a D455 proof-of-life sample.\n{joined}")


if __name__ == "__main__":
    raise SystemExit(main())
