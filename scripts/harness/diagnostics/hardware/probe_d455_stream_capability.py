from __future__ import annotations

import argparse
import copy
import json
import platform
import shutil
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STABLE_SERIAL_ORDER = [
    "239222300433",
    "239222300781",
    "239222303506",
]

DEFAULT_OUTPUT_ROOT = ROOT / "data" / "ffs_proof_of_life" / "d455_stream_probe"
DEFAULT_DOC_JSON = ROOT / "docs" / "generated" / "d455_stream_probe_results.json"
DEFAULT_DOC_MD = ROOT / "docs" / "generated" / "d455_stream_probe_results.md"

EXPECTATION_SOURCES = [
    {
        "name": "Projection in RealSense SDK 2.0",
        "url": "https://dev.realsenseai.com/docs/projection-in-intel-realsense-sdk-20",
        "expectation": (
            "Each stream has its own pixel and 3D coordinate system, and extrinsics are rigid "
            "transforms in meters between stream coordinate frames."
        ),
    },
    {
        "name": "Intel RealSense D400 Series Datasheet / D455 product page",
        "url": "https://www.realsenseai.com/wp-content/uploads/2023/10/Intel-RealSense-D400-Series-Datasheet-September-2023.pdf",
        "expectation": (
            "D455 is a D400-family stereo device with a nominal 95 mm baseline and USB3-advertised "
            "848x480 and 640x480 operating points."
        ),
    },
    {
        "name": "Fast-FoundationStereo README",
        "url": "https://github.com/NVlabs/Fast-FoundationStereo",
        "expectation": (
            "Stereo inputs should be rectified, undistorted, and use the true left and right images."
        ),
    },
]

STABILITY_THRESHOLDS = {
    "min_fps_ratio": 0.8,
    "max_timestamp_gap_factor": 5.0,
}

STREAM_SET_DEFS: dict[str, list[dict[str, Any]]] = {
    "depth": [
        {"name": "depth", "stream_type": "depth", "index": 0, "format": "z16"},
    ],
    "color": [
        {"name": "color", "stream_type": "color", "index": 0, "format": "bgr8"},
    ],
    "ir_left": [
        {"name": "ir_left", "stream_type": "infrared", "index": 1, "format": "y8"},
    ],
    "ir_right": [
        {"name": "ir_right", "stream_type": "infrared", "index": 2, "format": "y8"},
    ],
    "ir_pair": [
        {"name": "ir_left", "stream_type": "infrared", "index": 1, "format": "y8"},
        {"name": "ir_right", "stream_type": "infrared", "index": 2, "format": "y8"},
    ],
    "rgbd": [
        {"name": "color", "stream_type": "color", "index": 0, "format": "bgr8"},
        {"name": "depth", "stream_type": "depth", "index": 0, "format": "z16"},
    ],
    "rgb_ir_pair": [
        {"name": "color", "stream_type": "color", "index": 0, "format": "bgr8"},
        {"name": "ir_left", "stream_type": "infrared", "index": 1, "format": "y8"},
        {"name": "ir_right", "stream_type": "infrared", "index": 2, "format": "y8"},
    ],
    "depth_ir_pair": [
        {"name": "depth", "stream_type": "depth", "index": 0, "format": "z16"},
        {"name": "ir_left", "stream_type": "infrared", "index": 1, "format": "y8"},
        {"name": "ir_right", "stream_type": "infrared", "index": 2, "format": "y8"},
    ],
    "rgbd_ir_pair": [
        {"name": "color", "stream_type": "color", "index": 0, "format": "bgr8"},
        {"name": "depth", "stream_type": "depth", "index": 0, "format": "z16"},
        {"name": "ir_left", "stream_type": "infrared", "index": 1, "format": "y8"},
        {"name": "ir_right", "stream_type": "infrared", "index": 2, "format": "y8"},
    ],
}


def stable_order_serials(serials: list[str]) -> list[str]:
    order = {serial: index for index, serial in enumerate(STABLE_SERIAL_ORDER)}
    return sorted(serials, key=lambda serial: (order.get(serial, len(order)), serial))


def stream_set_has_ir(stream_set_name: str) -> bool:
    return any(spec["stream_type"] == "infrared" for spec in STREAM_SET_DEFS[stream_set_name])


def make_case_id(case: dict[str, Any]) -> str:
    serial_suffix = "-".join(case["serials"])
    return (
        f"{case['topology_type']}-{serial_suffix}-{case['stream_set']}-"
        f"{case['width']}x{case['height']}-fps{case['fps']}-emitter-{case['emitter_request']}"
    )


def case_key(case: dict[str, Any] | tuple[Any, ...]) -> tuple[Any, ...]:
    if isinstance(case, tuple):
        return case
    return (
        case["topology_type"],
        tuple(case["serials"]),
        case["stream_set"],
        case["width"],
        case["height"],
        case["fps"],
        case["emitter_request"],
    )


def make_case(
    *,
    topology_type: str,
    serials: list[str],
    stream_set: str,
    width: int,
    height: int,
    fps: int,
    emitter_request: str,
    warmup_s: float,
    duration_s: float,
) -> dict[str, Any]:
    case = {
        "topology_type": topology_type,
        "serials": list(serials),
        "stream_set": stream_set,
        "requested_streams": copy.deepcopy(STREAM_SET_DEFS[stream_set]),
        "width": int(width),
        "height": int(height),
        "fps": int(fps),
        "emitter_request": emitter_request,
        "warmup_s": float(warmup_s),
        "duration_s": float(duration_s),
    }
    case["case_id"] = make_case_id(case)
    return case


def build_initial_probe_cases(
    serials: list[str],
    *,
    warmup_s: float = 2.0,
    duration_s: float = 10.0,
    primary_width: int = 848,
    primary_height: int = 480,
    fps: int = 30,
) -> list[dict[str, Any]]:
    ordered_serials = stable_order_serials(serials)
    topologies: list[tuple[str, list[str]]] = [("single", [serial]) for serial in ordered_serials]
    if len(ordered_serials) >= 3:
        topologies.append(("three_camera", ordered_serials[:3]))

    cases: list[dict[str, Any]] = []
    for topology_type, topology_serials in topologies:
        for stream_set in STREAM_SET_DEFS:
            emitter_request = "on" if stream_set_has_ir(stream_set) else "auto"
            cases.append(
                make_case(
                    topology_type=topology_type,
                    serials=topology_serials,
                    stream_set=stream_set,
                    width=primary_width,
                    height=primary_height,
                    fps=fps,
                    emitter_request=emitter_request,
                    warmup_s=warmup_s,
                    duration_s=duration_s,
                )
            )
    return cases


def build_followup_probe_cases(
    results: list[dict[str, Any]],
    *,
    fallback_width: int = 640,
    fallback_height: int = 480,
) -> list[dict[str, Any]]:
    existing_keys = {case_key(result) for result in results}
    followups: list[dict[str, Any]] = []
    followup_keys: set[tuple[Any, ...]] = set()

    for result in results:
        if result["success"] and stream_set_has_ir(result["stream_set"]) and result["emitter_request"] == "on":
            emitter_off_case = make_case(
                topology_type=result["topology_type"],
                serials=result["serials"],
                stream_set=result["stream_set"],
                width=result["width"],
                height=result["height"],
                fps=result["fps"],
                emitter_request="off",
                warmup_s=result["warmup_s"],
                duration_s=result["duration_s"],
            )
            key = case_key(emitter_off_case)
            if key not in existing_keys and key not in followup_keys:
                followups.append(emitter_off_case)
                followup_keys.add(key)

        if (
            not result["success"]
            and result["width"] == 848
            and result["height"] == 480
        ):
            fallback_case = make_case(
                topology_type=result["topology_type"],
                serials=result["serials"],
                stream_set=result["stream_set"],
                width=fallback_width,
                height=fallback_height,
                fps=result["fps"],
                emitter_request=result["emitter_request"],
                warmup_s=result["warmup_s"],
                duration_s=result["duration_s"],
            )
            key = case_key(fallback_case)
            if key not in existing_keys and key not in followup_keys:
                followups.append(fallback_case)
                followup_keys.add(key)

    return followups


def normalize_case_result(case: dict[str, Any], updates: dict[str, Any] | None = None) -> dict[str, Any]:
    result = {
        "schema_version": "d455_stream_probe_case_v1",
        "case_id": case["case_id"],
        "topology_type": case["topology_type"],
        "serials": list(case["serials"]),
        "stream_set": case["stream_set"],
        "requested_streams": copy.deepcopy(case["requested_streams"]),
        "width": int(case["width"]),
        "height": int(case["height"]),
        "fps": int(case["fps"]),
        "emitter_request": case["emitter_request"],
        "warmup_s": float(case["warmup_s"]),
        "duration_s": float(case["duration_s"]),
        "start_success": False,
        "success": False,
        "status": "fail",
        "time_to_first_frame_s": None,
        "all_requested_streams_delivered": False,
        "sample_frame_dir": None,
        "log_path": None,
        "error_type": None,
        "error_message": None,
        "per_camera": [],
        "notes": [],
    }
    if updates:
        result.update(updates)
    return result


def build_run_document(
    *,
    run_id: str,
    output_root: Path,
    cases: list[dict[str, Any]],
    recommendation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "d455_stream_probe_run_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "stable_serial_order": STABLE_SERIAL_ORDER,
        "expectation_sources": EXPECTATION_SOURCES,
        "stability_thresholds": STABILITY_THRESHOLDS,
        "output_root": str(output_root),
        "cases": cases,
        "recommendation": recommendation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe D455 stream capability for one-camera and three-camera topologies.")
    parser.add_argument("--serials", nargs="*", default=STABLE_SERIAL_ORDER)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--primary_width", type=int, default=848)
    parser.add_argument("--primary_height", type=int, default=480)
    parser.add_argument("--fallback_width", type=int, default=640)
    parser.add_argument("--fallback_height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup_s", type=float, default=2.0)
    parser.add_argument("--duration_s", type=float, default=10.0)
    parser.add_argument("--copy_docs", type=int, choices=(0, 1), default=1)
    return parser.parse_args()


def _stream_name_to_enum(stream_type: str):
    import pyrealsense2 as rs

    mapping = {
        "depth": rs.stream.depth,
        "color": rs.stream.color,
        "infrared": rs.stream.infrared,
    }
    return mapping[stream_type]


def _format_name_to_enum(format_name: str):
    import pyrealsense2 as rs

    mapping = {
        "z16": rs.format.z16,
        "bgr8": rs.format.bgr8,
        "y8": rs.format.y8,
    }
    return mapping[format_name]


def _device_info_dict(device) -> dict[str, Any]:
    import pyrealsense2 as rs

    fields = {
        "serial": rs.camera_info.serial_number,
        "model_name": rs.camera_info.name,
        "product_line": rs.camera_info.product_line,
        "firmware_version": rs.camera_info.firmware_version,
        "usb_type_descriptor": rs.camera_info.usb_type_descriptor,
        "physical_port": rs.camera_info.physical_port,
    }
    info: dict[str, Any] = {}
    for key, field in fields.items():
        try:
            info[key] = device.get_info(field)
        except Exception:
            info[key] = None
    return info


def _extract_frame(frameset, stream_spec: dict[str, Any]):
    if stream_spec["stream_type"] == "depth":
        return frameset.get_depth_frame()
    if stream_spec["stream_type"] == "color":
        return frameset.get_color_frame()
    if stream_spec["stream_type"] == "infrared":
        return frameset.get_infrared_frame(stream_spec["index"])
    raise ValueError(f"Unsupported stream type: {stream_spec['stream_type']}")


def _init_stream_metric(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "requested_stream": copy.deepcopy(spec),
        "actual_format": None,
        "actual_stream_index": None,
        "image_shape": None,
        "image_dtype": None,
        "frame_count": 0,
        "first_timestamp_ms": None,
        "last_timestamp_ms": None,
        "timestamp_monotonic": True,
        "max_timestamp_gap_ms": 0.0,
        "observed_fps": 0.0,
        "stalled_mid_run": False,
    }


def _update_stream_metric(metric: dict[str, Any], frame, elapsed_measure_s: float, expected_period_ms: float) -> None:
    import numpy as np

    array = np.asanyarray(frame.get_data())
    timestamp_ms = float(frame.get_timestamp())
    metric["frame_count"] += 1
    metric["image_shape"] = list(array.shape)
    metric["image_dtype"] = str(array.dtype)
    metric["actual_format"] = str(frame.get_profile().format())
    metric["actual_stream_index"] = int(frame.get_profile().stream_index())

    if metric["first_timestamp_ms"] is None:
        metric["first_timestamp_ms"] = timestamp_ms
    else:
        if timestamp_ms <= float(metric["last_timestamp_ms"]):
            metric["timestamp_monotonic"] = False
        gap_ms = timestamp_ms - float(metric["last_timestamp_ms"])
        metric["max_timestamp_gap_ms"] = max(float(metric["max_timestamp_gap_ms"]), gap_ms)
    metric["last_timestamp_ms"] = timestamp_ms

    if metric["frame_count"] > 1 and metric["first_timestamp_ms"] is not None and metric["last_timestamp_ms"] is not None:
        duration_ms = float(metric["last_timestamp_ms"]) - float(metric["first_timestamp_ms"])
        if duration_ms > 0:
            metric["observed_fps"] = (metric["frame_count"] - 1) / (duration_ms / 1000.0)
    elif elapsed_measure_s > 0:
        metric["observed_fps"] = metric["frame_count"] / elapsed_measure_s

    metric["stalled_mid_run"] = bool(metric["max_timestamp_gap_ms"] > (expected_period_ms * STABILITY_THRESHOLDS["max_timestamp_gap_factor"]))


def _save_sample_frame(frame, stream_name: str, output_dir: Path) -> None:
    import cv2
    import numpy as np
    from data_process.visualization.depth_colormap import colorize_depth_units

    output_dir.mkdir(parents=True, exist_ok=True)
    array = np.asanyarray(frame.get_data())
    if stream_name == "depth":
        np.save(output_dir / "depth.npy", array)
        try:
            depth_scale_m_per_unit = float(frame.get_units())
        except Exception:
            depth_scale_m_per_unit = 0.001
        depth_vis = colorize_depth_units(
            array,
            depth_scale_m_per_unit=depth_scale_m_per_unit,
        )
        cv2.imwrite(str(output_dir / "depth.png"), depth_vis)
    else:
        cv2.imwrite(str(output_dir / f"{stream_name}.png"), array)


def _write_contact_sheet(case_dir: Path) -> None:
    import cv2
    import numpy as np

    tiles: list[np.ndarray] = []
    for camera_dir in sorted(case_dir.glob("cam*_*/")):
        for path in sorted(camera_dir.iterdir()):
            if path.suffix.lower() != ".png":
                continue
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            label = f"{camera_dir.name}:{path.stem}"
            cv2.putText(image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(image)

    if not tiles:
        return

    max_height = max(tile.shape[0] for tile in tiles)
    resized = []
    for tile in tiles:
        if tile.shape[0] != max_height:
            scale = max_height / tile.shape[0]
            tile = cv2.resize(tile, (int(tile.shape[1] * scale), max_height), interpolation=cv2.INTER_NEAREST)
        resized.append(tile)
    contact_sheet = np.hstack(resized)
    cv2.imwrite(str(case_dir / "contact_sheet.png"), contact_sheet)


def _write_case_log(case: dict[str, Any], result: dict[str, Any], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({"case": case, "result": result}, indent=2), encoding="utf-8")


def execute_case(case: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    import pyrealsense2 as rs

    result = normalize_case_result(case)
    ctx = rs.context()
    devices = {dev.get_info(rs.camera_info.serial_number): dev for dev in ctx.query_devices()}
    missing_serials = [serial for serial in case["serials"] if serial not in devices]
    if missing_serials:
        result["error_type"] = "MissingDevice"
        result["error_message"] = f"Missing serial(s): {missing_serials}"
        result["log_path"] = str(run_dir / "logs" / f"{case['case_id']}.json")
        _write_case_log(case, result, Path(result["log_path"]))
        return result

    sample_case_dir = run_dir / "sample_frames" / case["case_id"]
    log_path = run_dir / "logs" / f"{case['case_id']}.json"
    result["log_path"] = str(log_path)

    camera_results: list[dict[str, Any]] = []
    active_pipelines: list[tuple[str, Any]] = []
    expected_period_ms = 1000.0 / case["fps"]
    case_start_wall = time.perf_counter()
    try:
        for camera_index, serial in enumerate(case["serials"]):
            device = devices[serial]
            pipeline = rs.pipeline(ctx)
            config = rs.config()
            config.enable_device(serial)
            for spec in case["requested_streams"]:
                stream = _stream_name_to_enum(spec["stream_type"])
                fmt = _format_name_to_enum(spec["format"])
                if spec["stream_type"] == "infrared":
                    config.enable_stream(stream, spec["index"], case["width"], case["height"], fmt, case["fps"])
                else:
                    config.enable_stream(stream, case["width"], case["height"], fmt, case["fps"])

            profile = pipeline.start(config)
            depth_sensor = profile.get_device().first_depth_sensor()
            if case["emitter_request"] in {"on", "off"} and depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if case["emitter_request"] == "on" else 0.0)
            emitter_actual = None
            if depth_sensor.supports(rs.option.emitter_enabled):
                emitter_actual = float(depth_sensor.get_option(rs.option.emitter_enabled))

            stream_metrics = {spec["name"]: _init_stream_metric(spec) for spec in case["requested_streams"]}
            active_profile_map: dict[str, Any] = {}
            for spec in case["requested_streams"]:
                stream = _stream_name_to_enum(spec["stream_type"])
                if spec["stream_type"] == "infrared":
                    active_profile = profile.get_stream(stream, spec["index"]).as_video_stream_profile()
                else:
                    active_profile = profile.get_stream(stream).as_video_stream_profile()
                active_profile_map[spec["name"]] = {
                    "stream_type": spec["stream_type"],
                    "stream_index": int(active_profile.stream_index()),
                    "format": str(active_profile.format()),
                    "width": int(active_profile.width()),
                    "height": int(active_profile.height()),
                    "fps": int(active_profile.fps()),
                }

            camera_results.append(
                {
                    "camera_index": camera_index,
                    "serial": serial,
                    "device_info": _device_info_dict(device),
                    "emitter_actual": emitter_actual,
                    "time_to_first_frame_s": None,
                    "timeouts": 0,
                    "timeouts_after_first_frame": 0,
                    "complete_framesets": 0,
                    "all_requested_streams_delivered": False,
                    "streams": stream_metrics,
                    "active_profiles": active_profile_map,
                }
            )
            active_pipelines.append((serial, pipeline))
        result["start_success"] = True

        warmup_deadline = case_start_wall + case["warmup_s"]
        end_deadline = warmup_deadline + case["duration_s"]
        camera_by_serial = {camera_result["serial"]: camera_result for camera_result in camera_results}
        saved_samples: set[tuple[str, str]] = set()

        while time.perf_counter() < end_deadline:
            for serial, pipeline in active_pipelines:
                now = time.perf_counter()
                if now >= end_deadline:
                    break
                timeout_ms = max(250, int(min(1000, (end_deadline - now) * 1000)))
                camera_result = camera_by_serial[serial]
                try:
                    frameset = pipeline.wait_for_frames(timeout_ms)
                except RuntimeError as exc:
                    camera_result["timeouts"] += 1
                    if camera_result["time_to_first_frame_s"] is not None:
                        camera_result["timeouts_after_first_frame"] += 1
                    result["notes"].append(f"{serial}: timeout during measurement: {exc}")
                    continue

                present_stream_names: list[str] = []
                for spec in case["requested_streams"]:
                    frame = _extract_frame(frameset, spec)
                    if not frame:
                        continue
                    present_stream_names.append(spec["name"])

                    if camera_result["time_to_first_frame_s"] is None:
                        camera_result["time_to_first_frame_s"] = time.perf_counter() - case_start_wall

                    sample_key = (serial, spec["name"])
                    if sample_key not in saved_samples:
                        sample_dir = sample_case_dir / f"cam{camera_result['camera_index']}_{serial}"
                        _save_sample_frame(frame, spec["name"], sample_dir)
                        saved_samples.add(sample_key)

                    if time.perf_counter() < warmup_deadline:
                        continue

                    elapsed_measure_s = max(1e-6, time.perf_counter() - warmup_deadline)
                    _update_stream_metric(
                        camera_result["streams"][spec["name"]],
                        frame,
                        elapsed_measure_s=elapsed_measure_s,
                        expected_period_ms=expected_period_ms,
                    )

                if set(present_stream_names) == {spec["name"] for spec in case["requested_streams"]}:
                    camera_result["complete_framesets"] += 1

        if sample_case_dir.exists():
            _write_contact_sheet(sample_case_dir)
            result["sample_frame_dir"] = str(sample_case_dir)

        case_first_frames = [
            camera_result["time_to_first_frame_s"]
            for camera_result in camera_results
            if camera_result["time_to_first_frame_s"] is not None
        ]
        if case_first_frames:
            result["time_to_first_frame_s"] = max(case_first_frames)

        all_streams_ok = True
        for camera_result in camera_results:
            camera_result["all_requested_streams_delivered"] = all(
                stream_metric["frame_count"] > 0 for stream_metric in camera_result["streams"].values()
            )
            for stream_metric in camera_result["streams"].values():
                observed_fps = float(stream_metric["observed_fps"])
                fps_ok = observed_fps >= (case["fps"] * STABILITY_THRESHOLDS["min_fps_ratio"])
                monotonic_ok = bool(stream_metric["timestamp_monotonic"])
                no_stall = not bool(stream_metric["stalled_mid_run"])
                stream_metric["stable"] = bool(
                    stream_metric["frame_count"] > 0 and fps_ok and monotonic_ok and no_stall
                )
                if not stream_metric["stable"]:
                    all_streams_ok = False
            if not camera_result["all_requested_streams_delivered"] or camera_result["timeouts_after_first_frame"] > 0:
                all_streams_ok = False

        result["per_camera"] = camera_results
        result["all_requested_streams_delivered"] = all(
            camera_result["all_requested_streams_delivered"] for camera_result in camera_results
        )
        result["success"] = bool(result["start_success"] and all_streams_ok and result["all_requested_streams_delivered"])
        result["status"] = "pass" if result["success"] else "fail"
        if not result["success"] and result["error_message"] is None:
            result["error_type"] = "StabilityThresholdNotMet"
            result["error_message"] = "Case started but failed delivery / fps / stall thresholds."
        return result
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        result["status"] = "fail"
        result["success"] = False
        return result
    finally:
        for _, pipeline in active_pipelines:
            try:
                pipeline.stop()
            except Exception:
                pass
        _write_case_log(case, result, log_path)
        time.sleep(0.2)


def _best_result(
    results: list[dict[str, Any]],
    *,
    topology_type: str,
    stream_set: str,
    width: int | None = None,
) -> dict[str, Any] | None:
    candidates = [
        result
        for result in results
        if result["topology_type"] == topology_type and result["stream_set"] == stream_set and (width is None or result["width"] == width)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda result: (not result["success"], result["width"], result["emitter_request"]))
    return candidates[0]


def compute_recommendation(results: list[dict[str, Any]]) -> dict[str, Any]:
    triple_rgb_ir_848 = _best_result(results, topology_type="three_camera", stream_set="rgb_ir_pair", width=848)
    triple_rgb_ir_640 = _best_result(results, topology_type="three_camera", stream_set="rgb_ir_pair", width=640)
    triple_ir_pair = _best_result(results, topology_type="three_camera", stream_set="ir_pair")
    triple_depth_compare = [
        result
        for result in results
        if result["topology_type"] == "three_camera"
        and result["stream_set"] in {"depth_ir_pair", "rgbd_ir_pair"}
        and result["success"]
    ]
    single_ir_support = [
        result
        for result in results
        if result["topology_type"] == "single"
        and result["stream_set"] in {"ir_pair", "rgb_ir_pair"}
        and result["success"]
    ]

    if triple_rgb_ir_848 and triple_rgb_ir_848["success"]:
        primary_case = "A"
        statement = "rgb_ir_pair is stable on all 3 cameras at 848x480@30."
        evidence = [triple_rgb_ir_848["case_id"]]
    elif triple_rgb_ir_640 and triple_rgb_ir_640["success"]:
        primary_case = "B"
        statement = "rgb_ir_pair is only stable on all 3 cameras at 640x480@30."
        evidence = [triple_rgb_ir_640["case_id"]]
    elif triple_ir_pair and triple_ir_pair["success"]:
        primary_case = "C"
        statement = "ir_pair is stable on all 3 cameras, but rgb_ir_pair is not yet stable as a same-take mode."
        evidence = [triple_ir_pair["case_id"]]
    elif single_ir_support:
        primary_case = "F"
        statement = "Only single-camera IR-related support is stable; keep FFS work at probe/offline stage."
        evidence = [result["case_id"] for result in single_ir_support]
    else:
        primary_case = "F"
        statement = "No grounded multi-camera IR mode is stable enough yet; keep record_data.py untouched."
        evidence = []

    comparison_case = "D" if triple_depth_compare else "E"
    comparison_statement = (
        "Same-take hardware-depth vs IR-stereo comparison modes appear feasible."
        if triple_depth_compare
        else "Do not promise same-take depth-vs-FFS comparison yet; depth_ir_pair / rgbd_ir_pair remain unstable or unsupported."
    )

    return {
        "primary_case": primary_case,
        "primary_statement": statement,
        "comparison_case": comparison_case,
        "comparison_statement": comparison_statement,
        "evidence_case_ids": evidence,
    }


def main() -> int:
    args = parse_args()
    ordered_serials = stable_order_serials(list(args.serials))
    output_root = Path(args.output_root).resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / "runs" / run_id
    latest_dir = output_root / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)

    pending_cases = build_initial_probe_cases(
        ordered_serials,
        warmup_s=args.warmup_s,
        duration_s=args.duration_s,
        primary_width=args.primary_width,
        primary_height=args.primary_height,
        fps=args.fps,
    )
    completed_results: list[dict[str, Any]] = []
    queued_keys = {case_key(case) for case in pending_cases}

    while pending_cases:
        case = pending_cases.pop(0)
        queued_keys.discard(case_key(case))
        print(
            f"[probe] {case['case_id']} "
            f"streams={case['stream_set']} serials={','.join(case['serials'])} "
            f"{case['width']}x{case['height']}@{case['fps']} emitter={case['emitter_request']}",
            flush=True,
        )
        result = execute_case(case, run_dir)
        completed_results.append(result)

        for followup_case in build_followup_probe_cases(
            completed_results,
            fallback_width=args.fallback_width,
            fallback_height=args.fallback_height,
        ):
            key = case_key(followup_case)
            if key not in queued_keys and all(case_key(existing) != key for existing in pending_cases):
                pending_cases.append(followup_case)
                queued_keys.add(key)

    recommendation = compute_recommendation(completed_results)
    run_document = build_run_document(
        run_id=run_id,
        output_root=output_root,
        cases=completed_results,
        recommendation=recommendation,
    )
    results_path = run_dir / "results.json"
    results_path.write_text(json.dumps(run_document, indent=2), encoding="utf-8")

    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)

    from scripts.harness.render_d455_stream_probe_report import render_probe_report

    render_probe_report(
        results_path=results_path,
        summary_path=run_dir / "summary.md",
        doc_json_path=DEFAULT_DOC_JSON if args.copy_docs else None,
        doc_md_path=DEFAULT_DOC_MD if args.copy_docs else None,
    )
    if args.copy_docs:
        shutil.copy2(run_dir / "summary.md", latest_dir / "summary.md")
        shutil.copy2(results_path, latest_dir / "results.json")

    print(f"[probe] wrote results to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
