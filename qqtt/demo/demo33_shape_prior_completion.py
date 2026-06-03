from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

from qqtt.demo.demo33_shape_prior_warmup import (
    DEFAULT_FUTUREPHYSTWIN_PYTHON,
    DEFAULT_FUTUREPHYSTWIN_ROOT,
    DEFAULT_SAM3D_ROOT,
    DEFAULT_SHAPE_PRIOR_CAMERA_IDX,
    DEFAULT_SHAPE_PRIOR_GROUND_POLICY,
    DEFAULT_SHAPE_PRIOR_GROUND_Z,
    ShapePriorWarmupConfig,
    load_shape_prior_final_data,
    run_futurephystwin_single_view_route,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        return True
    return True


def wait_for_pid_exit(pid: int, *, timeout_s: float = 900.0, poll_s: float = 0.5) -> dict[str, Any]:
    start_s = time.perf_counter()
    if pid <= 0:
        return {"waited_for_pid": 0, "pid_exit_observed": True, "pid_wait_elapsed_ms": 0.0}
    while _pid_alive(pid):
        if time.perf_counter() - start_s > timeout_s:
            return {
                "waited_for_pid": int(pid),
                "pid_exit_observed": False,
                "pid_wait_elapsed_ms": float((time.perf_counter() - start_s) * 1000.0),
            }
        time.sleep(max(0.01, float(poll_s)))
    return {
        "waited_for_pid": int(pid),
        "pid_exit_observed": True,
        "pid_wait_elapsed_ms": float((time.perf_counter() - start_s) * 1000.0),
    }


def _config_from_args(args: argparse.Namespace) -> ShapePriorWarmupConfig:
    case_dir = Path(args.case_dir).expanduser().resolve()
    run_id = case_dir.parent.name
    output_root = case_dir.parents[2]
    return ShapePriorWarmupConfig(
        enabled=True,
        output_root=output_root,
        run_id=run_id,
        futurephystwin_root=Path(args.futurephystwin_root).expanduser().resolve(),
        futurephystwin_python=str(args.futurephystwin_python),
        sam3d_root=Path(args.sam3d_root).expanduser().resolve(),
        camera_idx=int(args.shape_prior_camera_idx),
        force=bool(args.force),
        object_label=str(args.object_label),
        controller_label=str(args.controller_label),
        ground_policy=str(args.ground_policy),
        ground_z=float(args.ground_z),
        cuda_visible_devices=str(args.cuda_visible_devices or ""),
        cuda_allocator_config=str(args.cuda_alloc_conf or ""),
        skip_route_visualizations=bool(args.skip_route_visualizations),
    )


def _completion_base_profile(args: argparse.Namespace, wait_profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "shape_prior_warmup_enabled": True,
        "shape_prior_status": "running",
        "shape_prior_start_policy": "after-teardown",
        "shape_prior_start_trigger": "after_teardown_detached",
        "shape_prior_case_dir": str(Path(args.case_dir).expanduser()),
        "shape_prior_gpu": str(args.cuda_visible_devices or ""),
        "shape_prior_cuda_visible_devices": str(args.cuda_visible_devices or ""),
        "shape_prior_cuda_alloc_conf": str(args.cuda_alloc_conf or ""),
        "shape_prior_skip_route_visualizations": bool(args.skip_route_visualizations),
        "shape_prior_async_started": True,
        "shape_prior_async_completed": False,
        "shape_prior_thread_alive": False,
        "shape_prior_detached_completion_started": True,
        "shape_prior_detached_completion_pid": int(os.getpid()),
        "shape_prior_detached_completion_json": str(Path(args.completion_json).expanduser()),
        "shape_prior_detached_completion_log": str(Path(args.completion_json).expanduser().with_suffix(".log")),
        "shape_prior_detached_completion_wait_for_pid": int(args.wait_for_pid or 0),
        "shape_prior_snapshot_ready": False,
        "shape_prior_blocks_tracker_input": False,
        "shape_prior_blocks_first_render": False,
        "shape_prior_affects_tracker_input": False,
        "shape_prior_affects_live_observation_pcd": False,
        **wait_profile,
    }


def merge_completion_into_live_profile(
    *,
    profile_json: Path,
    completion_json: Path,
    completion_profile: dict[str, Any],
) -> dict[str, Any]:
    def _merge_payload(payload: dict[str, Any]) -> dict[str, Any]:
        shape_profile = dict(completion_profile)
        payload["shape_prior_warmup"] = shape_profile
        payload["shape_prior_completion_json"] = str(completion_json)
        payload["shape_prior_completion_updated_at_s"] = float(time.time())
        for key, value in shape_profile.items():
            if str(key).startswith("shape_prior_"):
                payload[key] = value

        summary = payload.get("summary")
        if isinstance(summary, dict):
            for key, value in shape_profile.items():
                if str(key).startswith("shape_prior_"):
                    summary[key] = value

        for snapshot_key in ("cotracker_process_snapshot", "tracker_process_snapshot"):
            snapshot = payload.get(snapshot_key)
            if isinstance(snapshot, dict):
                snapshot["shape_prior_warmup"] = shape_profile
                snapshot.update(shape_profile)
        return payload

    payload = _load_json(profile_json)
    if not payload:
        return {}
    payload = _merge_payload(payload)
    shared_profile_raw = payload.get("shared_runtime_profile")
    if isinstance(shared_profile_raw, str) and shared_profile_raw:
        shared_profile = Path(shared_profile_raw).expanduser()
        if not shared_profile.is_absolute():
            shared_profile = profile_json.parent.parent / shared_profile if shared_profile.parts[:1] == ("..",) else shared_profile
            if not shared_profile.is_absolute():
                shared_profile = Path.cwd() / shared_profile
        shared_payload = _load_json(shared_profile)
        if shared_payload:
            _write_json(shared_profile, _merge_payload(shared_payload))

    _write_json(profile_json, payload)
    return payload


def complete_shape_prior_case(args: argparse.Namespace) -> dict[str, Any]:
    completion_json = Path(args.completion_json).expanduser()
    profile_json = Path(args.profile_json).expanduser() if args.profile_json else None
    wait_profile = wait_for_pid_exit(
        int(args.wait_for_pid or 0),
        timeout_s=float(args.wait_timeout_s),
        poll_s=float(args.wait_poll_s),
    )
    start_s = time.perf_counter()
    profile = _completion_base_profile(args, wait_profile)
    _write_json(completion_json, {"summary": dict(profile), "shape_prior_warmup": dict(profile)})
    if not bool(wait_profile.get("pid_exit_observed", True)):
        elapsed_ms = float((time.perf_counter() - start_s) * 1000.0)
        profile.update(
            {
                "shape_prior_status": "error",
                "shape_prior_error": "Timed out waiting for live Demo 3.3 process to exit",
                "shape_prior_last_error": "Timed out waiting for live Demo 3.3 process to exit",
                "shape_prior_async_completed": True,
                "shape_prior_async_elapsed_ms": elapsed_ms,
                "shape_prior_render_layer_enabled": False,
            }
        )
        _write_json(completion_json, {"summary": dict(profile), "shape_prior_warmup": dict(profile)})
        if profile_json is not None:
            merge_completion_into_live_profile(
                profile_json=profile_json,
                completion_json=completion_json,
                completion_profile=profile,
            )
        return profile

    config = _config_from_args(args)
    try:
        command_records = run_futurephystwin_single_view_route(config=config, runner=subprocess.run)
        result = load_shape_prior_final_data(config.case_dir)
    except BaseException as exc:
        elapsed_ms = float((time.perf_counter() - start_s) * 1000.0)
        profile.update(
            {
                "shape_prior_status": "error",
                "shape_prior_error": f"{type(exc).__name__}: {exc}",
                "shape_prior_last_error": f"{type(exc).__name__}: {exc}",
                "shape_prior_async_completed": True,
                "shape_prior_async_elapsed_ms": elapsed_ms,
                "shape_prior_render_layer_enabled": False,
            }
        )
    else:
        elapsed_ms = float((time.perf_counter() - start_s) * 1000.0)
        profile.update(result.profile)
        profile.update(
            {
                "shape_prior_warmup_enabled": True,
                "shape_prior_command_records": command_records,
                "shape_prior_command_order": [record["stage"] for record in command_records],
                "shape_prior_start_policy": "after-teardown",
                "shape_prior_start_trigger": "after_teardown_detached",
                "shape_prior_gpu": str(args.cuda_visible_devices or ""),
                "shape_prior_cuda_visible_devices": str(args.cuda_visible_devices or ""),
                "shape_prior_cuda_alloc_conf": str(args.cuda_alloc_conf or ""),
                "shape_prior_skip_route_visualizations": bool(args.skip_route_visualizations),
                "shape_prior_async_started": True,
                "shape_prior_async_completed": True,
                "shape_prior_thread_alive": False,
                "shape_prior_detached_completion_started": True,
                "shape_prior_detached_completion_pid": int(os.getpid()),
                "shape_prior_detached_completion_json": str(completion_json),
                "shape_prior_detached_completion_log": str(completion_json.with_suffix(".log")),
                "shape_prior_detached_completion_wait_for_pid": int(args.wait_for_pid or 0),
                "shape_prior_async_elapsed_ms": elapsed_ms,
                "shape_prior_snapshot_ready": False,
                "shape_prior_blocks_tracker_input": False,
                "shape_prior_blocks_first_render": False,
                **wait_profile,
            }
        )

    payload = {"summary": dict(profile), "shape_prior_warmup": dict(profile)}
    _write_json(completion_json, payload)
    if profile_json is not None:
        merge_completion_into_live_profile(
            profile_json=profile_json,
            completion_json=completion_json,
            completion_profile=profile,
        )
    return profile


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Complete a Demo 3.3 shape-prior case after live teardown.")
    parser.add_argument("--profile-json", type=Path, default=None)
    parser.add_argument("--completion-json", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--futurephystwin-root", type=Path, default=DEFAULT_FUTUREPHYSTWIN_ROOT)
    parser.add_argument("--futurephystwin-python", default=DEFAULT_FUTUREPHYSTWIN_PYTHON)
    parser.add_argument("--sam3d-root", type=Path, default=DEFAULT_SAM3D_ROOT)
    parser.add_argument("--shape-prior-camera-idx", type=int, default=DEFAULT_SHAPE_PRIOR_CAMERA_IDX)
    parser.add_argument("--object-label", default="stuffed animal")
    parser.add_argument("--controller-label", default="towel")
    parser.add_argument("--ground-policy", default=DEFAULT_SHAPE_PRIOR_GROUND_POLICY)
    parser.add_argument("--ground-z", type=float, default=DEFAULT_SHAPE_PRIOR_GROUND_Z)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--cuda-alloc-conf", default="")
    parser.add_argument(
        "--skip-route-visualizations",
        dest="skip_route_visualizations",
        action="store_true",
        default=False,
        help="Skip optional FuturePhysTwin route visualization videos/plots.",
    )
    parser.add_argument("--wait-for-pid", type=int, default=0)
    parser.add_argument("--wait-timeout-s", type=float, default=900.0)
    parser.add_argument("--wait-poll-s", type=float, default=0.5)
    parser.add_argument("--force", action="store_true", default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    profile = complete_shape_prior_case(args)
    print(json.dumps(profile, indent=2, sort_keys=True))
    return 0 if str(profile.get("shape_prior_status", "")) in {"ready", "invalid_coordinate_policy"} else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
