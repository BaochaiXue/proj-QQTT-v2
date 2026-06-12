from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Sequence

import numpy as np

from qqtt.demo import demo31_runtime as demo31
from qqtt.demo import demo32_runtime as demo32
from qqtt.demo import demo3_runtime
from qqtt.demo.demo33_shape_prior_warmup import (
    DEFAULT_FUTUREPHYSTWIN_PYTHON,
    DEFAULT_FUTUREPHYSTWIN_ROOT,
    DEFAULT_SAM3D_ROOT,
    DEFAULT_SHAPE_PRIOR_CAMERA_IDX,
    DEFAULT_SHAPE_PRIOR_COORDINATE_FRAME,
    DEFAULT_SHAPE_PRIOR_GROUND_POLICY,
    DEFAULT_SHAPE_PRIOR_GROUND_Z,
    DEFAULT_SHAPE_PRIOR_UNITS,
    ShapePriorWarmupConfig,
    SubprocessRunner,
    load_shape_prior_final_data,
    run_futurephystwin_single_view_route,
    run_shape_prior_warmup,
    validate_original_sam3d_root,
    write_futurephystwin_warmup_case,
)


PRESET_DEMO32_FFS_TAPNEXTPP = demo32.PRESET_DEMO32_FFS_TAPNEXTPP
PRESET_DEMO32_FFS_LITETRACKER = demo32.PRESET_DEMO32_FFS_LITETRACKER
DEMO33_RUNTIME_MODULE = "qqtt.demo.demo33_runtime"
DEMO33_RUNTIME_OWNER = "demo33_shape_prior_warmup"
DEMO33_SHAPE_PRIOR_ROUTE = (
    "image_upscale.py",
    "segment_util_image.py",
    "data_process_sam3d/shape_prior.py",
    "data_process/align.py",
    "data_process_sam3d/data_process_sample.py",
)
SHAPE_PRIOR_START_POLICY_IMMEDIATE = "immediate"
SHAPE_PRIOR_START_POLICY_AFTER_FIRST_RENDER = "after-first-render"
SHAPE_PRIOR_START_POLICY_AFTER_TEARDOWN = "after-teardown"
SHAPE_PRIOR_START_POLICIES = (
    SHAPE_PRIOR_START_POLICY_AFTER_TEARDOWN,
    SHAPE_PRIOR_START_POLICY_AFTER_FIRST_RENDER,
    SHAPE_PRIOR_START_POLICY_IMMEDIATE,
)
DEFAULT_SHAPE_PRIOR_WARMUP = True


def _resolve_shape_prior_gpu(args: argparse.Namespace) -> str:
    requested = str(getattr(args, "shape_prior_gpu", "auto") or "auto").strip()
    if requested and requested.lower() != "auto":
        return requested
    mask_gpu = str(getattr(args, "mask_gpu", "0"))
    _ = getattr(args, "cotracker_gpu", mask_gpu)
    return mask_gpu


def _shape_prior_contract_fields(args: argparse.Namespace) -> dict[str, Any]:
    enabled = bool(getattr(args, "shape_prior_warmup", DEFAULT_SHAPE_PRIOR_WARMUP))
    case_template = Path(getattr(args, "output_root", "result/demo32_ffs_tapnextpp"))
    case_template = case_template / "demo33_shape_prior_warmup" / "<run_id>" / "case"
    start_policy = str(
        getattr(args, "shape_prior_start_policy", SHAPE_PRIOR_START_POLICY_AFTER_TEARDOWN)
        or SHAPE_PRIOR_START_POLICY_AFTER_TEARDOWN
    )
    shape_prior_gpu = _resolve_shape_prior_gpu(args)
    cuda_alloc_conf = str(
        getattr(args, "shape_prior_cuda_alloc_conf", "expandable_segments:True")
        or ""
    )
    return {
        "shape_prior_warmup_enabled": enabled,
        "shape_prior_status": "pending" if enabled else "disabled",
        "shape_prior_route": list(DEMO33_SHAPE_PRIOR_ROUTE),
        "shape_prior_case_dir": str(case_template),
        "shape_prior_camera_idx": int(getattr(args, "shape_prior_camera_idx", DEFAULT_SHAPE_PRIOR_CAMERA_IDX)),
        "futurephystwin_root": str(getattr(args, "futurephystwin_root", DEFAULT_FUTUREPHYSTWIN_ROOT)),
        "futurephystwin_python": str(getattr(args, "futurephystwin_python", DEFAULT_FUTUREPHYSTWIN_PYTHON)),
        "sam3d_root": str(getattr(args, "sam3d_root", DEFAULT_SAM3D_ROOT)),
        "shape_prior_force": bool(getattr(args, "shape_prior_force", False)),
        "shape_prior_source_group_id": -1,
        "shape_prior_coordinate_frame": DEFAULT_SHAPE_PRIOR_COORDINATE_FRAME,
        "shape_prior_units": DEFAULT_SHAPE_PRIOR_UNITS,
        "shape_prior_ground_policy": DEFAULT_SHAPE_PRIOR_GROUND_POLICY,
        "shape_prior_ground_z": DEFAULT_SHAPE_PRIOR_GROUND_Z,
        "shape_prior_coordinate_validation_status": "pending" if enabled else "disabled",
        "shape_prior_coordinate_validation_reason": "",
        "shape_prior_object_points0": 0,
        "shape_prior_surface_points": 0,
        "shape_prior_interior_points": 0,
        "shape_prior_structure_points": 0,
        "shape_prior_raw_structure_points": 0,
        "shape_prior_render_layer_enabled": False,
        "shape_prior_render_layer": "gray_canonical_reference",
        "shape_prior_execution_mode": "async_background_thread" if enabled else "disabled",
        "shape_prior_start_policy": start_policy if enabled else "disabled",
        "shape_prior_gpu": shape_prior_gpu,
        "shape_prior_cuda_visible_devices": shape_prior_gpu,
        "shape_prior_cuda_alloc_conf": cuda_alloc_conf,
        "shape_prior_retry_after_teardown": bool(getattr(args, "shape_prior_retry_after_teardown", True)),
        "shape_prior_skip_route_visualizations": bool(
            getattr(args, "shape_prior_skip_route_visualizations", True)
        ),
        "shape_prior_retry_count": 0,
        "shape_prior_snapshot_ready": False,
        "shape_prior_snapshot_group_id": -1,
        "shape_prior_start_trigger": "",
        "shape_prior_blocks_tracker_input": False,
        "shape_prior_blocks_first_render": False,
        "shape_prior_async_started": False,
        "shape_prior_async_completed": False,
        "shape_prior_thread_alive": False,
        "shape_prior_detached_completion_started": False,
        "shape_prior_detached_completion_pid": 0,
        "shape_prior_detached_completion_json": "",
        "shape_prior_detached_completion_log": "",
        "shape_prior_detached_completion_wait_for_pid": 0,
        "shape_prior_async_elapsed_ms": 0.0,
        "shape_prior_affects_tracker_input": False,
        "shape_prior_affects_live_observation_pcd": False,
    }


def _shape_prior_profile_fields(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "shape_prior_warmup_enabled": bool(contract.get("shape_prior_warmup_enabled", False)),
        "shape_prior_status": str(contract.get("shape_prior_status", "disabled")),
        "shape_prior_case_dir": str(contract.get("shape_prior_case_dir", "")),
        "shape_prior_source_group_id": int(contract.get("shape_prior_source_group_id", -1) or -1),
        "shape_prior_coordinate_frame": str(
            contract.get("shape_prior_coordinate_frame", DEFAULT_SHAPE_PRIOR_COORDINATE_FRAME)
        ),
        "shape_prior_units": str(contract.get("shape_prior_units", DEFAULT_SHAPE_PRIOR_UNITS)),
        "shape_prior_ground_policy": str(contract.get("shape_prior_ground_policy", DEFAULT_SHAPE_PRIOR_GROUND_POLICY)),
        "shape_prior_ground_z": float(contract.get("shape_prior_ground_z", DEFAULT_SHAPE_PRIOR_GROUND_Z) or 0.0),
        "shape_prior_coordinate_validation_status": str(
            contract.get("shape_prior_coordinate_validation_status", "disabled")
        ),
        "shape_prior_coordinate_validation_reason": str(
            contract.get("shape_prior_coordinate_validation_reason", "")
        ),
        "shape_prior_last_error": str(contract.get("shape_prior_last_error", "")),
        "shape_prior_object_points0": int(contract.get("shape_prior_object_points0", 0) or 0),
        "shape_prior_surface_points": int(contract.get("shape_prior_surface_points", 0) or 0),
        "shape_prior_interior_points": int(contract.get("shape_prior_interior_points", 0) or 0),
        "shape_prior_structure_points": int(contract.get("shape_prior_structure_points", 0) or 0),
        "shape_prior_raw_structure_points": int(contract.get("shape_prior_raw_structure_points", 0) or 0),
        "shape_prior_render_layer_enabled": bool(contract.get("shape_prior_render_layer_enabled", False)),
        "shape_prior_execution_mode": str(contract.get("shape_prior_execution_mode", "disabled")),
        "shape_prior_start_policy": str(contract.get("shape_prior_start_policy", "disabled")),
        "shape_prior_gpu": str(contract.get("shape_prior_gpu", "")),
        "shape_prior_cuda_visible_devices": str(contract.get("shape_prior_cuda_visible_devices", "")),
        "shape_prior_cuda_alloc_conf": str(contract.get("shape_prior_cuda_alloc_conf", "")),
        "shape_prior_retry_after_teardown": bool(contract.get("shape_prior_retry_after_teardown", False)),
        "shape_prior_skip_route_visualizations": bool(
            contract.get("shape_prior_skip_route_visualizations", False)
        ),
        "shape_prior_retry_count": int(contract.get("shape_prior_retry_count", 0) or 0),
        "shape_prior_snapshot_ready": bool(contract.get("shape_prior_snapshot_ready", False)),
        "shape_prior_snapshot_group_id": int(contract.get("shape_prior_snapshot_group_id", -1) or -1),
        "shape_prior_start_trigger": str(contract.get("shape_prior_start_trigger", "")),
        "shape_prior_blocks_tracker_input": bool(contract.get("shape_prior_blocks_tracker_input", False)),
        "shape_prior_blocks_first_render": bool(contract.get("shape_prior_blocks_first_render", False)),
        "shape_prior_async_started": bool(contract.get("shape_prior_async_started", False)),
        "shape_prior_async_completed": bool(contract.get("shape_prior_async_completed", False)),
        "shape_prior_thread_alive": bool(contract.get("shape_prior_thread_alive", False)),
        "shape_prior_detached_completion_started": bool(
            contract.get("shape_prior_detached_completion_started", False)
        ),
        "shape_prior_detached_completion_pid": int(contract.get("shape_prior_detached_completion_pid", 0) or 0),
        "shape_prior_detached_completion_json": str(contract.get("shape_prior_detached_completion_json", "")),
        "shape_prior_detached_completion_log": str(contract.get("shape_prior_detached_completion_log", "")),
        "shape_prior_detached_completion_wait_for_pid": int(
            contract.get("shape_prior_detached_completion_wait_for_pid", 0) or 0
        ),
        "shape_prior_async_elapsed_ms": float(contract.get("shape_prior_async_elapsed_ms", 0.0) or 0.0),
        "shape_prior_affects_tracker_input": False,
        "shape_prior_affects_live_observation_pcd": False,
    }


def _merge_shape_prior_profile_into_payload(
    payload: dict[str, Any],
    shape_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    profile = dict(shape_profile or {})
    payload["shape_prior_warmup"] = profile
    payload.update(_shape_prior_profile_fields(profile))
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = demo32.build_arg_parser()
    group = parser.add_argument_group("Demo 3.3 shape-prior warmup")
    group.add_argument(
        "--shape-prior-warmup",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SHAPE_PRIOR_WARMUP,
        help="Run the Demo 3.3 FuturePhysTwin/SAM3D shape-prior warmup once from a first-frame snapshot.",
    )
    group.add_argument("--futurephystwin-root", type=Path, default=DEFAULT_FUTUREPHYSTWIN_ROOT)
    group.add_argument("--futurephystwin-python", default=DEFAULT_FUTUREPHYSTWIN_PYTHON)
    group.add_argument("--sam3d-root", type=Path, default=DEFAULT_SAM3D_ROOT)
    group.add_argument("--shape-prior-camera-idx", type=int, default=DEFAULT_SHAPE_PRIOR_CAMERA_IDX)
    group.add_argument("--shape-prior-force", action="store_true", default=False)
    group.add_argument(
        "--shape-prior-start-policy",
        choices=SHAPE_PRIOR_START_POLICIES,
        default=SHAPE_PRIOR_START_POLICY_AFTER_TEARDOWN,
        help=(
            "Schedule the full FuturePhysTwin/SAM3D shape-prior route. "
            "after-teardown snapshots first valid inputs but runs the heavy GPU route after live workers exit."
        ),
    )
    group.add_argument(
        "--shape-prior-gpu",
        default="auto",
        help=(
            "CUDA_VISIBLE_DEVICES used by FuturePhysTwin shape-prior subprocesses. "
            "auto uses the mask/render GPU; the heavy route starts after first render to protect startup."
        ),
    )
    group.add_argument(
        "--shape-prior-cuda-alloc-conf",
        default="expandable_segments:True",
        help="PYTORCH_CUDA_ALLOC_CONF for FuturePhysTwin shape-prior subprocesses.",
    )
    group.add_argument(
        "--shape-prior-retry-after-teardown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If the background shape-prior route fails during live GPU contention, "
            "retry the same snapshot after live workers have torn down."
        ),
    )
    group.add_argument(
        "--shape-prior-skip-route-visualizations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip optional FuturePhysTwin diagnostic videos for Demo 3.3 while still "
            "running shape prior, alignment, sampling, and final_data.pkl generation."
        ),
    )
    return parser


def apply_preset_defaults(
    args: argparse.Namespace,
    *,
    explicit_options: set[str] | None = None,
) -> argparse.Namespace:
    explicit = set(explicit_options or set())
    args = demo32.apply_preset_defaults(args, explicit_options=explicit)
    if "--overlay-display-scope" not in explicit:
        args.overlay_display_scope = demo31.SURFACE_ANCHOR_LABEL_UNION
    return args


def validate_args(
    args: argparse.Namespace,
    *,
    require_calibration: bool = False,
    cuda_device_count_provider: demo31.CudaDeviceCountProvider | None = None,
) -> None:
    demo32.validate_args(
        args,
        require_calibration=require_calibration,
        cuda_device_count_provider=cuda_device_count_provider,
    )
    if not bool(getattr(args, "shape_prior_warmup", DEFAULT_SHAPE_PRIOR_WARMUP)):
        return
    camera_id_list = [int(item) for item in getattr(args, "camera_ids", [])]
    camera_ids = set(camera_id_list)
    camera_idx = int(getattr(args, "shape_prior_camera_idx", DEFAULT_SHAPE_PRIOR_CAMERA_IDX))
    if camera_id_list != [0, 1, 2]:
        raise ValueError("Demo 3.3 shape-prior warmup requires --camera-ids 0,1,2 in that order for FuturePhysTwin align.py.")
    if camera_idx not in camera_ids:
        raise ValueError(f"Demo 3.3 --shape-prior-camera-idx {camera_idx} is not in --camera-ids {sorted(camera_ids)}.")
    if not bool(getattr(args, "dry_run", False)):
        future_root = Path(getattr(args, "futurephystwin_root", DEFAULT_FUTUREPHYSTWIN_ROOT))
        sam3d_root = Path(getattr(args, "sam3d_root", DEFAULT_SAM3D_ROOT))
        if not future_root.is_dir():
            raise FileNotFoundError(f"Missing FuturePhysTwin root for Demo 3.3: {future_root}")
        validate_original_sam3d_root(sam3d_root)
    shape_prior_gpu = str(getattr(args, "shape_prior_gpu", "auto") or "auto").strip()
    if not shape_prior_gpu:
        raise ValueError("Demo 3.3 --shape-prior-gpu must be a CUDA device id or 'auto'.")


def validate_live_contract(
    args: argparse.Namespace,
    *,
    connected_serials_provider: demo31.ConnectedSerialsProvider | None = None,
    cuda_device_count_provider: demo31.CudaDeviceCountProvider | None = None,
) -> dict[str, Any]:
    return demo32.validate_live_contract(
        args,
        connected_serials_provider=connected_serials_provider,
        cuda_device_count_provider=cuda_device_count_provider,
    )


def build_contract(
    args: argparse.Namespace,
    *,
    cuda_device_count_provider: demo31.CudaDeviceCountProvider | None = None,
) -> dict[str, Any]:
    contract = demo32.build_contract(args, cuda_device_count_provider=cuda_device_count_provider)
    contract.update(
        {
            "demo": "demo3.3",
            "runtime_module": DEMO33_RUNTIME_MODULE,
            "runtime_owner": DEMO33_RUNTIME_OWNER,
            "demo_display_name": "Demo 3.3",
            "derived_from_demo32_runtime": True,
            "shape_prior_usage": "render_only_canonical_reference",
        }
    )
    contract.update(_shape_prior_contract_fields(args))
    contract["profile_summary_fields"] = demo31.build_empty_dual_gpu_profile_summary(contract)
    contract["profile_summary_fields"].update(_shape_prior_profile_fields(contract))
    return contract


def format_contract(contract: dict[str, Any]) -> str:
    shape_keys = (
        "shape_prior_warmup_enabled",
        "shape_prior_status",
        "shape_prior_route",
        "futurephystwin_root",
        "futurephystwin_python",
        "sam3d_root",
        "shape_prior_camera_idx",
        "shape_prior_coordinate_frame",
        "shape_prior_units",
        "shape_prior_ground_policy",
        "shape_prior_coordinate_validation_status",
        "shape_prior_execution_mode",
        "shape_prior_start_policy",
        "shape_prior_gpu",
        "shape_prior_cuda_visible_devices",
        "shape_prior_cuda_alloc_conf",
        "shape_prior_retry_after_teardown",
        "shape_prior_skip_route_visualizations",
        "shape_prior_blocks_tracker_input",
        "shape_prior_blocks_first_render",
        "shape_prior_affects_tracker_input",
        "shape_prior_affects_live_observation_pcd",
    )
    prefix: list[str] = []
    for key in shape_keys:
        value = contract[key]
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        prefix.append(f"{key} = {rendered}")
    return "\n".join([*prefix, demo32.format_contract(contract)])


def build_shared_runtime_args(
    args: argparse.Namespace,
    *,
    shared_runtime_module: Any | None,
    live_validation: dict[str, Any],
    shared_profile_path: Path | None,
) -> argparse.Namespace:
    shared_args = demo32.build_shared_runtime_args(
        args,
        shared_runtime_module=shared_runtime_module,
        live_validation=live_validation,
        shared_profile_path=shared_profile_path,
    )
    shared_args.demo_version_override = "demo3.3"
    shared_args.demo_display_name_override = "Demo 3.3"
    shared_args.demo33_independent_runtime = True
    shared_args.shape_prior_warmup_enabled = bool(
        getattr(args, "shape_prior_warmup", DEFAULT_SHAPE_PRIOR_WARMUP)
    )
    shared_args.futurephystwin_root = str(getattr(args, "futurephystwin_root", DEFAULT_FUTUREPHYSTWIN_ROOT))
    shared_args.futurephystwin_python = str(getattr(args, "futurephystwin_python", DEFAULT_FUTUREPHYSTWIN_PYTHON))
    shared_args.sam3d_root = str(getattr(args, "sam3d_root", DEFAULT_SAM3D_ROOT))
    shared_args.shape_prior_camera_idx = int(getattr(args, "shape_prior_camera_idx", DEFAULT_SHAPE_PRIOR_CAMERA_IDX))
    shared_args.shape_prior_force = bool(getattr(args, "shape_prior_force", False))
    shared_args.shape_prior_start_policy = str(
        getattr(args, "shape_prior_start_policy", SHAPE_PRIOR_START_POLICY_AFTER_TEARDOWN)
    )
    shared_args.shape_prior_gpu = str(getattr(args, "shape_prior_gpu", "auto") or "auto")
    shared_args.shape_prior_cuda_alloc_conf = str(
        getattr(args, "shape_prior_cuda_alloc_conf", "expandable_segments:True") or ""
    )
    shared_args.shape_prior_retry_after_teardown = bool(getattr(args, "shape_prior_retry_after_teardown", True))
    shared_args.shape_prior_skip_route_visualizations = bool(
        getattr(args, "shape_prior_skip_route_visualizations", True)
    )
    return shared_args


def make_demo33_live_runtime_class(
    shared_runtime_module: Any,
    *,
    process_client_factory: demo31.ProcessClientFactory | None = None,
    shape_prior_runner: SubprocessRunner | None = None,
):
    base_cls = demo32.make_demo32_live_runtime_class(
        shared_runtime_module,
        process_client_factory=process_client_factory,
    )

    class Demo33LiveRuntime(base_cls):
        """Demo 3.3 runtime with one-shot FuturePhysTwin/SAM3D warmup prior."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            enabled = bool(self.demo31_contract.get("shape_prior_warmup_enabled", False))
            self.demo33_shape_prior_result = None
            self.demo33_shape_prior_profile: dict[str, Any] = _shape_prior_profile_fields(self.demo31_contract)
            self.demo33_shape_prior_profile["shape_prior_status"] = "pending" if enabled else "disabled"
            self.demo33_shape_prior_run_id = time.strftime("%Y%m%d-%H%M%S")
            self.demo33_shape_prior_runner = shape_prior_runner
            self.demo33_shape_prior_thread: threading.Thread | None = None
            self.demo33_shape_prior_lock = threading.Lock()
            self.demo33_shape_prior_error: BaseException | None = None
            self.demo33_shape_prior_started_s: float | None = None
            self.demo33_shape_prior_completed_s: float | None = None
            self.demo33_shape_prior_pending_kwargs: dict[str, Any] | None = None
            self.demo33_shape_prior_last_kwargs: dict[str, Any] | None = None
            self.demo33_shape_prior_detached_completion_pid: int | None = None

        def _ensure_shape_prior_async_state(self) -> None:
            if not hasattr(self, "demo33_shape_prior_lock"):
                self.demo33_shape_prior_lock = threading.Lock()
            if not hasattr(self, "demo33_shape_prior_thread"):
                self.demo33_shape_prior_thread = None
            if not hasattr(self, "demo33_shape_prior_error"):
                self.demo33_shape_prior_error = None
            if not hasattr(self, "demo33_shape_prior_started_s"):
                self.demo33_shape_prior_started_s = None
            if not hasattr(self, "demo33_shape_prior_completed_s"):
                self.demo33_shape_prior_completed_s = None
            if not hasattr(self, "demo33_shape_prior_pending_kwargs"):
                self.demo33_shape_prior_pending_kwargs = None
            if not hasattr(self, "demo33_shape_prior_last_kwargs"):
                self.demo33_shape_prior_last_kwargs = None
            if not hasattr(self, "demo33_shape_prior_detached_completion_pid"):
                self.demo33_shape_prior_detached_completion_pid = None

        @staticmethod
        def _snapshot_array_mapping(mapping: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
            return {
                int(camera_idx): np.ascontiguousarray(np.asarray(value).copy())
                for camera_idx, value in mapping.items()
            }

        def _shape_prior_profile_snapshot(self) -> dict[str, Any]:
            self._ensure_shape_prior_async_state()
            thread = self.demo33_shape_prior_thread
            with self.demo33_shape_prior_lock:
                snapshot = dict(self.demo33_shape_prior_profile)
                snapshot["shape_prior_thread_alive"] = bool(thread is not None and thread.is_alive())
                if self.demo33_shape_prior_started_s is not None:
                    end_s = (
                        self.demo33_shape_prior_completed_s
                        if self.demo33_shape_prior_completed_s is not None
                        else time.perf_counter()
                    )
                    snapshot["shape_prior_async_elapsed_ms"] = float(
                        max(0.0, end_s - self.demo33_shape_prior_started_s) * 1000.0
                    )
                return snapshot

        def _run_shape_prior_warmup_background(
            self,
            *,
            config: ShapePriorWarmupConfig,
            rgb_by_camera: dict[int, np.ndarray],
            depth_by_camera: dict[int, np.ndarray],
            object_mask_by_camera: dict[int, np.ndarray],
            controller_mask_by_camera: dict[int, np.ndarray],
            intrinsics_by_camera: dict[int, np.ndarray],
            c2w_by_camera: dict[int, np.ndarray],
            camera_ids: Sequence[int],
            source_group_id: int,
        ) -> None:
            self._ensure_shape_prior_async_state()
            try:
                result = run_shape_prior_warmup(
                    config=config,
                    rgb_by_camera=rgb_by_camera,
                    depth_by_camera=depth_by_camera,
                    object_mask_by_camera=object_mask_by_camera,
                    controller_mask_by_camera=controller_mask_by_camera,
                    intrinsics_by_camera=intrinsics_by_camera,
                    c2w_by_camera=c2w_by_camera,
                    camera_ids=camera_ids,
                    source_group_id=int(source_group_id),
                    runner=self.demo33_shape_prior_runner or demo31.subprocess.run,
                )
            except BaseException as exc:
                with self.demo33_shape_prior_lock:
                    self.demo33_shape_prior_error = exc
                    self.demo33_shape_prior_completed_s = time.perf_counter()
                    elapsed_ms = 0.0
                    if self.demo33_shape_prior_started_s is not None:
                        elapsed_ms = float(
                            (self.demo33_shape_prior_completed_s - self.demo33_shape_prior_started_s) * 1000.0
                        )
                    self.demo33_shape_prior_profile.update(
                        {
                            "shape_prior_status": "error",
                            "shape_prior_error": f"{type(exc).__name__}: {exc}",
                            "shape_prior_last_error": f"{type(exc).__name__}: {exc}",
                            "shape_prior_async_completed": True,
                            "shape_prior_thread_alive": False,
                            "shape_prior_async_elapsed_ms": elapsed_ms,
                            "shape_prior_snapshot_ready": False,
                        }
                    )
                return
            with self.demo33_shape_prior_lock:
                self.demo33_shape_prior_result = result
                self.demo33_shape_prior_completed_s = time.perf_counter()
                elapsed_ms = 0.0
                if self.demo33_shape_prior_started_s is not None:
                    elapsed_ms = float((self.demo33_shape_prior_completed_s - self.demo33_shape_prior_started_s) * 1000.0)
                self.demo33_shape_prior_profile.update(result.profile)
                self.demo33_shape_prior_profile.update(
                    {
                        "shape_prior_async_started": True,
                        "shape_prior_async_completed": True,
                        "shape_prior_thread_alive": False,
                        "shape_prior_async_elapsed_ms": elapsed_ms,
                        "shape_prior_execution_mode": "async_background_thread",
                        "shape_prior_snapshot_ready": False,
                        "shape_prior_blocks_tracker_input": False,
                        "shape_prior_blocks_first_render": False,
                    }
                )

        def _run_shape_prior_route_background(
            self,
            *,
            config: ShapePriorWarmupConfig,
            source_group_id: int | None = None,
        ) -> None:
            self._ensure_shape_prior_async_state()
            try:
                command_records = run_futurephystwin_single_view_route(
                    config=config,
                    runner=self.demo33_shape_prior_runner or demo31.subprocess.run,
                )
                result = load_shape_prior_final_data(config.case_dir)
                result.profile.update(
                    {
                        "shape_prior_warmup_enabled": True,
                        "shape_prior_source_group_id": int(
                            source_group_id
                            if source_group_id is not None
                            else result.profile.get("shape_prior_source_group_id", -1)
                        ),
                        "shape_prior_command_records": command_records,
                        "shape_prior_command_order": [record["stage"] for record in command_records],
                    }
                )
            except BaseException as exc:
                with self.demo33_shape_prior_lock:
                    self.demo33_shape_prior_error = exc
                    self.demo33_shape_prior_completed_s = time.perf_counter()
                    elapsed_ms = 0.0
                    if self.demo33_shape_prior_started_s is not None:
                        elapsed_ms = float(
                            (self.demo33_shape_prior_completed_s - self.demo33_shape_prior_started_s) * 1000.0
                        )
                    self.demo33_shape_prior_profile.update(
                        {
                            "shape_prior_status": "error",
                            "shape_prior_error": f"{type(exc).__name__}: {exc}",
                            "shape_prior_last_error": f"{type(exc).__name__}: {exc}",
                            "shape_prior_async_completed": True,
                            "shape_prior_thread_alive": False,
                            "shape_prior_async_elapsed_ms": elapsed_ms,
                            "shape_prior_snapshot_ready": False,
                        }
                    )
                return
            with self.demo33_shape_prior_lock:
                self.demo33_shape_prior_result = result
                self.demo33_shape_prior_completed_s = time.perf_counter()
                elapsed_ms = 0.0
                if self.demo33_shape_prior_started_s is not None:
                    elapsed_ms = float((self.demo33_shape_prior_completed_s - self.demo33_shape_prior_started_s) * 1000.0)
                self.demo33_shape_prior_profile.update(result.profile)
                self.demo33_shape_prior_profile.update(
                    {
                        "shape_prior_async_started": True,
                        "shape_prior_async_completed": True,
                        "shape_prior_thread_alive": False,
                        "shape_prior_async_elapsed_ms": elapsed_ms,
                        "shape_prior_execution_mode": "async_background_thread",
                        "shape_prior_snapshot_ready": False,
                        "shape_prior_blocks_tracker_input": False,
                        "shape_prior_blocks_first_render": False,
                    }
                )

        def _wait_for_shape_prior_warmup(self) -> dict[str, Any]:
            self._ensure_shape_prior_async_state()
            if self.demo33_shape_prior_thread is None and self.demo33_shape_prior_pending_kwargs is not None:
                self._start_pending_shape_prior_warmup("teardown_wait")
            thread = self.demo33_shape_prior_thread
            if thread is not None:
                thread.join()
            with self.demo33_shape_prior_lock:
                if self.demo33_shape_prior_thread is not None and not self.demo33_shape_prior_thread.is_alive():
                    self.demo33_shape_prior_thread = None
                should_retry = (
                    bool(self.demo33_shape_prior_profile.get("shape_prior_retry_after_teardown", False))
                    and str(self.demo33_shape_prior_profile.get("shape_prior_status", "")) == "error"
                    and int(self.demo33_shape_prior_profile.get("shape_prior_retry_count", 0) or 0) <= 0
                    and self.demo33_shape_prior_result is None
                    and self.demo33_shape_prior_last_kwargs is not None
                )
                if should_retry:
                    self.demo33_shape_prior_pending_kwargs = dict(self.demo33_shape_prior_last_kwargs or {})
                    self.demo33_shape_prior_profile.update(
                        {
                            "shape_prior_status": "retry_pending",
                            "shape_prior_snapshot_ready": True,
                            "shape_prior_thread_alive": False,
                        }
                    )
            if should_retry:
                self._start_pending_shape_prior_warmup("teardown_retry")
                retry_thread = self.demo33_shape_prior_thread
                if retry_thread is not None:
                    retry_thread.join()
                with self.demo33_shape_prior_lock:
                    if self.demo33_shape_prior_thread is not None and not self.demo33_shape_prior_thread.is_alive():
                        self.demo33_shape_prior_thread = None
            return self._shape_prior_profile_snapshot()

        def _shape_prior_labels(self) -> tuple[str, str]:
            object_label = str(self.demo31_contract.get("object_prompt", "stuffed animal"))
            controller_label = str(
                self.demo31_contract.get(
                    "tracking_controller_label",
                    self.demo31_contract.get("controller_prompt", "towel"),
                )
            )
            return object_label, controller_label

        def _shape_prior_config(self) -> ShapePriorWarmupConfig:
            object_label, controller_label = self._shape_prior_labels()
            return ShapePriorWarmupConfig(
                enabled=bool(self.demo31_contract.get("shape_prior_warmup_enabled", False)),
                output_root=Path(getattr(self.args, "output_root", "result/demo32_ffs_tapnextpp")),
                run_id=self.demo33_shape_prior_run_id,
                futurephystwin_root=Path(self.demo31_contract.get("futurephystwin_root", DEFAULT_FUTUREPHYSTWIN_ROOT)),
                futurephystwin_python=str(
                    self.demo31_contract.get("futurephystwin_python", DEFAULT_FUTUREPHYSTWIN_PYTHON)
                ),
                sam3d_root=Path(self.demo31_contract.get("sam3d_root", DEFAULT_SAM3D_ROOT)),
                camera_idx=int(self.demo31_contract.get("shape_prior_camera_idx", DEFAULT_SHAPE_PRIOR_CAMERA_IDX)),
                force=bool(self.demo31_contract.get("shape_prior_force", False)),
                object_label=object_label,
                controller_label=controller_label,
                ground_policy=DEFAULT_SHAPE_PRIOR_GROUND_POLICY,
                ground_z=DEFAULT_SHAPE_PRIOR_GROUND_Z,
                cuda_visible_devices=str(self.demo31_contract.get("shape_prior_cuda_visible_devices", "") or ""),
                cuda_allocator_config=str(self.demo31_contract.get("shape_prior_cuda_alloc_conf", "") or ""),
                skip_route_visualizations=bool(
                    self.demo31_contract.get("shape_prior_skip_route_visualizations", False)
                ),
            )

        def _shape_prior_completion_paths(self) -> tuple[Path | None, Path | None]:
            path = getattr(self.args, "demo31_top_level_profile_json_output", None)
            if path is None:
                return None, None
            profile_path = Path(path)
            completion_json = profile_path.with_name(f"{profile_path.stem}_shape_prior_completion.json")
            completion_log = completion_json.with_suffix(".log")
            return completion_json, completion_log

        def _launch_detached_shape_prior_completion(self) -> dict[str, Any]:
            self._ensure_shape_prior_async_state()
            profile_path_raw = getattr(self.args, "demo31_top_level_profile_json_output", None)
            if profile_path_raw is None:
                return {}
            shape_profile = self._shape_prior_profile_snapshot()
            if str(shape_profile.get("shape_prior_status", "")) not in {
                "case_ready",
                "snapshot_ready",
                "retry_pending",
            }:
                return {}
            if self.demo33_shape_prior_detached_completion_pid:
                return {}
            case_dir = Path(str(shape_profile.get("shape_prior_case_dir", ""))).expanduser()
            if not case_dir.is_dir():
                return {}
            completion_json, completion_log = self._shape_prior_completion_paths()
            if completion_json is None or completion_log is None:
                return {}
            object_label, controller_label = self._shape_prior_labels()
            command = [
                sys.executable,
                "-m",
                "qqtt.demo.demo33_shape_prior_completion",
                "--profile-json",
                str(Path(profile_path_raw)),
                "--completion-json",
                str(completion_json),
                "--case-dir",
                str(case_dir),
                "--futurephystwin-root",
                str(self.demo31_contract.get("futurephystwin_root", DEFAULT_FUTUREPHYSTWIN_ROOT)),
                "--futurephystwin-python",
                str(self.demo31_contract.get("futurephystwin_python", DEFAULT_FUTUREPHYSTWIN_PYTHON)),
                "--sam3d-root",
                str(self.demo31_contract.get("sam3d_root", DEFAULT_SAM3D_ROOT)),
                "--shape-prior-camera-idx",
                str(int(self.demo31_contract.get("shape_prior_camera_idx", DEFAULT_SHAPE_PRIOR_CAMERA_IDX))),
                "--object-label",
                object_label,
                "--controller-label",
                controller_label,
                "--ground-policy",
                DEFAULT_SHAPE_PRIOR_GROUND_POLICY,
                "--ground-z",
                str(float(DEFAULT_SHAPE_PRIOR_GROUND_Z)),
                "--cuda-visible-devices",
                str(self.demo31_contract.get("shape_prior_cuda_visible_devices", "") or ""),
                "--cuda-alloc-conf",
                str(self.demo31_contract.get("shape_prior_cuda_alloc_conf", "") or ""),
                "--wait-for-pid",
                str(os.getpid()),
            ]
            if bool(self.demo31_contract.get("shape_prior_skip_route_visualizations", False)):
                command.append("--skip-route-visualizations")
            completion_json.parent.mkdir(parents=True, exist_ok=True)
            completion_log.parent.mkdir(parents=True, exist_ok=True)
            with completion_log.open("ab") as log_handle:
                process = demo31.subprocess.Popen(
                    command,
                    cwd=str(Path(__file__).resolve().parents[2]),
                    stdout=log_handle,
                    stderr=demo31.subprocess.STDOUT,
                    start_new_session=True,
                )
            self.demo33_shape_prior_detached_completion_pid = int(process.pid)
            update = {
                "shape_prior_detached_completion_started": True,
                "shape_prior_detached_completion_pid": int(process.pid),
                "shape_prior_detached_completion_json": str(completion_json),
                "shape_prior_detached_completion_log": str(completion_log),
                "shape_prior_detached_completion_wait_for_pid": int(os.getpid()),
                "shape_prior_start_trigger": "after_teardown_detached",
                "shape_prior_blocks_tracker_input": False,
                "shape_prior_blocks_first_render": False,
            }
            with self.demo33_shape_prior_lock:
                self.demo33_shape_prior_profile.update(update)
            return update

        def _start_pending_shape_prior_warmup(self, trigger: str) -> bool:
            self._ensure_shape_prior_async_state()
            if self.demo33_shape_prior_thread is not None:
                return False
            pending = self.demo33_shape_prior_pending_kwargs
            if pending is None:
                return False
            started_s = time.perf_counter()
            pending_payload = dict(pending)
            route_only = bool(pending_payload.pop("route_only", False))
            target = self._run_shape_prior_route_background if route_only else self._run_shape_prior_warmup_background
            worker = threading.Thread(
                target=target,
                name=f"demo33-shape-prior-{int(pending.get('source_group_id', -1))}",
                kwargs=pending_payload,
                daemon=False,
            )
            with self.demo33_shape_prior_lock:
                self.demo33_shape_prior_pending_kwargs = None
                self.demo33_shape_prior_last_kwargs = dict(pending)
                self.demo33_shape_prior_started_s = started_s
                self.demo33_shape_prior_completed_s = None
                self.demo33_shape_prior_thread = worker
                retry_count = int(self.demo33_shape_prior_profile.get("shape_prior_retry_count", 0) or 0)
                if str(trigger) == "teardown_retry":
                    retry_count += 1
                self.demo33_shape_prior_profile.update(
                    {
                        "shape_prior_status": "running",
                        "shape_prior_start_trigger": str(trigger),
                        "shape_prior_retry_count": retry_count,
                        "shape_prior_async_started": True,
                        "shape_prior_async_completed": False,
                        "shape_prior_thread_alive": True,
                        "shape_prior_async_elapsed_ms": 0.0,
                        "shape_prior_snapshot_ready": False,
                        "shape_prior_blocks_tracker_input": False,
                        "shape_prior_blocks_first_render": False,
                        "shape_prior_affects_tracker_input": False,
                        "shape_prior_affects_live_observation_pcd": False,
                    }
                )
            worker.start()
            return True

        def _maybe_build_shape_prior_warmup(
            self,
            *,
            group_id: int,
            timestamp_s: float,
            rgb_by_camera: dict[int, np.ndarray],
            depth_by_camera: dict[int, np.ndarray],
            object_mask_by_camera: dict[int, np.ndarray],
            controller_mask_by_camera: dict[int, np.ndarray],
            intrinsics_by_camera: dict[int, np.ndarray],
            c2w_by_camera: dict[int, np.ndarray],
            camera_ids: Sequence[int],
        ) -> dict[str, Any]:
            _ = timestamp_s
            self._ensure_shape_prior_async_state()
            if not bool(self.demo31_contract.get("shape_prior_warmup_enabled", False)):
                return self._shape_prior_profile_snapshot()
            if self.demo33_shape_prior_result is not None:
                return self._shape_prior_profile_snapshot()
            thread = self.demo33_shape_prior_thread
            if thread is not None:
                return self._shape_prior_profile_snapshot()
            if self.demo33_shape_prior_pending_kwargs is not None:
                start_policy = str(
                    self.demo31_contract.get(
                        "shape_prior_start_policy",
                        SHAPE_PRIOR_START_POLICY_AFTER_TEARDOWN,
                    )
                )
                if (
                    start_policy == SHAPE_PRIOR_START_POLICY_AFTER_FIRST_RENDER
                    and self.demo31_tracking_overlay_first_render_group_id is not None
                ):
                    self._start_pending_shape_prior_warmup("after_first_render")
                return self._shape_prior_profile_snapshot()
            config = self._shape_prior_config()
            rgb_snapshot = self._snapshot_array_mapping(rgb_by_camera)
            depth_snapshot = self._snapshot_array_mapping(depth_by_camera)
            object_mask_snapshot = self._snapshot_array_mapping(object_mask_by_camera)
            controller_mask_snapshot = self._snapshot_array_mapping(controller_mask_by_camera)
            intrinsics_snapshot = self._snapshot_array_mapping(intrinsics_by_camera)
            c2w_snapshot = self._snapshot_array_mapping(c2w_by_camera)
            camera_id_snapshot = [int(camera_idx) for camera_idx in camera_ids]
            try:
                case_profile = write_futurephystwin_warmup_case(
                    config=config,
                    rgb_by_camera=rgb_snapshot,
                    depth_by_camera=depth_snapshot,
                    object_mask_by_camera=object_mask_snapshot,
                    controller_mask_by_camera=controller_mask_snapshot,
                    intrinsics_by_camera=intrinsics_snapshot,
                    c2w_by_camera=c2w_snapshot,
                    camera_ids=camera_id_snapshot,
                    source_group_id=int(group_id),
                )
            except BaseException as exc:
                with self.demo33_shape_prior_lock:
                    self.demo33_shape_prior_error = exc
                    self.demo33_shape_prior_profile.update(
                        {
                            "shape_prior_status": "error",
                            "shape_prior_error": f"{type(exc).__name__}: {exc}",
                            "shape_prior_last_error": f"{type(exc).__name__}: {exc}",
                            "shape_prior_case_dir": str(config.case_dir),
                            "shape_prior_first_group_id": int(group_id),
                            "shape_prior_snapshot_group_id": int(group_id),
                            "shape_prior_snapshot_ready": False,
                            "shape_prior_blocks_tracker_input": False,
                            "shape_prior_blocks_first_render": False,
                            "shape_prior_affects_tracker_input": False,
                            "shape_prior_affects_live_observation_pcd": False,
                        }
                    )
                return self._shape_prior_profile_snapshot()
            pending_kwargs = {
                "route_only": True,
                "config": config,
                "source_group_id": int(group_id),
            }
            start_policy = str(
                self.demo31_contract.get(
                    "shape_prior_start_policy",
                    SHAPE_PRIOR_START_POLICY_AFTER_TEARDOWN,
                )
            )
            with self.demo33_shape_prior_lock:
                self.demo33_shape_prior_pending_kwargs = dict(pending_kwargs)
                self.demo33_shape_prior_profile.update(case_profile)
                self.demo33_shape_prior_profile.update(
                    {
                        "shape_prior_warmup_enabled": True,
                        "shape_prior_status": "case_ready",
                        "shape_prior_case_dir": str(config.case_dir),
                        "shape_prior_first_group_id": int(group_id),
                        "shape_prior_snapshot_ready": True,
                        "shape_prior_snapshot_group_id": int(group_id),
                        "shape_prior_execution_mode": "async_background_thread",
                        "shape_prior_start_policy": start_policy,
                        "shape_prior_gpu": str(self.demo31_contract.get("shape_prior_gpu", "")),
                        "shape_prior_cuda_visible_devices": str(
                            self.demo31_contract.get("shape_prior_cuda_visible_devices", "")
                        ),
                        "shape_prior_cuda_alloc_conf": str(
                            self.demo31_contract.get("shape_prior_cuda_alloc_conf", "")
                        ),
                        "shape_prior_skip_route_visualizations": bool(
                            self.demo31_contract.get("shape_prior_skip_route_visualizations", False)
                        ),
                        "shape_prior_start_trigger": "",
                        "shape_prior_async_started": False,
                        "shape_prior_async_completed": False,
                        "shape_prior_thread_alive": False,
                        "shape_prior_async_elapsed_ms": 0.0,
                        "shape_prior_blocks_tracker_input": False,
                        "shape_prior_blocks_first_render": False,
                        "shape_prior_affects_tracker_input": False,
                        "shape_prior_affects_live_observation_pcd": False,
                    }
                )
            if start_policy == SHAPE_PRIOR_START_POLICY_IMMEDIATE:
                self._start_pending_shape_prior_warmup("immediate")
            elif (
                start_policy == SHAPE_PRIOR_START_POLICY_AFTER_FIRST_RENDER
                and self.demo31_tracking_overlay_first_render_group_id is not None
            ):
                self._start_pending_shape_prior_warmup("after_first_render")
            return self._shape_prior_profile_snapshot()

        def _attach_shape_prior_render_layer(self, packet: Any) -> Any:
            self._ensure_shape_prior_async_state()
            result = self.demo33_shape_prior_result
            if result is None:
                return packet
            if result.status != "ready" or len(result.structure_points) == 0:
                return packet
            if not hasattr(packet, "shape_prior_points_m"):
                return packet
            return replace(
                packet,
                shape_prior_points_m=result.structure_points,
                shape_prior_colors_rgb=result.structure_colors_rgb,
                shape_prior_profile=dict(result.profile),
            )

        def _remember_pending_render_packet(self, packet: Any) -> None:
            super()._remember_pending_render_packet(self._attach_shape_prior_render_layer(packet))

        def _publish_render_packet(self, packet: Any) -> None:
            super()._publish_render_packet(self._attach_shape_prior_render_layer(packet))
            if (
                str(
                    self.demo31_contract.get(
                        "shape_prior_start_policy",
                        SHAPE_PRIOR_START_POLICY_AFTER_TEARDOWN,
                    )
                )
                == SHAPE_PRIOR_START_POLICY_AFTER_FIRST_RENDER
            ):
                self._start_pending_shape_prior_warmup("after_first_render")

        def _build_profile_payload(self) -> dict[str, Any]:
            payload = super()._build_profile_payload()
            return _merge_shape_prior_profile_into_payload(payload, self._shape_prior_profile_snapshot())

        def _write_demo31_pre_teardown_profile(self) -> None:
            if (
                str(
                    self.demo31_contract.get(
                        "shape_prior_start_policy",
                        SHAPE_PRIOR_START_POLICY_AFTER_TEARDOWN,
                    )
                )
                == SHAPE_PRIOR_START_POLICY_AFTER_TEARDOWN
            ):
                self._launch_detached_shape_prior_completion()
            super()._write_demo31_pre_teardown_profile()

        def demo31_snapshot(self) -> dict[str, Any]:
            snapshot = super().demo31_snapshot()
            shape_profile = self._shape_prior_profile_snapshot()
            snapshot["shape_prior_warmup"] = shape_profile
            snapshot.update(shape_profile)
            return snapshot

    Demo33LiveRuntime.__name__ = "Demo33LiveRuntime"
    return Demo33LiveRuntime


class Demo33Runtime(demo32.Demo32Runtime):
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        shared_runtime_module: Any | None = None,
        shared_runtime_cls: type | None = None,
        connected_serials_provider: demo31.ConnectedSerialsProvider | None = None,
        cuda_device_count_provider: demo31.CudaDeviceCountProvider | None = None,
        process_client_factory: demo31.ProcessClientFactory | None = None,
        shape_prior_runner: SubprocessRunner | None = None,
    ) -> None:
        super().__init__(
            args,
            shared_runtime_module=shared_runtime_module,
            shared_runtime_cls=shared_runtime_cls,
            connected_serials_provider=connected_serials_provider,
            cuda_device_count_provider=cuda_device_count_provider,
            process_client_factory=process_client_factory,
        )
        self.contract = build_contract(args, cuda_device_count_provider=cuda_device_count_provider)
        self.shape_prior_runner = shape_prior_runner

    def _complete_shape_prior_case_from_profile(self, shape_profile: dict[str, Any]) -> dict[str, Any]:
        case_dir = Path(str(shape_profile.get("shape_prior_case_dir", ""))).expanduser()
        if not case_dir.is_dir():
            return {}
        run_id = case_dir.parent.name
        output_root = case_dir.parents[2]
        object_label = str(self.contract.get("object_prompt", "stuffed animal"))
        controller_label = str(
            self.contract.get(
                "tracking_controller_label",
                self.contract.get("controller_prompt", "towel"),
            )
        )
        config = ShapePriorWarmupConfig(
            enabled=True,
            output_root=output_root,
            run_id=run_id,
            futurephystwin_root=Path(self.contract.get("futurephystwin_root", DEFAULT_FUTUREPHYSTWIN_ROOT)),
            futurephystwin_python=str(
                self.contract.get("futurephystwin_python", DEFAULT_FUTUREPHYSTWIN_PYTHON)
            ),
            sam3d_root=Path(self.contract.get("sam3d_root", DEFAULT_SAM3D_ROOT)),
            camera_idx=int(self.contract.get("shape_prior_camera_idx", DEFAULT_SHAPE_PRIOR_CAMERA_IDX)),
            force=bool(self.contract.get("shape_prior_force", False)),
            object_label=object_label,
            controller_label=controller_label,
            ground_policy=DEFAULT_SHAPE_PRIOR_GROUND_POLICY,
            ground_z=DEFAULT_SHAPE_PRIOR_GROUND_Z,
            cuda_visible_devices=str(self.contract.get("shape_prior_cuda_visible_devices", "") or ""),
            cuda_allocator_config=str(self.contract.get("shape_prior_cuda_alloc_conf", "") or ""),
            skip_route_visualizations=bool(self.contract.get("shape_prior_skip_route_visualizations", False)),
        )
        start_s = time.perf_counter()
        base_profile = dict(shape_profile)
        base_profile.update(
            {
                "shape_prior_status": "running",
                "shape_prior_start_trigger": "teardown_wait",
                "shape_prior_async_started": True,
                "shape_prior_async_completed": False,
                "shape_prior_thread_alive": False,
                "shape_prior_snapshot_ready": False,
            }
        )
        try:
            command_records = run_futurephystwin_single_view_route(
                config=config,
                runner=self.shape_prior_runner or demo31.subprocess.run,
            )
            result = load_shape_prior_final_data(config.case_dir)
        except BaseException as exc:
            elapsed_ms = float((time.perf_counter() - start_s) * 1000.0)
            base_profile.update(
                {
                    "shape_prior_status": "error",
                    "shape_prior_error": f"{type(exc).__name__}: {exc}",
                    "shape_prior_last_error": f"{type(exc).__name__}: {exc}",
                    "shape_prior_async_completed": True,
                    "shape_prior_async_elapsed_ms": elapsed_ms,
                    "shape_prior_render_layer_enabled": False,
                }
            )
            return base_profile
        elapsed_ms = float((time.perf_counter() - start_s) * 1000.0)
        result.profile.update(
            {
                "shape_prior_warmup_enabled": True,
                "shape_prior_command_records": command_records,
                "shape_prior_command_order": [record["stage"] for record in command_records],
                "shape_prior_start_policy": str(self.contract.get("shape_prior_start_policy", "")),
                "shape_prior_start_trigger": "teardown_wait",
                "shape_prior_gpu": str(self.contract.get("shape_prior_gpu", "")),
                "shape_prior_cuda_visible_devices": str(self.contract.get("shape_prior_cuda_visible_devices", "")),
                "shape_prior_cuda_alloc_conf": str(self.contract.get("shape_prior_cuda_alloc_conf", "")),
                "shape_prior_retry_after_teardown": bool(self.contract.get("shape_prior_retry_after_teardown", True)),
                "shape_prior_skip_route_visualizations": bool(
                    self.contract.get("shape_prior_skip_route_visualizations", False)
                ),
                "shape_prior_retry_count": int(base_profile.get("shape_prior_retry_count", 0) or 0),
                "shape_prior_async_started": True,
                "shape_prior_async_completed": True,
                "shape_prior_thread_alive": False,
                "shape_prior_async_elapsed_ms": elapsed_ms,
                "shape_prior_snapshot_ready": False,
                "shape_prior_blocks_tracker_input": False,
                "shape_prior_blocks_first_render": False,
            }
        )
        return dict(result.profile)

    def run(self) -> dict[str, Any]:
        live_validation = validate_live_contract(
            self.args,
            connected_serials_provider=self.connected_serials_provider,
            cuda_device_count_provider=self.cuda_device_count_provider,
        )
        shared = self.shared_runtime_module or demo3_runtime._load_shared_runtime_module()
        shared_profile = demo3_runtime._shared_profile_path(self.args)
        shared_args = build_shared_runtime_args(
            self.args,
            shared_runtime_module=shared,
            live_validation=live_validation,
            shared_profile_path=shared_profile,
        )
        runtime_cls = self.shared_runtime_cls or make_demo33_live_runtime_class(
            shared,
            process_client_factory=self.process_client_factory,
            shape_prior_runner=self.shape_prior_runner,
        )
        if self.shared_runtime_cls is None:
            runtime = runtime_cls(
                shared_args,
                demo31_contract=self.contract,
                cotracker_process_config=demo31.build_cotracker_process_config(self.args),
                cotracker_enabled=not bool(self.args.disable_cotracker),
            )
        else:
            runtime = runtime_cls(shared_args)
        exit_code = int(runtime.run())
        if hasattr(runtime, "_wait_for_shape_prior_warmup"):
            runtime._wait_for_shape_prior_warmup()
            shape_status = str(getattr(runtime, "demo33_shape_prior_profile", {}).get("shape_prior_status", ""))
            if shape_status == "error":
                exit_code = 1
        shared_payload = demo3_runtime._load_json_if_exists(shared_profile)
        snapshot = runtime.demo31_snapshot() if hasattr(runtime, "demo31_snapshot") else None
        shape_profile = (snapshot or {}).get("shape_prior_warmup", {}) if snapshot else {}
        if isinstance(shape_profile, dict) and str(shape_profile.get("shape_prior_status", "")) in {
            "case_ready",
            "snapshot_ready",
            "error",
        }:
            completed_shape_profile = self._complete_shape_prior_case_from_profile(shape_profile)
            if completed_shape_profile:
                snapshot = dict(snapshot or {})
                snapshot["shape_prior_warmup"] = completed_shape_profile
                snapshot.update(completed_shape_profile)
                if str(completed_shape_profile.get("shape_prior_status", "")) == "error":
                    exit_code = 1
        summary = self._build_summary(
            runtime=runtime,
            exit_code=exit_code,
            snapshot=snapshot,
            shared_payload=shared_payload,
        )
        profile = {
            "contract": self.contract,
            "summary": summary,
            "live_validation": live_validation,
            "shared_runtime_profile": None if shared_profile is None else str(shared_profile),
            "shared_runtime_profile_payload": shared_payload,
            "tracker_process_snapshot": snapshot,
            "runtime_note": (
                "Demo 3.3 is the Demo 3.2 runtime plus a one-shot FuturePhysTwin/SAM3D "
                "warmup prior rendered as a gray canonical reference layer."
            ),
            "exit_code": exit_code,
        }
        demo31._write_profile(self.args.profile_json_output, profile)
        return profile

    def _build_summary(
        self,
        *,
        runtime: Any,
        exit_code: int,
        snapshot: dict[str, Any] | None,
        shared_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        summary = super()._build_summary(
            runtime=runtime,
            exit_code=exit_code,
            snapshot=snapshot,
            shared_payload=shared_payload,
        )
        shape_profile = (snapshot or {}).get("shape_prior_warmup", {}) if snapshot else {}
        if not isinstance(shape_profile, dict):
            shape_profile = {}
        shape_defaults = _shape_prior_profile_fields(self.contract)
        shape_defaults.update(shape_profile)
        summary.update(_shape_prior_profile_fields(shape_defaults))
        return summary


def main(
    argv: Sequence[str] | None = None,
    *,
    cuda_device_count_provider: demo31.CudaDeviceCountProvider | None = None,
) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(argv)
        args = apply_preset_defaults(args, explicit_options=demo3_runtime._explicit_cli_options(argv))
        validate_args(args, require_calibration=False, cuda_device_count_provider=cuda_device_count_provider)
        contract = build_contract(args, cuda_device_count_provider=cuda_device_count_provider)
        if args.dry_run:
            print(format_contract(contract))
            demo31._write_profile(
                args.profile_json_output,
                {"contract": contract, "summary": contract["profile_summary_fields"]},
            )
            return 0
        profile = Demo33Runtime(args, cuda_device_count_provider=cuda_device_count_provider).run()
        print(json.dumps(profile["summary"], indent=2, sort_keys=True))
        return int(profile.get("exit_code", 0))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


__all__ = [
    "DEMO33_RUNTIME_MODULE",
    "DEMO33_RUNTIME_OWNER",
    "Demo33Runtime",
    "PRESET_DEMO32_FFS_TAPNEXTPP",
    "PRESET_DEMO32_FFS_LITETRACKER",
    "apply_preset_defaults",
    "build_arg_parser",
    "build_contract",
    "build_shared_runtime_args",
    "format_contract",
    "main",
    "make_demo33_live_runtime_class",
    "validate_args",
    "validate_live_contract",
]
