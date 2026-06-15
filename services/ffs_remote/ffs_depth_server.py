#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


def _resolve_repo_root() -> Path:
    candidates = [Path(__file__).resolve().parents[2], Path.cwd()]
    env_root = os.environ.get("QQTT_REPO_ROOT")
    if env_root:
        candidates.insert(0, Path(env_root))
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "data_process").is_dir() and (root / "services").is_dir():
            return root
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.depth_backends.ffs_defaults import (  # noqa: E402
    DEFAULT_FFS_REPO,
    DEFAULT_FFS_MODEL_NAME,
    DEFAULT_FFS_MAX_DISP,
    DEFAULT_FFS_TRT_BUILDER_OPTIMIZATION_LEVEL,
    DEFAULT_FFS_TRT_ENGINE_SIZE,
    DEFAULT_FFS_TRT_INPUT_SIZE,
    DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR,
    DEFAULT_FFS_VALID_ITERS,
)
from data_process.depth_backends.geometry import quantize_depth_with_invalid_zero  # noqa: E402
from qqtt.demo.realtime_single_camera_pointcloud import FfsIrToColorAligner, warm_up_numba_ffs_align  # noqa: E402
from services.ffs_remote.protocol import (  # noqa: E402
    COMPRESSION_MODES,
    RETURN_TYPES,
    SPARSE_RETURN_TYPES,
    build_depth_response_parts,
    matrix_from_metadata,
    parse_depth_request_parts,
)


def _elapsed_ms(start_s: float, end_s: float | None = None) -> float:
    stop_s = time.perf_counter() if end_s is None else end_s
    return (stop_s - start_s) * 1000.0


class _CachedAligner:
    def __init__(self) -> None:
        self._key: tuple[Any, ...] | None = None
        self._aligner: FfsIrToColorAligner | None = None

    def align(
        self,
        *,
        depth_ir_left_m: np.ndarray,
        color_shape: tuple[int, int],
        k_ir_left: np.ndarray,
        t_ir_left_to_color: np.ndarray,
        k_color: np.ndarray,
    ) -> np.ndarray:
        depth_shape = tuple(int(item) for item in depth_ir_left_m.shape)
        key = (
            depth_shape,
            tuple(int(item) for item in color_shape),
            tuple(float(v) for v in np.asarray(k_ir_left, dtype=np.float32).reshape(3, 3).ravel()),
            tuple(float(v) for v in np.asarray(t_ir_left_to_color, dtype=np.float32).reshape(4, 4).ravel()),
            tuple(float(v) for v in np.asarray(k_color, dtype=np.float32).reshape(3, 3).ravel()),
        )
        if self._key != key or self._aligner is None:
            self._aligner = FfsIrToColorAligner(
                k_ir_left=np.asarray(k_ir_left, dtype=np.float32).reshape(3, 3),
                t_ir_left_to_color=np.asarray(t_ir_left_to_color, dtype=np.float32).reshape(4, 4),
                k_color=np.asarray(k_color, dtype=np.float32).reshape(3, 3),
                ir_shape=depth_shape,
                color_shape=tuple(int(item) for item in color_shape),
            )
            self._key = key
        return np.ascontiguousarray(self._aligner.align(depth_ir_left_m), dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Long-lived remote FFS TensorRT depth server for Demo 2.")
    parser.add_argument("--bind", default="tcp://0.0.0.0:7001", help="ZeroMQ REP bind endpoint.")
    parser.add_argument("--ffs-repo", type=Path, default=DEFAULT_FFS_REPO, help="Fast-FoundationStereo repo path.")
    parser.add_argument(
        "--ffs-trt-model-dir",
        type=Path,
        default=DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR,
        help="Two-stage TensorRT FFS engine directory.",
    )
    parser.add_argument("--ffs-trt-root", type=Path, default=None, help="Optional TensorRT runtime root.")
    parser.add_argument("--return", dest="return_type", choices=RETURN_TYPES, default="depth_u16")
    parser.add_argument(
        "--compress",
        choices=COMPRESSION_MODES,
        default="none",
        help="Compress response payloads. png is useful for uint8/uint16 image-like payloads.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Lazy warmup iterations performed with the first real request's IR pair/calibration.",
    )
    parser.add_argument("--echo-only", action="store_true", help="Network/protocol echo mode; returns zero depth without FFS.")
    parser.add_argument("--debug", action="store_true", help="Print per-request timing.")
    parser.add_argument("--strict-engine-contract", action="store_true", help="Fail startup unless the FFS engine path matches the required Demo 2 contract.")
    parser.add_argument("--required-model", default=DEFAULT_FFS_MODEL_NAME)
    parser.add_argument("--required-valid-iters", type=int, default=DEFAULT_FFS_VALID_ITERS)
    parser.add_argument("--required-height", type=int, default=DEFAULT_FFS_TRT_ENGINE_SIZE[0])
    parser.add_argument("--required-width", type=int, default=DEFAULT_FFS_TRT_ENGINE_SIZE[1])
    parser.add_argument("--required-capture-height", type=int, default=DEFAULT_FFS_TRT_INPUT_SIZE[0])
    parser.add_argument("--required-capture-width", type=int, default=DEFAULT_FFS_TRT_INPUT_SIZE[1])
    parser.add_argument("--required-builder-optimization-level", type=int, default=DEFAULT_FFS_TRT_BUILDER_OPTIMIZATION_LEVEL)
    parser.add_argument("--required-max-disp", type=int, default=DEFAULT_FFS_MAX_DISP)
    return parser


def _sparse_uv_depth(depth_m: np.ndarray, mask_u8: np.ndarray | None) -> np.ndarray:
    if mask_u8 is None:
        return np.empty((0, 4), dtype=np.float32)
    if depth_m.shape != mask_u8.shape:
        raise ValueError(f"mask shape {mask_u8.shape} does not match depth shape {depth_m.shape}")
    mask = np.asarray(mask_u8, dtype=np.uint8)
    valid = (mask > 0) & np.isfinite(depth_m) & (depth_m > np.float32(0.0))
    if not np.any(valid):
        return np.empty((0, 4), dtype=np.float32)
    rows, cols = np.nonzero(valid)
    z = depth_m[rows, cols].astype(np.float32, copy=False)
    labels = mask[rows, cols].astype(np.float32, copy=False)
    return np.ascontiguousarray(np.stack([cols.astype(np.float32), rows.astype(np.float32), z, labels], axis=1))


def _sparse_xyz(depth_m: np.ndarray, mask_u8: np.ndarray | None, k_color: np.ndarray) -> np.ndarray:
    uv_depth = _sparse_uv_depth(depth_m, mask_u8)
    if uv_depth.shape[0] == 0:
        return uv_depth
    k = np.asarray(k_color, dtype=np.float32).reshape(3, 3)
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    if fx == 0.0 or fy == 0.0:
        raise ValueError("invalid k_color focal length for masked_xyz")
    u = uv_depth[:, 0]
    v = uv_depth[:, 1]
    z = uv_depth[:, 2]
    label = uv_depth[:, 3]
    x = (u - np.float32(cx)) * z / np.float32(fx)
    y = (v - np.float32(cy)) * z / np.float32(fy)
    return np.ascontiguousarray(np.stack([x, y, z, label, u, v], axis=1), dtype=np.float32)


def _engine_contract_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "ffs_contract_model": str(args.required_model),
        "ffs_contract_valid_iters": int(args.required_valid_iters),
        "ffs_contract_engine_height": int(args.required_height),
        "ffs_contract_engine_width": int(args.required_width),
        "ffs_contract_capture_height": int(args.required_capture_height),
        "ffs_contract_capture_width": int(args.required_capture_width),
        "ffs_contract_builder_optimization_level": int(args.required_builder_optimization_level),
        "ffs_contract_max_disp": int(args.required_max_disp),
        "ffs_contract_padding_policy": (
            f"{int(args.required_capture_height)}x{int(args.required_capture_width)}"
            f"->pad_to_{int(args.required_height)}x{int(args.required_width)}"
        ),
        "ffs_contract_engine_dir": str(Path(args.ffs_trt_model_dir)),
        "ffs_contract_strict": bool(args.strict_engine_contract),
    }


def _validate_engine_contract(args: argparse.Namespace) -> None:
    if int(args.required_height) <= 0 or int(args.required_width) <= 0:
        raise ValueError("required engine dimensions must be positive")
    if int(args.required_capture_height) <= 0 or int(args.required_capture_width) <= 0:
        raise ValueError("required capture dimensions must be positive")
    if int(args.required_valid_iters) <= 0 or int(args.required_max_disp) <= 0:
        raise ValueError("required valid_iters and max_disp must be positive")
    if not bool(args.strict_engine_contract):
        return
    path_text = str(Path(args.ffs_trt_model_dir)).lower().replace("-", "_")
    model_token = str(args.required_model).lower().replace("-", "_")
    expected_res = f"res_{int(args.required_height)}x{int(args.required_width)}"
    checks = {
        f"model {args.required_model}": model_token in path_text,
        f"valid_iters {int(args.required_valid_iters)}": f"iters_{int(args.required_valid_iters)}" in path_text,
        f"engine resolution {int(args.required_height)}x{int(args.required_width)}": expected_res in path_text,
        f"builderOptimizationLevel {int(args.required_builder_optimization_level)}": (
            f"builderopt{int(args.required_builder_optimization_level)}" in path_text
            or f"builder_optimization_level_{int(args.required_builder_optimization_level)}" in path_text
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(
            "strict FFS engine contract failed for "
            f"{Path(args.ffs_trt_model_dir)}: missing {', '.join(failed)}"
        )


def _make_runner(args: argparse.Namespace) -> Any | None:
    if args.echo_only:
        return None
    from data_process.depth_backends import FastFoundationStereoTensorRTRunner

    return FastFoundationStereoTensorRTRunner(
        ffs_repo=Path(args.ffs_repo),
        model_dir=Path(args.ffs_trt_model_dir),
        trt_root=None if args.ffs_trt_root is None else Path(args.ffs_trt_root),
    )


def _run_warmup_if_needed(
    *,
    runner: Any,
    request: Any,
    k_ir_left: np.ndarray,
    baseline_m: float,
    warmup_remaining: int,
    debug: bool,
) -> int:
    count = int(warmup_remaining)
    if count <= 0:
        return 0
    start_s = time.perf_counter()
    for _ in range(count):
        runner.run_pair(
            request.ir_left_u8,
            request.ir_right_u8,
            K_ir_left=k_ir_left,
            baseline_m=baseline_m,
        )
    if debug:
        print(f"[ffs-remote-server] lazy_warmup count={count} elapsed_ms={_elapsed_ms(start_s):.2f}", flush=True)
    return 0


def _handle_request(*, request_parts: list[bytes], runner: Any | None, aligner: _CachedAligner, args: argparse.Namespace) -> list[bytes]:
    request_start_s = time.perf_counter()
    try:
        request = parse_depth_request_parts(request_parts)
        metadata = request.metadata
        frame_id = int(metadata["frame_id"])
        color_shape_value = metadata["color_shape"]
        color_shape = (int(color_shape_value[0]), int(color_shape_value[1]))
        k_ir_left = matrix_from_metadata(metadata, "k_ir_left", shape=(3, 3))
        k_color = matrix_from_metadata(metadata, "k_color", shape=(3, 3))
        t_ir_left_to_color = matrix_from_metadata(metadata, "t_ir_left_to_color", shape=(4, 4))
        baseline_m = float(metadata["baseline_m"])
        depth_scale = float(metadata.get("depth_scale_m_per_unit", 0.001))
        return_type = str(metadata.get("return_type", args.return_type))
        if return_type != args.return_type:
            return_type = args.return_type

        if args.echo_only:
            if return_type in SPARSE_RETURN_TYPES:
                depth_float_m = np.ones(color_shape, dtype=np.float32)
            else:
                depth_float_m = np.zeros(color_shape, dtype=np.float32)
            server_ffs_ms = 0.0
            server_align_ms = 0.0
        else:
            assert runner is not None
            args.warmup = _run_warmup_if_needed(
                runner=runner,
                request=request,
                k_ir_left=k_ir_left,
                baseline_m=baseline_m,
                warmup_remaining=int(args.warmup),
                debug=bool(args.debug),
            )
            ffs_start_s = time.perf_counter()
            output = runner.run_pair(
                request.ir_left_u8,
                request.ir_right_u8,
                K_ir_left=k_ir_left,
                baseline_m=baseline_m,
            )
            ffs_done_s = time.perf_counter()
            depth_ir_left_m = np.asarray(output["depth_ir_left_m"], dtype=np.float32)
            k_ir_left_used = np.asarray(output.get("K_ir_left_used", k_ir_left), dtype=np.float32)
            align_start_s = time.perf_counter()
            depth_float_m = aligner.align(
                depth_ir_left_m=depth_ir_left_m,
                color_shape=color_shape,
                k_ir_left=k_ir_left_used,
                t_ir_left_to_color=t_ir_left_to_color,
                k_color=k_color,
            )
            align_done_s = time.perf_counter()
            server_ffs_ms = _elapsed_ms(ffs_start_s, ffs_done_s)
            server_align_ms = _elapsed_ms(align_start_s, align_done_s)

        if return_type == "depth_u16":
            depth_payload = quantize_depth_with_invalid_zero(depth_float_m, depth_scale)
            dtype = "uint16"
        elif return_type == "depth_float_m":
            depth_payload = np.ascontiguousarray(depth_float_m, dtype=np.float32)
            dtype = "float32"
        elif return_type == "masked_uv_depth":
            depth_payload = _sparse_uv_depth(depth_float_m, request.mask_u8)
            dtype = "float32"
            sparse_format = "uv_depth_label_float32"
        elif return_type == "masked_xyz":
            depth_payload = _sparse_xyz(depth_float_m, request.mask_u8, k_color)
            dtype = "float32"
            sparse_format = "xyz_label_uv_float32"
        else:
            raise ValueError(f"unsupported return_type: {return_type}")
        if return_type not in SPARSE_RETURN_TYPES:
            sparse_format = ""
        server_total_ms = _elapsed_ms(request_start_s)
        if args.debug:
            print(
                "[ffs-remote-server] "
                f"frame_id={frame_id} status=ok return={return_type} "
                f"shape={tuple(depth_payload.shape)} "
                f"ffs_ms={server_ffs_ms:.2f} align_ms={server_align_ms:.2f} total_ms={server_total_ms:.2f}",
                flush=True,
            )
        return build_depth_response_parts(
            frame_id=frame_id,
            depth=depth_payload,
            depth_dtype=dtype,
            status="ok",
            server_ffs_ms=server_ffs_ms,
            server_align_ms=server_align_ms,
            server_total_ms=server_total_ms,
            depth_scale_m_per_unit=depth_scale,
            return_type=return_type,
            compression=args.compress,
            extra_metadata={
                **_engine_contract_metadata(args),
                "sparse_format": sparse_format,
            },
        )
    except Exception as exc:
        frame_id = -1
        try:
            parsed = parse_depth_request_parts(request_parts)
            frame_id = int(parsed.metadata.get("frame_id", -1))
            shape = parsed.metadata.get("color_shape", [1, 1])
            color_shape = (max(1, int(shape[0])), max(1, int(shape[1])))
        except Exception:
            color_shape = (1, 1)
        if args.debug:
            print(f"[ffs-remote-server] frame_id={frame_id} status=error error={type(exc).__name__}: {exc}", flush=True)
        return build_depth_response_parts(
            frame_id=frame_id,
            depth=np.zeros(color_shape, dtype=np.uint16),
            depth_dtype="uint16",
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            server_total_ms=_elapsed_ms(request_start_s),
            return_type=args.return_type,
            compression="none",
            extra_metadata=_engine_contract_metadata(args),
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.compress == "png" and args.return_type != "depth_u16":
        parser.exit(2, "ffs_depth_server.py: error: --compress png currently requires --return depth_u16\n")
    try:
        _validate_engine_contract(args)
    except ValueError as exc:
        parser.exit(2, f"ffs_depth_server.py: error: {exc}\n")
    import zmq

    if not args.echo_only:
        warm_up_numba_ffs_align()
    runner = _make_runner(args)
    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(str(args.bind))
    aligner = _CachedAligner()
    print(
        "[ffs-remote-server] "
        + json.dumps(
            {
                "bind": args.bind,
                "echo_only": bool(args.echo_only),
                "return_type": args.return_type,
                "compress": args.compress,
                "ffs_repo": str(args.ffs_repo),
                "ffs_trt_model_dir": str(args.ffs_trt_model_dir),
                "warmup": int(args.warmup),
                "engine_contract": _engine_contract_metadata(args),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        while True:
            request_parts = socket.recv_multipart()
            response_parts = _handle_request(request_parts=request_parts, runner=runner, aligner=aligner, args=args)
            socket.send_multipart(response_parts)
    except KeyboardInterrupt:
        return 0
    finally:
        socket.close(linger=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
