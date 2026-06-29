from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import threading
import time
from dataclasses import replace
from typing import Any, Callable

from demo_v5_1.shape_prior import (
    DEFAULT_SHAPE_PRIOR_DEVICE,
    DEFAULT_SHAPE_PRIOR_ENDPOINT,
    DEFAULT_SHAPE_PRIOR_TIMEOUT_MS,
    SHAPE_BACKEND_SAM3D_OBJECTS,
    SHAPE_PRIOR_STATUS_DISABLED,
    SHAPE_PRIOR_STATUS_FAILED,
    SHAPE_PRIOR_STATUS_PENDING,
    SHAPE_PRIOR_STATUS_READY,
    ShapePriorFrame0Request,
    ShapePriorResult,
    pack_shape_prior_request,
    unpack_shape_prior_result,
)


def default_profile(*, enabled: bool) -> dict[str, Any]:
    return {
        "shape_prior_enabled": bool(enabled),
        "shape_prior_status": (
            SHAPE_PRIOR_STATUS_PENDING if enabled else SHAPE_PRIOR_STATUS_DISABLED
        ),
        "shape_backend": SHAPE_BACKEND_SAM3D_OBJECTS if enabled else None,
        "shape_prior_error": None,
    }


class ShapePriorRemoteClient:
    def __init__(self, *, endpoint: str, timeout_ms: int) -> None:
        self.endpoint = str(endpoint)
        self.timeout_ms = int(timeout_ms)
        self._socket: Any | None = None

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = None

    def _connect(self) -> Any:
        if self._socket is not None:
            return self._socket
        import zmq

        socket = zmq.Context.instance().socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.connect(self.endpoint)
        self._socket = socket
        return socket

    def request_shape_prior(self, frame0: ShapePriorFrame0Request) -> ShapePriorResult:
        socket = self._connect()
        start_s = time.perf_counter()
        try:
            socket.send_multipart(pack_shape_prior_request(frame0))
            result = unpack_shape_prior_result(socket.recv_multipart())
        except Exception:
            self.close()
            raise
        metadata = dict(result.metadata)
        metadata.setdefault(
            "response_download_ms",
            (time.perf_counter() - start_s) * 1000.0,
        )
        return replace(
            result,
            source_seq=int(frame0.seq),
            source_timestamp_s=frame0.source_timestamp_s,
            metadata=metadata,
        )


class ShapePriorWarmupManager:
    def __init__(self, *, enabled: bool, client: Any | None) -> None:
        self.enabled = bool(enabled)
        self.client = client
        self.created_perf_s = time.perf_counter()
        self._lock = threading.Lock()
        self._submitted = False
        self._result: ShapePriorResult | None = None
        self._profile = default_profile(enabled=self.enabled)
        self._thread: threading.Thread | None = None

    def maybe_submit(self, frame0: ShapePriorFrame0Request) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            if self._submitted:
                return False
            self._submitted = True
            self._profile.update(
                {
                    "shape_prior_status": SHAPE_PRIOR_STATUS_PENDING,
                    "shape_prior_source_seq": int(frame0.seq),
                    "shape_prior_source_time_s": frame0.source_timestamp_s,
                    "shape_prior_submit_ms": (
                        time.perf_counter() - self.created_perf_s
                    )
                    * 1000.0,
                }
            )
        thread = threading.Thread(target=self._run, args=(frame0,), daemon=True)
        self._thread = thread
        thread.start()
        return True

    def _run(self, frame0: ShapePriorFrame0Request) -> None:
        try:
            if self.client is None:
                raise RuntimeError("shape-prior client is unavailable")
            result = self.client.request_shape_prior(frame0)
            status = SHAPE_PRIOR_STATUS_READY if result.ready else result.status
            with self._lock:
                self._result = result if result.ready else None
                self._profile.update(result.metadata)
                self._profile.update(
                    {
                        "shape_prior_status": status,
                        "shape_prior_ready_seq": int(result.seq),
                        "shape_prior_error": result.error,
                        "time_to_shape_prior_ready_ms": (
                            time.perf_counter() - self.created_perf_s
                        )
                        * 1000.0,
                    }
                )
        except Exception as exc:
            with self._lock:
                self._profile.update(
                    {
                        "shape_prior_status": SHAPE_PRIOR_STATUS_FAILED,
                        "shape_prior_error": str(exc),
                    }
                )

    def wait(self, timeout_s: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout_s)

    def ready_result(self) -> ShapePriorResult | None:
        with self._lock:
            return self._result

    def profile(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._profile)


def prepare_shape_prior_worker_startup(
    worker: Any,
    *,
    preload_models: bool,
) -> dict[str, Any]:
    startup_start_s = time.perf_counter()
    if bool(preload_models):
        worker.preload_models()
        worker._startup_metadata["worker_preloaded_models"] = True
    worker._startup_metadata["worker_ready_ms"] = (
        time.perf_counter() - startup_start_s
    ) * 1000.0
    return worker.startup_metadata()


def python_command_prefix(conda_env: str | None) -> list[str]:
    env_name = "" if conda_env is None else str(conda_env).strip()
    if env_name:
        return ["conda", "run", "-n", env_name, "--no-capture-output", "python"]
    return ["python"]


def _prepend_env_path(env: dict[str, str], key: str, path: Path) -> None:
    value = str(path)
    current = env.get(key, "")
    parts = [item for item in current.split(os.pathsep) if item]
    if value not in parts:
        env[key] = value if not parts else value + os.pathsep + os.pathsep.join(parts)


def _conda_env_prefix(conda_env: str | None) -> Path | None:
    env_name = "" if conda_env is None else str(conda_env).strip()
    if not env_name:
        return None
    candidates: list[Path] = []
    current_prefix = os.environ.get("CONDA_PREFIX")
    if current_prefix:
        current = Path(current_prefix)
        if current.name == env_name:
            candidates.append(current)
        if current.parent.name == "envs":
            candidates.append(current.parent / env_name)
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        exe = Path(conda_exe)
        if len(exe.parents) >= 2:
            candidates.append(exe.parents[1] / "envs" / env_name)
    candidates.append(Path.home() / "miniconda3" / "envs" / env_name)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def apply_shape_prior_worker_cuda_build_env(
    args: argparse.Namespace,
    env: dict[str, str],
) -> None:
    """Expose conda CUDA headers/libs for workers that JIT-compile extensions."""
    prefix = _conda_env_prefix(getattr(args, "shape_prior_worker_conda_env", None))
    if prefix is None:
        return
    include_dir = prefix / "targets" / "x86_64-linux" / "include"
    lib_dir = prefix / "targets" / "x86_64-linux" / "lib"
    if include_dir.is_dir():
        _prepend_env_path(env, "CPATH", include_dir)
        _prepend_env_path(env, "CPLUS_INCLUDE_PATH", include_dir)
    if lib_dir.is_dir():
        _prepend_env_path(env, "LIBRARY_PATH", lib_dir)
        _prepend_env_path(env, "LD_LIBRARY_PATH", lib_dir)
    nvcc = prefix / "bin" / "nvcc"
    env.setdefault("CUDA_HOME", str(prefix))
    if nvcc.is_file():
        env.setdefault("CUDACXX", str(nvcc))
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    env.setdefault("MAX_JOBS", "8")


def build_shape_prior_worker_command(args: argparse.Namespace) -> list[str]:
    """Build the command for the managed remote shape-prior worker."""
    command = [
        *python_command_prefix(getattr(args, "shape_prior_worker_conda_env", None)),
        str(Path("demo_v5_1") / "shape_prior_worker.py"),
        "--bind",
        str(args.shape_prior_endpoint),
        "--device",
        str(args.shape_prior_worker_device),
    ]
    if args.shape_prior_worker_sam3d_root is not None:
        command.extend(["--sam3d-root", str(args.shape_prior_worker_sam3d_root)])
    if args.shape_prior_worker_config is not None:
        command.extend(["--config", str(args.shape_prior_worker_config)])
    if bool(args.shape_prior_worker_preload_models):
        command.append("--preload-models")
    return command


def start_managed_shape_prior_worker(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    resolve_cuda_visible_devices: Callable[[argparse.Namespace], str],
) -> subprocess.Popen[bytes] | None:
    """Start the optional SAM3D worker under Demo v5.1 lifecycle control."""
    if not bool(args.shape_prior_warmup):
        return None
    if str(args.shape_prior_worker_mode) != "managed":
        return None
    command = build_shape_prior_worker_command(args)
    env = os.environ.copy()
    apply_shape_prior_worker_cuda_build_env(args, env)
    cuda_visible_devices = resolve_cuda_visible_devices(args)
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        env=env,
        start_new_session=True,
    )
    grace_s = float(args.shape_prior_worker_startup_grace_s)
    if grace_s > 0.0:
        time.sleep(grace_s)
    if process.poll() is not None:
        raise RuntimeError(
            "managed shape-prior worker exited during startup with code "
            f"{process.returncode}"
        )
    return process
