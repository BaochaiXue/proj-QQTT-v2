"""Launch the Phystwin_shen online full-pipeline supervisor.

Demo v6.2 owns the online stream, runtime parameters, GPU namespace, viewer
ports, and supervisor lifetime. The external wrapper owns the ordered
Stage-1/Stage-2/train execution and its combined HTML viewer child.
"""

from __future__ import annotations

import math
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

FULL_PIPELINE_SCRIPT_RELATIVE = Path("scripts") / "run_online_full_pipeline.py"
PORT_KILL_TERM_TIMEOUT_S = 5.0
PORT_KILL_KILL_TIMEOUT_S = 3.0
PIPELINE_OUTPUT_RELAY_JOIN_TIMEOUT_S = 5.0
PIPELINE_OUTPUT_PREFIX = "[phystwin_shen] "
WILDCARD_LISTENER_HOSTS = frozenset({"", "0.0.0.0", "::"})
ZERO_ORDER_BACKENDS = frozenset({"legacy", "boba"})
SIM_FORCE_MODES = frozenset({"gather", "template_state_batched_atomic"})
STAGE_WINDOW_KEYS = ("batch_size", "segment_len", "segment_stride")

TOP_LEVEL_OVERRIDE_KEYS = (
    "wandb_mode",
    "phys_config",
)
SECTION_OVERRIDE_KEYS = {
    "common": (
        "device",
        "recent_window_count",
        "poll_sec",
        "no_sample_recent",
        "max_online_chunks",
        "seed",
    ),
    "stage1": (
        "max_online_chunks",
        "enabled",
        "script",
        "experiments_dir",
        "max_iter",
        "cma_popsize",
        "zero_order_backend",
        "sim_force_mode",
        "cma_timing",
        "realtime_vis",
        "realtime_vis_dir",
        "realtime_vis_every",
        "no_realtime_iteration_history",
    ),
    "stage2": (
        "max_online_chunks",
        "enabled",
        "script",
        "experiments_dir",
        "max_iter",
        "cma_popsize",
        "zero_order_backend",
        "sim_force_mode",
        "cma_timing",
        "realtime_vis",
        "realtime_vis_dir",
        "realtime_vis_every",
        "no_realtime_iteration_history",
        "stage1_params",
    ),
    "train": (
        "enabled",
        "script",
        "experiments_dir",
        "iterations",
        "train_frame",
        "checkpoint_interval",
        "stop_when_finished",
        "realtime_vis",
        "realtime_vis_every",
        "no_realtime_iteration_history",
        "realtime_vis_dir",
        "zero_order_source",
        "optimal_params_path",
    ),
    "cma_viewer": (
        "enabled",
        "script",
        "source",
        "host",
        "port",
        "cam_idx",
        "point_mode",
        "point_stride",
        "npz_cache_size",
        "quiet",
        "keep_running",
        "base_path",
        "case_name",
        "realtime_dir",
        "rgb_dir",
    ),
    "train_viewer": (
        "enabled",
        "script",
        "host",
        "port",
        "cam_idx",
        "point_mode",
        "point_stride",
        "npz_cache_size",
        "quiet",
        "keep_running",
        "base_path",
        "case_name",
        "realtime_dir",
        "rgb_dir",
    ),
}
OPTIONAL_SECTION_OVERRIDE_KEYS = {
    "common": STAGE_WINDOW_KEYS,
    "stage1": STAGE_WINDOW_KEYS,
    "stage2": STAGE_WINDOW_KEYS,
    "train": STAGE_WINDOW_KEYS,
}


class PhystwinShenLaunchError(RuntimeError):
    """Phystwin_shen could not be validated, launched, or kept healthy."""


def _write_pipeline_console_line(raw_line: bytes) -> None:
    """Write one combined supervisor output line to the Demo terminal."""
    text = raw_line.decode("utf-8", errors="replace")
    sys.stdout.write(f"{PIPELINE_OUTPUT_PREFIX}{text}")
    if text and not text.endswith(("\n", "\r")):
        sys.stdout.write("\n")
    sys.stdout.flush()


class _PipelineOutputRelay:
    """Tee one supervisor's combined output to its log and the Demo terminal."""

    def __init__(
        self,
        *,
        process: subprocess.Popen[bytes],
        pipeline_log: BinaryIO,
    ) -> None:
        self._process = process
        self._pipeline_log = pipeline_log
        self._failure: Exception | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="phystwin-shen-output-relay",
            daemon=True,
        )

    def start(self) -> None:
        """Start draining the supervisor pipe before it can fill."""
        self._thread.start()

    def _run(self) -> None:
        """Copy every output line to the retained log and parent terminal."""
        output: BinaryIO | None = None
        try:
            output = self._process.stdout
            if output is None:
                raise RuntimeError("Phystwin_shen supervisor stdout pipe is missing")
            for raw_line in iter(output.readline, b""):
                self._pipeline_log.write(raw_line)
                self._pipeline_log.flush()
                _write_pipeline_console_line(raw_line)
        except Exception as error:
            self._failure = error
            if self._process.poll() is None:
                self._process.terminate()
        finally:
            if output is not None:
                try:
                    output.close()
                except Exception as error:
                    if self._failure is None:
                        self._failure = error
            try:
                self._pipeline_log.close()
            except Exception as error:
                if self._failure is None:
                    self._failure = error

    def assert_healthy(self) -> None:
        """Raise when output forwarding stopped before the supervisor did."""
        if self._failure is not None:
            raise PhystwinShenLaunchError(
                f"Phystwin_shen console-output relay failed: {self._failure}"
            ) from self._failure

    def finish(self) -> None:
        """Drain the final output after the supervisor process group stops."""
        self._thread.join(timeout=PIPELINE_OUTPUT_RELAY_JOIN_TIMEOUT_S)
        if self._thread.is_alive():
            raise PhystwinShenLaunchError(
                "Phystwin_shen console-output relay did not exit after the "
                "supervisor stopped"
            )
        self.assert_healthy()


def _resolve_repo_file(repo: Path, value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo / path
    path = path.resolve()
    if not path.is_file():
        raise PhystwinShenLaunchError(f"{label} is not a file: {path}")
    return path


def validate_phystwin_shen_repo(
    repo_path: str | Path,
    pipeline_config: str | Path,
) -> tuple[Path, Path]:
    """Return the validated checkout and full-pipeline config paths."""
    repo = Path(repo_path).expanduser().resolve()
    if not repo.is_dir():
        raise PhystwinShenLaunchError(
            f"phystwin_shen repo_path is not a directory: {repo}"
        )
    _resolve_repo_file(
        repo,
        FULL_PIPELINE_SCRIPT_RELATIVE,
        label="phystwin_shen full-pipeline script",
    )
    config = _resolve_repo_file(
        repo,
        pipeline_config,
        label="phystwin_shen full-pipeline config",
    )
    return repo, config


def _resolved_bind_hosts(host: str) -> tuple[set[str], bool]:
    """Resolve a requested bind host into concrete addresses."""
    bind_host = str(host).strip()
    if bind_host in WILDCARD_LISTENER_HOSTS:
        return {bind_host}, True
    try:
        resolved = {
            str(info[4][0])
            for info in socket.getaddrinfo(
                bind_host,
                None,
                type=socket.SOCK_STREAM,
            )
        }
    except socket.gaierror as error:
        raise PhystwinShenLaunchError(
            f"viewer host {host!r} could not be resolved: {error}"
        ) from error
    if not resolved:
        raise PhystwinShenLaunchError(f"viewer host {host!r} resolved to no addresses")
    return resolved, False


def _connection_laddr(conn: Any) -> tuple[str, int] | None:
    """Return ``(host, port)`` for a psutil connection local address."""
    if not conn.laddr:
        return None
    try:
        return str(conn.laddr.ip), int(conn.laddr.port)
    except AttributeError:
        return str(conn.laddr[0]), int(conn.laddr[1])


def _listener_blocks_bind_host(
    listener_host: str,
    *,
    bind_hosts: set[str],
    bind_is_wildcard: bool,
) -> bool:
    if bind_is_wildcard or listener_host in WILDCARD_LISTENER_HOSTS:
        return True
    return listener_host in bind_hosts


def _listening_pids(host: str, port: int) -> list[int | None]:
    """Return PIDs of listeners that block ``host:port``."""
    import psutil  # noqa: PLC0415

    bind_hosts, bind_is_wildcard = _resolved_bind_hosts(host)
    pids: list[int | None] = []
    for conn in psutil.net_connections(kind="tcp"):
        if conn.status != psutil.CONN_LISTEN:
            continue
        local = _connection_laddr(conn)
        if local is None:
            continue
        listener_host, listener_port = local
        if listener_port != int(port):
            continue
        if _listener_blocks_bind_host(
            listener_host,
            bind_hosts=bind_hosts,
            bind_is_wildcard=bind_is_wildcard,
        ):
            pids.append(conn.pid)
    return pids


def ensure_port_free(host: str, port: int) -> dict[str, Any]:
    """Kill listeners blocking one configured HTML viewer endpoint."""
    import psutil  # noqa: PLC0415

    pids = _listening_pids(host, port)
    if not pids:
        return {
            "host": str(host),
            "port": int(port),
            "status": "free",
            "killed_pids": [],
        }
    if any(pid is None for pid in pids):
        raise PhystwinShenLaunchError(
            f"port {port} is in use by a process this user cannot identify or kill"
        )
    killed: list[dict[str, Any]] = []
    for pid in pids:
        try:
            proc = psutil.Process(int(pid))
            name = " ".join(proc.cmdline()) or proc.name()
            proc.terminate()
            try:
                proc.wait(timeout=PORT_KILL_TERM_TIMEOUT_S)
            except psutil.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=PORT_KILL_KILL_TIMEOUT_S)
            killed.append({"pid": int(pid), "command": name})
        except psutil.NoSuchProcess:
            killed.append({"pid": int(pid), "command": "<already exited>"})
        except (psutil.AccessDenied, psutil.TimeoutExpired) as error:
            raise PhystwinShenLaunchError(
                f"failed to kill pid {pid} occupying port {port}: {error}"
            ) from error
    deadline = time.monotonic() + PORT_KILL_KILL_TIMEOUT_S
    while _listening_pids(host, port):
        if time.monotonic() >= deadline:
            raise PhystwinShenLaunchError(
                f"port {port} is still bound after killing "
                f"{[entry['pid'] for entry in killed]}"
            )
        time.sleep(0.05)
    return {
        "host": str(host),
        "port": int(port),
        "status": "killed_occupant",
        "killed_pids": killed,
    }


def _runtime_section(
    runtime_config: Mapping[str, Any], section: str
) -> Mapping[str, Any]:
    value = runtime_config.get(section)
    if not isinstance(value, Mapping):
        raise PhystwinShenLaunchError(
            f"phystwin_shen runtime config {section!r} must be a mapping"
        )
    return value


def _override_text(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


@dataclass(frozen=True)
class PhystwinShenSettings:
    """Validated inputs for one external full-pipeline supervisor."""

    repo_path: Path
    pipeline_config: Path
    conda_env: str
    base_path: Path
    cuda_visible_devices: str
    runtime_config: Mapping[str, Any]

    @property
    def online_dir(self) -> Path:
        return self.base_path / "online_data"

    def viewer_endpoint(self, section: str) -> tuple[str, int] | None:
        viewer = _runtime_section(self.runtime_config, section)
        if not bool(viewer["enabled"]):
            return None
        return str(viewer["host"]), int(viewer["port"])

    @property
    def viewer_urls(self) -> dict[str, str]:
        urls: dict[str, str] = {}
        for section in ("cma_viewer", "train_viewer"):
            endpoint = self.viewer_endpoint(section)
            if endpoint is not None:
                host, port = endpoint
                urls[section] = f"http://{host}:{port}/"
        return urls


def build_full_pipeline_command(
    settings: PhystwinShenSettings,
    *,
    python_prefix: Sequence[str],
) -> list[str]:
    """Build one command with every locally owned value passed explicitly."""
    repo, config = validate_phystwin_shen_repo(
        settings.repo_path,
        settings.pipeline_config,
    )
    command = [
        *python_prefix,
        str(FULL_PIPELINE_SCRIPT_RELATIVE),
        "--config",
        str(config),
        "--online_dir",
        str(settings.online_dir),
        "--cuda_visible_devices",
        str(settings.cuda_visible_devices),
    ]
    for key in TOP_LEVEL_OVERRIDE_KEYS:
        if key not in settings.runtime_config:
            raise PhystwinShenLaunchError(
                f"phystwin_shen runtime config is missing {key!r}"
            )
        command.extend([f"--{key}", _override_text(settings.runtime_config[key])])
    for section, keys in SECTION_OVERRIDE_KEYS.items():
        values = _runtime_section(settings.runtime_config, section)
        missing = [key for key in keys if key not in values]
        if missing:
            raise PhystwinShenLaunchError(
                f"phystwin_shen runtime config {section!r} is missing {missing}"
            )
        for key in keys:
            command.extend([f"--{section}_{key}", _override_text(values[key])])
    for section, keys in OPTIONAL_SECTION_OVERRIDE_KEYS.items():
        values = _runtime_section(settings.runtime_config, section)
        for key in keys:
            if key in values:
                command.extend([f"--{section}_{key}", _override_text(values[key])])
    if repo != settings.repo_path.resolve():
        raise AssertionError("validated repo path changed unexpectedly")
    return command


def _viewer_endpoints(
    settings: PhystwinShenSettings,
) -> list[tuple[str, str, int]]:
    enabled_viewers: list[tuple[str, str, int]] = []
    for section in ("cma_viewer", "train_viewer"):
        endpoint = settings.viewer_endpoint(section)
        if endpoint is None:
            continue
        host, port = endpoint
        enabled_viewers.append((section, host, port))

    if len(enabled_viewers) > 1:
        raise PhystwinShenLaunchError(
            "phystwin_shen supports at most one enabled HTML viewer; use "
            "cma_viewer.source='all' and disable train_viewer"
        )

    for section, host, port in enabled_viewers:
        if not (0 < int(port) < 65536):
            raise PhystwinShenLaunchError(f"{section}.port must be 1..65535")
        _resolved_bind_hosts(host)
    return enabled_viewers


def validate_phystwin_shen_settings(
    settings: PhystwinShenSettings,
    *,
    python_prefix: Sequence[str],
) -> list[str]:
    """Validate local runtime values before starting the camera process."""
    if not str(settings.conda_env).strip():
        raise PhystwinShenLaunchError("phystwin_shen conda_env must be non-empty")
    if not python_prefix or any(not str(item).strip() for item in python_prefix):
        raise PhystwinShenLaunchError(
            "phystwin_shen Python command prefix must be non-empty"
        )
    command = build_full_pipeline_command(
        settings,
        python_prefix=python_prefix,
    )
    common = _runtime_section(settings.runtime_config, "common")
    for key in ("recent_window_count",):
        if int(common[key]) <= 0:
            raise PhystwinShenLaunchError(
                f"phystwin_shen common.{key} must be positive"
            )
    for section in ("common", "stage1", "stage2", "train"):
        values = _runtime_section(settings.runtime_config, section)
        for key in STAGE_WINDOW_KEYS:
            minimum = 2 if key == "segment_len" else 1
            if key in values and int(values[key]) < minimum:
                raise PhystwinShenLaunchError(
                    f"phystwin_shen {section}.{key} must be >= {minimum}"
                )
    for section in ("stage1", "stage2", "train"):
        values = _runtime_section(settings.runtime_config, section)
        if not bool(values["enabled"]):
            continue
        missing = [
            key for key in STAGE_WINDOW_KEYS if key not in values and key not in common
        ]
        if missing:
            raise PhystwinShenLaunchError(
                f"phystwin_shen enabled {section} is missing effective window "
                f"settings {missing}; define them in {section} or common"
            )
    poll_sec = float(common["poll_sec"])
    if not math.isfinite(poll_sec) or poll_sec <= 0.0:
        raise PhystwinShenLaunchError(
            "phystwin_shen common.poll_sec must be finite and positive"
        )
    max_online_chunks = common["max_online_chunks"]
    if max_online_chunks is not None and int(max_online_chunks) <= 0:
        raise PhystwinShenLaunchError(
            "phystwin_shen common.max_online_chunks must be positive or null"
        )
    for section in ("stage1", "stage2"):
        values = _runtime_section(settings.runtime_config, section)
        max_online_chunks = values["max_online_chunks"]
        if max_online_chunks is not None and int(max_online_chunks) <= 0:
            raise PhystwinShenLaunchError(
                f"phystwin_shen {section}.max_online_chunks must be positive or null"
            )
        cma_popsize = values["cma_popsize"]
        if cma_popsize is not None and int(cma_popsize) <= 0:
            raise PhystwinShenLaunchError(
                f"phystwin_shen {section}.cma_popsize must be positive or null"
            )
        zero_order_backend = str(values["zero_order_backend"])
        if zero_order_backend not in ZERO_ORDER_BACKENDS:
            raise PhystwinShenLaunchError(
                f"phystwin_shen {section}.zero_order_backend must be one of "
                f"{sorted(ZERO_ORDER_BACKENDS)}, got {zero_order_backend!r}"
            )
        sim_force_mode = str(values["sim_force_mode"])
        if sim_force_mode not in SIM_FORCE_MODES:
            raise PhystwinShenLaunchError(
                f"phystwin_shen {section}.sim_force_mode must be one of "
                f"{sorted(SIM_FORCE_MODES)}, got {sim_force_mode!r}"
            )
        if bool(values["enabled"]) and int(values["max_iter"]) <= 0:
            raise PhystwinShenLaunchError(
                f"phystwin_shen {section}.max_iter must be positive"
            )
        if bool(values["enabled"]) and int(values["realtime_vis_every"]) <= 0:
            raise PhystwinShenLaunchError(
                f"phystwin_shen {section}.realtime_vis_every must be positive"
            )
    train = _runtime_section(settings.runtime_config, "train")
    if bool(train["enabled"]) and int(train["iterations"]) <= 0:
        raise PhystwinShenLaunchError("phystwin_shen train.iterations must be positive")
    if bool(train["enabled"]) and int(train["realtime_vis_every"]) <= 0:
        raise PhystwinShenLaunchError(
            "phystwin_shen train.realtime_vis_every must be positive"
        )
    for section, _, _ in _viewer_endpoints(settings):
        viewer = _runtime_section(settings.runtime_config, section)
        if int(viewer["cam_idx"]) < 0:
            raise PhystwinShenLaunchError(
                f"phystwin_shen {section}.cam_idx must be non-negative"
            )
        for key in ("point_stride", "npz_cache_size"):
            if int(viewer[key]) <= 0:
                raise PhystwinShenLaunchError(
                    f"phystwin_shen {section}.{key} must be positive"
                )
    return command


def _free_viewer_ports(settings: PhystwinShenSettings) -> dict[str, Any]:
    takeover: dict[str, Any] = {}
    for section, host, port in _viewer_endpoints(settings):
        takeover[section] = ensure_port_free(host, port)
    return takeover


@dataclass
class PhystwinShenLaunch:
    """Handle and provenance for one full-pipeline supervisor."""

    settings: PhystwinShenSettings
    pipeline_process: subprocess.Popen[bytes]
    process_group_id: int
    pipeline_command: list[str]
    pipeline_log_path: Path
    pipeline_output_relay: _PipelineOutputRelay
    port_takeover: dict[str, Any]
    start_wall_s: float
    trigger: str
    summary_extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return_code = self.pipeline_process.poll()
        return {
            "phystwin_shen_started": True,
            "phystwin_shen_trigger": self.trigger,
            "phystwin_shen_repo_path": str(self.settings.repo_path),
            "phystwin_shen_pipeline_config": str(self.settings.pipeline_config),
            "phystwin_shen_conda_env": str(self.settings.conda_env),
            "phystwin_shen_cuda_visible_devices": str(
                self.settings.cuda_visible_devices
            ),
            "phystwin_shen_pipeline_command": list(self.pipeline_command),
            "phystwin_shen_pipeline_log": str(self.pipeline_log_path),
            "phystwin_shen_pipeline_console_output": True,
            "phystwin_shen_pipeline_return_code": return_code,
            "phystwin_shen_pipeline_left_running": return_code is None,
            "phystwin_shen_process_group_id": int(self.process_group_id),
            "phystwin_shen_viewer_urls": self.settings.viewer_urls,
            "phystwin_shen_port_takeover": dict(self.port_takeover),
            "phystwin_shen_start_wall_s": float(self.start_wall_s),
            **self.summary_extra,
        }

    def assert_pipeline_output_relay_healthy(self) -> None:
        """Fail fast when terminal/log forwarding stopped unexpectedly."""
        self.pipeline_output_relay.assert_healthy()

    def finish_pipeline_output_relay(self) -> None:
        """Flush and close the terminal/log relay after process cleanup."""
        self.pipeline_output_relay.finish()


def launch_phystwin_shen(
    settings: PhystwinShenSettings,
    *,
    python_prefix: Sequence[str],
    log_dir: str | Path,
    trigger: str,
    wall_time_origin_s: float,
) -> PhystwinShenLaunch:
    """Free the combined viewer port and start one external supervisor."""
    repo, _ = validate_phystwin_shen_repo(
        settings.repo_path,
        settings.pipeline_config,
    )
    port_takeover = _free_viewer_ports(settings)
    logs = Path(log_dir)
    logs.mkdir(parents=True, exist_ok=True)
    pipeline_log_path = logs / "online_full_pipeline.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(settings.cuda_visible_devices)
    env["PYTHONUNBUFFERED"] = "1"
    pipeline_command = validate_phystwin_shen_settings(
        settings,
        python_prefix=python_prefix,
    )
    pipeline_log = pipeline_log_path.open("wb")
    try:
        pipeline_process = subprocess.Popen(
            pipeline_command,
            cwd=repo,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            bufsize=0,
        )
    except BaseException:
        pipeline_log.close()
        raise
    pipeline_output_relay = _PipelineOutputRelay(
        process=pipeline_process,
        pipeline_log=pipeline_log,
    )
    try:
        pipeline_output_relay.start()
    except BaseException:
        pipeline_log.close()
        pipeline_process.terminate()
        pipeline_process.wait()
        raise
    return PhystwinShenLaunch(
        settings=settings,
        pipeline_process=pipeline_process,
        process_group_id=int(pipeline_process.pid),
        pipeline_command=pipeline_command,
        pipeline_log_path=pipeline_log_path,
        pipeline_output_relay=pipeline_output_relay,
        port_takeover=port_takeover,
        start_wall_s=time.monotonic() - float(wall_time_origin_s),
        trigger=str(trigger),
    )


__all__ = [
    "PhystwinShenLaunch",
    "PhystwinShenLaunchError",
    "PhystwinShenSettings",
    "build_full_pipeline_command",
    "ensure_port_free",
    "launch_phystwin_shen",
    "validate_phystwin_shen_repo",
    "validate_phystwin_shen_settings",
]
