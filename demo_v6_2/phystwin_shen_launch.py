"""Launch Phystwin_shen online training + HTML viewer as demo subprocesses.

``downstream.mode: phystwin_shen`` replaces the manual post-run launch of
``train_online_warp.py`` and ``scripts/html_realtime_viewer.py``: both start
automatically once the shape prior is ready (warmup finished, its GPU freed),
pinned to the shape-prior GPU namespace (``CUDA_VISIBLE_DEVICES`` from
``gpu.phystwin_shen_cuda_visible_devices`` with ``--device cuda:0``), while
``train_online_warp.py`` itself keeps waiting for the first committed chunk.

Both tools read the ``base_path/<case_name>`` case dir (``calibrate.pkl``,
``metadata.json``, ``color/``) that demo v6.1 seeds at capture start and the
``online_data`` chunk stream. The viewer binds ``viewer_host:viewer_port``;
a process already listening there is killed first, and a kill that does not
free the port fails the run fast.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

VIEWER_SCRIPT_RELATIVE = Path("scripts") / "html_realtime_viewer.py"
TRAIN_SCRIPT_RELATIVE = Path("train_online_warp.py")
PORT_KILL_TERM_TIMEOUT_S = 5.0
PORT_KILL_KILL_TIMEOUT_S = 3.0
WILDCARD_LISTENER_HOSTS = frozenset({"", "0.0.0.0", "::"})


class PhystwinShenLaunchError(RuntimeError):
    """Phystwin_shen could not be validated or launched."""


def validate_phystwin_shen_repo(repo_path: str | Path) -> Path:
    """Fail fast unless the checkout provides both entry scripts."""
    repo = Path(repo_path).expanduser()
    if not repo.is_dir():
        raise PhystwinShenLaunchError(
            f"phystwin_shen repo_path is not a directory: {repo}"
        )
    for script in (TRAIN_SCRIPT_RELATIVE, VIEWER_SCRIPT_RELATIVE):
        if not (repo / script).is_file():
            raise PhystwinShenLaunchError(
                f"phystwin_shen repo_path {repo} is missing {script}"
            )
    return repo


def _resolved_bind_hosts(host: str) -> tuple[set[str], bool]:
    """Resolve a requested bind host into concrete addresses.

    A wildcard bind target is special: any existing listener on the same port
    blocks it. For concrete bind targets, wildcard listeners still block that
    target, but listeners on unrelated concrete addresses do not.
    """
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
    """Return whether an existing listener blocks the requested viewer bind."""
    if bind_is_wildcard:
        return True
    if listener_host in WILDCARD_LISTENER_HOSTS:
        return True
    return listener_host in bind_hosts


def _listening_pids(host: str, port: int) -> list[int | None]:
    """PIDs of listeners blocking ``host:port`` (None when unreadable)."""
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
    """Free ``port`` for the HTML viewer, killing any current listener.

    The occupying process gets SIGTERM then SIGKILL; if the listener cannot
    be identified (other user) or the port is still bound afterwards, the
    launch fails fast instead of starting a viewer that would crash with
    EADDRINUSE (html_realtime_viewer.py binds without retry).
    """
    import psutil  # noqa: PLC0415

    pids = _listening_pids(host, port)
    if not pids:
        return {"port": int(port), "status": "free", "killed_pids": []}
    if any(pid is None for pid in pids):
        raise PhystwinShenLaunchError(
            f"port {port} is in use by a process this user cannot identify "
            "or kill; free the port and retry"
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
    # The socket should close with the process; verify before handing the
    # port to the viewer.
    deadline = time.monotonic() + PORT_KILL_KILL_TIMEOUT_S
    while _listening_pids(host, port):
        if time.monotonic() >= deadline:
            raise PhystwinShenLaunchError(
                f"port {port} is still bound after killing "
                f"{[entry['pid'] for entry in killed]}"
            )
        time.sleep(0.05)
    return {"port": int(port), "status": "killed_occupant", "killed_pids": killed}


@dataclass(frozen=True)
class PhystwinShenSettings:
    """Everything needed to launch the two Phystwin_shen subprocesses."""

    repo_path: Path
    conda_env: str
    case_name: str
    base_path: Path
    cuda_visible_devices: str
    viewer_host: str
    viewer_port: int
    viewer_cam_idx: int
    viewer_point_mode: str
    viewer_point_stride: int
    viewer_image_index_mode: str
    train_device: str
    train_batch_size: int
    train_segment_len: int
    train_segment_stride: int
    train_poll_sec: float
    train_recent_window_count: int
    train_realtime_vis_every: int
    train_stop_when_finished: bool

    @property
    def realtime_dir(self) -> Path:
        """Snapshot dir shared by the trainer (writer) and viewer (reader)."""
        return self.repo_path / "experiments_online" / self.case_name / "realtime"

    @property
    def viewer_url(self) -> str:
        return f"http://{self.viewer_host}:{int(self.viewer_port)}/"


def build_viewer_command(
    settings: PhystwinShenSettings, *, python_prefix: Sequence[str]
) -> list[str]:
    """HTML viewer command, mirroring the manual launch script."""
    return [
        *python_prefix,
        str(VIEWER_SCRIPT_RELATIVE),
        "--base_path",
        str(settings.base_path),
        "--case_name",
        str(settings.case_name),
        "--realtime_dir",
        str(settings.realtime_dir),
        "--host",
        str(settings.viewer_host),
        "--port",
        str(int(settings.viewer_port)),
        "--cam_idx",
        str(int(settings.viewer_cam_idx)),
        "--point_mode",
        str(settings.viewer_point_mode),
        "--point_stride",
        str(int(settings.viewer_point_stride)),
        "--image_index_mode",
        str(settings.viewer_image_index_mode),
    ]


def build_train_command(
    settings: PhystwinShenSettings, *, python_prefix: Sequence[str]
) -> list[str]:
    """train_online_warp.py command, mirroring the manual launch script."""
    command = [
        *python_prefix,
        str(TRAIN_SCRIPT_RELATIVE),
        "--base_path",
        str(settings.base_path),
        "--case_name",
        str(settings.case_name),
        "--online_dir",
        str(settings.base_path / "online_data"),
        "--experiments_dir",
        str(settings.repo_path / "experiments_online"),
        "--static_data_path",
        str(settings.base_path / "data" / "final_data.pkl"),
        "--device",
        str(settings.train_device),
        "--batch_size",
        str(int(settings.train_batch_size)),
        "--segment_len",
        str(int(settings.train_segment_len)),
        "--segment_stride",
        str(int(settings.train_segment_stride)),
        "--poll_sec",
        str(float(settings.train_poll_sec)),
        "--recent_window_count",
        str(int(settings.train_recent_window_count)),
        "--realtime_vis",
        "--realtime_vis_dir",
        str(settings.realtime_dir),
        "--realtime_vis_every",
        str(int(settings.train_realtime_vis_every)),
    ]
    if bool(settings.train_stop_when_finished):
        command.append("--stop_when_finished")
    return command


@dataclass
class PhystwinShenLaunch:
    """Handles and provenance for the two launched subprocesses."""

    settings: PhystwinShenSettings
    viewer_process: subprocess.Popen[bytes]
    train_process: subprocess.Popen[bytes]
    viewer_command: list[str]
    train_command: list[str]
    viewer_log_path: Path
    train_log_path: Path
    port_takeover: dict[str, Any]
    start_wall_s: float
    trigger: str
    summary_extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """run_summary.json fields for this launch."""
        viewer_return = self.viewer_process.poll()
        train_return = self.train_process.poll()
        return {
            "phystwin_shen_started": True,
            "phystwin_shen_trigger": self.trigger,
            "phystwin_shen_repo_path": str(self.settings.repo_path),
            "phystwin_shen_conda_env": str(self.settings.conda_env),
            "phystwin_shen_cuda_visible_devices": str(
                self.settings.cuda_visible_devices
            ),
            "phystwin_shen_viewer_url": self.settings.viewer_url,
            "phystwin_shen_viewer_command": list(self.viewer_command),
            "phystwin_shen_train_command": list(self.train_command),
            "phystwin_shen_viewer_log": str(self.viewer_log_path),
            "phystwin_shen_train_log": str(self.train_log_path),
            "phystwin_shen_realtime_dir": str(self.settings.realtime_dir),
            "phystwin_shen_port_takeover": dict(self.port_takeover),
            "phystwin_shen_start_wall_s": float(self.start_wall_s),
            "phystwin_shen_viewer_return_code": viewer_return,
            "phystwin_shen_train_return_code": train_return,
            "phystwin_shen_viewer_left_running": viewer_return is None,
            "phystwin_shen_train_left_running": train_return is None,
            **self.summary_extra,
        }


def launch_phystwin_shen(
    settings: PhystwinShenSettings,
    *,
    python_prefix: Sequence[str],
    log_dir: str | Path,
    trigger: str,
    wall_time_origin_s: float,
) -> PhystwinShenLaunch:
    """Free the viewer port, then start the viewer and trainer subprocesses.

    Both children run with cwd at the Phystwin_shen repo root (their configs
    and imports are repo-relative) inside the configured GPU namespace; logs
    stream to ``log_dir`` under the demo's base path so each run's downstream
    output lives next to its chunks.
    """
    repo = validate_phystwin_shen_repo(settings.repo_path)
    port_takeover = ensure_port_free(settings.viewer_host, settings.viewer_port)
    settings.realtime_dir.mkdir(parents=True, exist_ok=True)
    logs = Path(log_dir)
    logs.mkdir(parents=True, exist_ok=True)
    viewer_log_path = logs / "html_realtime_viewer.log"
    train_log_path = logs / "train_online_warp.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(settings.cuda_visible_devices)
    viewer_command = build_viewer_command(settings, python_prefix=python_prefix)
    train_command = build_train_command(settings, python_prefix=python_prefix)
    with viewer_log_path.open("wb") as viewer_log:
        viewer_process = subprocess.Popen(
            viewer_command,
            cwd=repo,
            env=env,
            stdout=viewer_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        with train_log_path.open("wb") as train_log:
            train_process = subprocess.Popen(
                train_command,
                cwd=repo,
                env=env,
                stdout=train_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except Exception:
        viewer_process.terminate()
        raise
    return PhystwinShenLaunch(
        settings=settings,
        viewer_process=viewer_process,
        train_process=train_process,
        viewer_command=viewer_command,
        train_command=train_command,
        viewer_log_path=viewer_log_path,
        train_log_path=train_log_path,
        port_takeover=port_takeover,
        start_wall_s=time.monotonic() - float(wall_time_origin_s),
        trigger=str(trigger),
    )


__all__ = [
    "PhystwinShenLaunch",
    "PhystwinShenLaunchError",
    "PhystwinShenSettings",
    "build_train_command",
    "build_viewer_command",
    "ensure_port_free",
    "launch_phystwin_shen",
    "validate_phystwin_shen_repo",
]
