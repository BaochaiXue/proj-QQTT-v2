"""demo_v7 parent-process orchestration (no Qt imports allowed here).

``OrchestratorSession`` owns everything demo_v6_2/main.py did on the parent
side of the process boundary, retargeted at the button-driven v7 camera
service:

- run output-dir preparation + ``pipeline_status.jsonl`` run lifecycle
  (reusing ``prepare_realtime_output_for_new_run`` / ``PipelineStatusWriter``);
- spawning ``demo_v7/service/camera_service.py`` with the exact demo_v6_2
  camera argv (reusing ``build_main_data_processing_command``, swapping the
  script target and appending the v7-only flags ``--socket-dir`` /
  ``--channel-max-hz-json`` / ``--shape-prior-backend``) and the same env
  resolution (SAM3.1 checkpoint env, ``CUDA_VISIBLE_DEVICES``);
- connect-with-retry ``ControlClient``/``FrameStreamClient`` bridging;
- one background ``ChunkStreamSession`` started when the service reports
  ``STATE_FORMAL``, wired identically to demo_v6_2/main.py:279-299 including
  ``points_npz`` and the Phystwin_shen downstream trigger.

Every callback (``on_event``/``on_frame``/``on_chunk_written``) fires on an
IPC or chunk-stream thread, never the caller's thread; a Qt GUI must hop to
its main thread itself (signals). The v6.2 ``demo_visualizer`` downstream
mode is rejected: the v7 GUI is the live viewer.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence

# Keep this repo at the front of the import path when launched from another
# working directory (same bootstrap contract as demo_v6_2/main.py); must run
# before any demo_v6_2/demo_v7 import below.
_REPO_ROOT_STR = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)

import yaml

from demo_v6_2.main_cli import build_parser
from demo_v6_2.main_options import load_optional_points, python_command_prefix
from demo_v6_2.main_subprocess import (
    build_main_data_processing_command,
    default_capture_dir,
    stop_process,
)
from demo_v6_2.orchestration.main_config import (
    DEFAULT_SAM31_CHECKPOINT_PATH,
    REPO_ROOT,
    SAM31_CHECKPOINT_ENV,
)
from demo_v6_2.orchestration.main_layout import (
    prepare_realtime_output_for_new_run,
    resolve_shape_prior_points_npz,
)
from demo_v6_2.orchestration.run_config import OrchestratorRunConfig
from demo_v6_2.phystwin_shen_launch import (
    PhystwinShenLaunch,
    PhystwinShenLaunchError,
    launch_phystwin_shen,
)
from demo_v6_2.pipeline_status import (
    STAGE_CHUNK_COMMITTED,
    STAGE_DOWNSTREAM_START,
    STAGE_FATAL,
    STAGE_RUN_FINISHED,
    STAGE_RUN_START,
    PipelineStatusWriter,
)
from demo_v6_2.streaming.session import ChunkStreamSession
from demo_v7.ipc import protocol
from demo_v7.ipc.channel import ControlClient, FrameStreamClient
from demo_v7.service import backend_options, gaussian_options

V7_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
CAMERA_SERVICE_SCRIPT = Path("demo_v7") / "service" / "camera_service.py"


# ---------------------------------------------------------------------------
# v7 config (v7-only keys; every pipeline knob stays in demo_v6_2's config)
# ---------------------------------------------------------------------------


def load_v7_config(path: str | Path = V7_CONFIG_PATH) -> dict[str, Any]:
    """Load demo_v7-only defaults from YAML (shared with the service side)."""
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"demo_v7 config must be a mapping: {path}")
    for section in ("session", "preview"):
        if not isinstance(loaded.get(section), dict):
            raise ValueError(f"demo_v7 config section {section!r} must be a mapping")
    return dict(loaded)


def resolve_preview_channel_max_hz(config: dict[str, Any]) -> dict[str, float]:
    """Merge preview.channel_max_hz overrides onto the protocol defaults."""
    merged = dict(protocol.CHANNEL_MAX_HZ)
    overrides = config["preview"].get("channel_max_hz") or {}
    if not isinstance(overrides, dict):
        raise ValueError("preview.channel_max_hz must be a mapping")
    for channel, hz in overrides.items():
        if channel not in protocol.FRAME_CHANNELS:
            raise ValueError(
                f"unknown preview channel {channel!r}; expected one of "
                f"{protocol.FRAME_CHANNELS}"
            )
        if float(hz) <= 0.0:
            raise ValueError(f"preview.channel_max_hz[{channel!r}] must be positive")
        merged[str(channel)] = float(hz)
    return merged


def resolve_socket_dir(config: dict[str, Any], base_path: Path) -> Path:
    """Expand the socket-dir template; fail fast on over-long UDS paths."""
    template = str(config["session"]["socket_dir_template"])
    socket_dir = Path(template.format(base_path=str(base_path)))
    longest = socket_dir / protocol.FRAMES_SOCKET_NAME
    # sockaddr_un.sun_path is 108 bytes on Linux (incl. NUL terminator).
    if len(str(longest).encode()) > 100:
        raise ValueError(
            f"socket dir path too long for a Unix socket: {socket_dir} "
            "(shorten --base-path or session.socket_dir_template)"
        )
    return socket_dir


def case_table_calibrate_snapshot(case_dir: str | Path | None) -> Path | None:
    """The case's record-time table calibration, if it carries one.

    Only the real table-calibration snapshot pair counts
    (``table_calibrate.pkl`` + ``table_calibrate_metadata.json``, schema
    qqtt_table_calibration_v1) — the legacy per-case ``calibrate.pkl`` is a
    different pipeline with an undefined world frame and must never be fed
    to ``--table-calibrate``.
    """
    if case_dir is None:
        return None
    snapshot = Path(case_dir) / "table_calibrate.pkl"
    sidecar = snapshot.with_name("table_calibrate_metadata.json")
    if snapshot.is_file() and sidecar.is_file():
        return snapshot
    return None


# ---------------------------------------------------------------------------
# Link-death-aware IPC clients
# ---------------------------------------------------------------------------


class _WatchedControlClient(ControlClient):
    """ControlClient that reports reader-loop death not caused by ``close()``.

    The base reader thread exits silently on EOF/OSError (e.g. the service
    dropped the connection); ``on_dead`` fires exactly once, on the dying
    reader thread, so the session can reconnect mid-run. A send failure is
    covered too: the kernel tears the connection down for both directions,
    so the reader sees EOF moments later.
    """

    def __init__(
        self,
        socket_path: Path,
        *,
        on_event: Callable[[dict], None],
        on_dead: Callable[[], None],
    ) -> None:
        self._on_dead = on_dead
        super().__init__(socket_path, on_event=on_event)

    def _reader_loop(self) -> None:
        try:
            super()._reader_loop()
        finally:
            if not self._closed.is_set():
                self._on_dead()


class _WatchedFrameStreamClient(FrameStreamClient):
    """FrameStreamClient with the same death notification as the control link."""

    def __init__(
        self,
        socket_path: Path,
        *,
        on_frame: Callable[[protocol.FrameHeader, bytes], None],
        on_dead: Callable[[], None],
    ) -> None:
        self._on_dead = on_dead
        super().__init__(socket_path, on_frame=on_frame)

    def _reader_loop(self) -> None:
        try:
            super()._reader_loop()
        finally:
            if not self._closed.is_set():
                self._on_dead()


# ---------------------------------------------------------------------------
# Orchestrator session
# ---------------------------------------------------------------------------


class OrchestratorSession:
    """One GUI-side run: spawn the camera service, bridge IPC, stream chunks.

    ``__init__`` only resolves configuration (cheap, validates fail-fast);
    ``start()`` prepares the output dir, spawns the service, and connects the
    control/frame clients. ``shutdown()`` is idempotent and thread-safe.
    """

    def __init__(
        self,
        *,
        source: str | None = None,
        fake_live_case: str | Path | None = None,
        base_path: str | Path | None = None,
        downstream_mode: str | None = None,
        shape_prior_backend: str | None = None,
        shape_prior_upscale: bool | str | None = None,
        gaussian_backend: str | None = None,
        record_dir: str | Path | None = None,
        extra_v62_argv: Sequence[str] = (),
        v7_config_path: str | Path = V7_CONFIG_PATH,
        on_event: Callable[[dict], None] | None = None,
        on_frame: Callable[[protocol.FrameHeader, bytes], None] | None = None,
        on_chunk_written: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._v7_config = load_v7_config(v7_config_path)
        session_cfg = self._v7_config["session"]
        resolved_source = str(source if source is not None else session_cfg["source"])
        if fake_live_case is None:
            fake_live_case = session_cfg.get("fake_live_case")
        # Shape-prior generate backend (GUI selector; sam3d/trellis2/none).
        # Resolved before the strict v6.2 parse because backend "none" maps
        # onto existing v6.2 switches: the shape-prior chain off, ASAP off
        # (it hard-requires a mesh), and — unless the caller explicitly chose
        # a downstream — PhysTwin off too (phystwin_shen rejects no-ASAP runs
        # at parse time by design).
        self.shape_prior_backend = backend_options.normalize_backend(
            shape_prior_backend
            if shape_prior_backend is not None
            else session_cfg.get("shape_prior_backend")
        )
        if self.shape_prior_backend == backend_options.BACKEND_TRELLIS2:
            # GUI-side fail-fast: without this the missing-install error only
            # lands in the service's stderr and the operator sees a generic
            # connect timeout.
            backend_options.ensure_trellis2_available()
        # 上采样 toggle (GUI selector; off = crop-only passthrough stage).
        self.shape_prior_upscale = backend_options.normalize_upscale(
            shape_prior_upscale
            if shape_prior_upscale is not None
            else session_cfg.get("shape_prior_upscale")
        )
        # Gaussian generator (GUI selector; triposplat/none). Depends on the
        # shape-prior chain for its input image + world alignment, so backend
        # "none" forces it off; availability degrading (missing checkout)
        # happens service-side — display-only must never block a start.
        self.gaussian_backend = gaussian_options.normalize_gaussian_backend(
            gaussian_backend
            if gaussian_backend is not None
            else session_cfg.get("gaussian_backend")
        )
        if self.shape_prior_backend == backend_options.BACKEND_NONE:
            self.gaussian_backend = gaussian_options.GAUSSIAN_NONE
        # 录制 option: tee the whole run into a data_collect-format fake-live
        # case. GUI-side fail-fast on a dirty target so the operator sees the
        # error before the service spawns; relative paths resolve against the
        # repo root (same convention as --fake-live-case).
        self.record_dir: Path | None = None
        if record_dir is not None:
            resolved_record = Path(record_dir).expanduser()
            if not resolved_record.is_absolute():
                resolved_record = REPO_ROOT / resolved_record
            if resolved_record.exists():
                try:
                    dirty = not resolved_record.is_dir() or any(
                        resolved_record.iterdir()
                    )
                except OSError:
                    dirty = True
                if dirty:
                    raise ValueError(
                        f"record dir is not an empty directory: "
                        f"{resolved_record} — pick a fresh directory"
                    )
            self.record_dir = resolved_record
        if (
            self.shape_prior_backend == backend_options.BACKEND_NONE
            and downstream_mode == "phystwin_shen"
        ):
            raise ValueError(
                "shape-prior backend 'none' 关闭 ASAP(无 mesh 可用),无法与 "
                "downstream phystwin_shen 共存(run_config 会在解析期拒绝)"
            )
        argv: list[str] = ["--input-source", resolved_source]
        if resolved_source == "fake-live" and fake_live_case is not None:
            argv.extend(["--fake-live-case", str(fake_live_case)])
        if base_path is not None:
            argv.extend(["--base-path", str(base_path)])
        if self.shape_prior_backend == backend_options.BACKEND_NONE:
            argv.extend(["--no-shape-prior-warmup", "--no-asap-augment"])
            if downstream_mode is None:
                downstream_mode = "disabled"
        if downstream_mode is not None:
            argv.extend(["--downstream-mode", str(downstream_mode)])
        # The v7 GUI replaces every v6.2 window: the camera service must not
        # open the warmup preview or the in-process live viewer.
        argv.extend(["--no-warmup-rgb-preview", "--no-live-dataprocess-viewer"])
        argv.extend(str(item) for item in extra_v62_argv)
        self._args = build_parser().parse_args(argv)
        self._config = OrchestratorRunConfig.from_args(self._args)
        if self._config.demo_visualizer_enabled:
            raise ValueError(
                "downstream mode demo_visualizer is not supported by demo_v7: "
                "the GUI is the live viewer (use disabled or phystwin_shen)"
            )
        self.base_path = Path(self._args.base_path)
        self.capture_dir = default_capture_dir(self._args, self.base_path)
        self.points_npz_path = resolve_shape_prior_points_npz(self._args)
        self.online_chunks_dir = self.base_path / "online_data" / "chunks"
        self.socket_dir = resolve_socket_dir(self._v7_config, self.base_path)
        self.preview_channel_max_hz = resolve_preview_channel_max_hz(self._v7_config)
        self._connect_timeout_s = float(session_cfg["connect_timeout_s"])
        self._connect_poll_interval_s = float(session_cfg["connect_poll_interval_s"])
        self._on_event = on_event
        self._on_frame = on_frame
        self._on_chunk_written_cb = on_chunk_written
        # Mutable run state; _state_cond guards state/artifacts/manifests so
        # wait_for_state() works from any thread.
        self._state_cond = threading.Condition()
        self._service_state: str | None = None
        self._last_error_event: dict[str, Any] | None = None
        self._chunk_manifests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._phystwin_lock = threading.Lock()
        # Latch (guarded by _phystwin_lock): once _stop_phystwin ran, the
        # chunk thread's before_poll trigger must never relaunch PhysTwin.
        self._phystwin_stopped = False
        self._started = False
        self._shutdown_done = False
        self._service: subprocess.Popen[bytes] | None = None
        self._control: ControlClient | None = None
        self._frames: FrameStreamClient | None = None
        # Sockets currently being re-dialed after a mid-run disconnect
        # (guarded by _lock so each link has at most one reconnect thread).
        self._reconnecting: set[str] = set()
        self._chunk_thread: threading.Thread | None = None
        self._chunk_error: BaseException | None = None
        self._capture_finished_event = threading.Event()
        self._phystwin_launch: PhystwinShenLaunch | None = None
        self._status: PipelineStatusWriter | None = None
        self._run_start_monotonic_s = 0.0
        self._camera_service_command: list[str] | None = None

    # -- callback wiring ----------------------------------------------------

    def set_on_event(self, callback: Callable[[dict], None] | None) -> None:
        """Replace the event callback (fires on the control reader thread)."""
        self._on_event = callback

    def set_on_frame(
        self, callback: Callable[[protocol.FrameHeader, bytes], None] | None
    ) -> None:
        """Replace the frame callback (fires on the frame reader thread)."""
        self._on_frame = callback

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Prepare outputs, spawn the camera service, connect both sockets."""
        with self._lock:
            if self._started:
                return
            self._started = True
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.startup_output_cleanup = prepare_realtime_output_for_new_run(
            self.base_path,
            legacy_case_prefix=str(self._args.case_prefix),
        )
        self._run_start_monotonic_s = time.monotonic()
        self._status = PipelineStatusWriter(self.base_path, "orchestrator")
        self._status.emit(
            STAGE_RUN_START,
            f"input={self._args.input_source} "
            f"downstream={self._config.downstream_mode}",
        )
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self._prepare_socket_dir()
        profile_json = (
            Path(self._args.shape_prior_profile_json)
            if self._args.shape_prior_profile_json is not None
            else self.capture_dir / "shape_prior_profile.json"
        )
        command = build_main_data_processing_command(
            self._args,
            capture_dir=self.capture_dir,
            profile_json=profile_json,
        )
        # Retarget the v6.2 camera argv at the v7 button-driven service:
        # command[0] is "python", command[1] the v6.2 script path. The parent
        # may be a GUI launched outside a shell, so the bare "python" is
        # replaced with this interpreter (same conda env either way).
        command[0] = sys.executable
        command[1] = str(CAMERA_SERVICE_SCRIPT)
        command.extend(["--socket-dir", str(self.socket_dir)])
        # Forward the resolved preview caps so preview.channel_max_hz config
        # overrides actually reach the service-side frame publisher.
        command.extend(
            ["--channel-max-hz-json", json.dumps(self.preview_channel_max_hz)]
        )
        command.extend(["--shape-prior-backend", self.shape_prior_backend])
        command.extend(
            ["--shape-prior-upscale", "on" if self.shape_prior_upscale else "off"]
        )
        command.extend(["--gaussian-backend", self.gaussian_backend])
        if self.record_dir is not None:
            command.extend(["--record-dir", str(self.record_dir)])
        # fake-live: a case that carries its record-time table_calibrate
        # snapshot (v7 recorder writes one since 2026-08-07) replays with
        # THAT c2w — recalibrating the camera later must not shift old
        # recordings into a wrong world frame. Snapshot-less legacy cases
        # keep the repo-root default the v6.2 builder put in the argv.
        if str(self._args.input_source) == "fake-live":
            snapshot = case_table_calibrate_snapshot(self._args.fake_live_case)
            if snapshot is not None:
                calibrate_index = command.index("--table-calibrate")
                command[calibrate_index + 1] = str(snapshot)
                print(
                    "[v7-session] fake-live uses the case's own table "
                    f"calibration snapshot: {snapshot}",
                    flush=True,
                )
        self._camera_service_command = command
        try:
            self._service = subprocess.Popen(
                command,
                env=self._camera_service_env(),
                cwd=str(REPO_ROOT),
                start_new_session=True,
            )
            self._connect_clients()
            self.send_command({"cmd": protocol.CMD_HELLO})
        except BaseException as error:
            self._status.emit(STAGE_FATAL, str(error), ok=False)
            self.shutdown()
            raise

    def _prepare_socket_dir(self) -> None:
        """Create the socket dir and drop stale socket files from a prior run."""
        self.socket_dir.mkdir(parents=True, exist_ok=True)
        for name in (protocol.CONTROL_SOCKET_NAME, protocol.FRAMES_SOCKET_NAME):
            stale = self.socket_dir / name
            if stale.exists():
                stale.unlink()

    def _camera_service_env(self) -> dict[str, str]:
        """Mirror demo_v6_2/main.py:154-175 env resolution for the service."""
        env = os.environ.copy()
        from demo_v6_2.perception.sam31_image_segmentation import (  # noqa: PLC0415
            QQTT_SAM31_CHECKPOINT_ENV,
        )

        assert SAM31_CHECKPOINT_ENV == QQTT_SAM31_CHECKPOINT_ENV, (
            "sam31 checkpoint env var name diverged: config/default.yaml "
            "paths.sam31_checkpoint_env vs "
            "demo_v6_2/perception/sam31_image_segmentation.py "
            "QQTT_SAM31_CHECKPOINT_ENV"
        )
        if not env.get(SAM31_CHECKPOINT_ENV):
            checkpoint_path = Path(DEFAULT_SAM31_CHECKPOINT_PATH).expanduser()
            if not checkpoint_path.is_absolute():
                checkpoint_path = REPO_ROOT / checkpoint_path
            env[SAM31_CHECKPOINT_ENV] = str(checkpoint_path)
        env["CUDA_VISIBLE_DEVICES"] = (
            self._config.main_data_processing_cuda_visible_devices
        )
        # SAM3D overwrites CUDA_HOME with CONDA_PREFIX at import; demo_v7
        # generates the shape prior live EVERY run, so the nvdiffrast JIT
        # rebuild must find nvcc no matter how the GUI was launched. A shell
        # without an activated env (or with one whose prefix lacks nvcc)
        # would fail the generate stage with 'code=127' build errors — known
        # env landmine; /usr/local/cuda is the documented working fallback.
        conda_prefix = env.get("CONDA_PREFIX", "")
        if not (conda_prefix and (Path(conda_prefix) / "bin" / "nvcc").is_file()):
            fallback = Path("/usr/local/cuda")
            if (fallback / "bin" / "nvcc").is_file():
                env["CONDA_PREFIX"] = str(fallback)
                print(
                    "[v7-session] CONDA_PREFIX lacks nvcc; using "
                    f"{fallback} for the SAM3D generate stage",
                    flush=True,
                )
        # Second documented generate landmine: torch >= 2.11's
        # _import_module_from_library no longer registers JIT extensions in
        # sys.modules, so nvdiffrast's mesh postprocess dies with
        # ModuleNotFoundError('nvdiffrast_plugin') even after a successful
        # build. Putting the BUILT plugin dir(s) on PYTHONPATH lets the plain
        # import find the .so. (A truly fresh machine still pays one failed
        # run before the plugin exists; rerun after the first build.)
        plugin_dirs = [
            str(path)
            for path in sorted(
                Path.home().glob(".cache/torch_extensions/*/nvdiffrast_plugin")
            )
            if (path / "nvdiffrast_plugin.so").is_file()
        ]
        if plugin_dirs:
            existing = env.get("PYTHONPATH", "")
            parts = [p for p in existing.split(os.pathsep) if p] + [
                p for p in plugin_dirs if p not in existing
            ]
            env["PYTHONPATH"] = os.pathsep.join(parts)
        return env

    def _connect_clients(self) -> None:
        """Connect control + frame sockets, retrying until the service binds."""
        deadline = time.monotonic() + self._connect_timeout_s
        while True:
            service = self._service
            if service is not None and service.poll() is not None:
                raise RuntimeError(
                    "camera service exited during startup "
                    f"(return code {service.returncode}); command: "
                    f"{self._camera_service_command}"
                )
            try:
                if self._control is None:
                    self._control = self._dial_control()
                if self._frames is None:
                    self._frames = self._dial_frames()
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "camera service sockets did not accept a connection "
                        f"within {self._connect_timeout_s:.0f}s: {self.socket_dir}"
                    ) from None
                time.sleep(self._connect_poll_interval_s)

    def _dial_control(self) -> ControlClient:
        return _WatchedControlClient(
            self.socket_dir / protocol.CONTROL_SOCKET_NAME,
            on_event=self._handle_event,
            on_dead=lambda: self._schedule_reconnect("control"),
        )

    def _dial_frames(self) -> FrameStreamClient:
        return _WatchedFrameStreamClient(
            self.socket_dir / protocol.FRAMES_SOCKET_NAME,
            on_frame=self._handle_frame,
            on_dead=lambda: self._schedule_reconnect("frames"),
        )

    # -- IPC link resilience ------------------------------------------------

    def _schedule_reconnect(self, which: str) -> None:
        """Reader-death handler: re-dial off the dying reader thread.

        At most one reconnect thread per link; a no-op once shutdown began
        (shutdown closes the clients itself and close() suppresses on_dead).
        """
        with self._lock:
            if self._shutdown_done or which in self._reconnecting:
                return
            self._reconnecting.add(which)
        threading.Thread(
            target=self._reconnect_client,
            args=(which,),
            name=f"v7-{which}-reconnect",
            daemon=True,
        ).start()

    def _reconnect_client(self, which: str) -> None:
        """Re-dial one socket with retry + re-hello while the service lives.

        Retries are bounded twice over: the loop only spins while the
        service process is alive (a dead SERVICE surfaces as an error event,
        never a reconnect loop) and gives up after connect_timeout_s even if
        the process lingers without re-binding its sockets.
        """
        try:
            deadline = time.monotonic() + self._connect_timeout_s
            while True:
                with self._lock:
                    if self._shutdown_done:
                        return
                service = self._service
                if service is None or service.poll() is not None:
                    self._note_link_error(
                        which,
                        f"{which} socket disconnected and the camera service "
                        "is not running; not reconnecting (return code "
                        f"{None if service is None else service.returncode})",
                    )
                    return
                if time.monotonic() >= deadline:
                    self._note_link_error(
                        which,
                        f"{which} socket reconnect timed out after "
                        f"{self._connect_timeout_s:.0f}s: {self.socket_dir}",
                    )
                    return
                try:
                    client: ControlClient | FrameStreamClient = (
                        self._dial_control()
                        if which == "control"
                        else self._dial_frames()
                    )
                except OSError:
                    time.sleep(self._connect_poll_interval_s)
                    continue
                old: ControlClient | FrameStreamClient | None = None
                installed = False
                with self._lock:
                    if not self._shutdown_done:
                        installed = True
                        if which == "control":
                            old, self._control = self._control, client
                        else:
                            old, self._frames = self._frames, client
                if not installed:
                    client.close()
                    return
                if old is not None and old is not client:
                    try:
                        old.close()
                    except Exception:
                        pass
                if isinstance(client, ControlClient):
                    # Re-hello: the ack carries the current state, so the
                    # session (and GUI) re-syncs after the gap.
                    client.send_command({"cmd": protocol.CMD_HELLO})
                return
        finally:
            with self._lock:
                self._reconnecting.discard(which)

    def _note_link_error(self, which: str, message: str) -> None:
        """Surface a dead link as an error event (stored + forwarded)."""
        event = {
            "event": protocol.EVT_ERROR,
            "where": f"{which}_link",
            "message": str(message),
        }
        with self._state_cond:
            self._last_error_event = dict(event)
            self._state_cond.notify_all()
        callback = self._on_event
        if callback is not None:
            try:
                callback(event)
            except Exception:
                traceback.print_exc()

    # -- IPC dispatch -------------------------------------------------------

    def send_command(self, cmd: dict[str, Any]) -> None:
        """Send one control command (fire-and-forget; acks arrive as events)."""
        control = self._control
        if control is None:
            raise RuntimeError("control channel is not connected")
        control.send_command(cmd)

    def _handle_event(self, event: dict[str, Any]) -> None:
        kind = event.get("event")
        if kind == protocol.EVT_STATE:
            self._note_state(str(event.get("state")))
        elif (
            kind == protocol.EVT_ACK
            and event.get("cmd") == protocol.CMD_HELLO
            and event.get("ok")
            and event.get("state")
        ):
            # The hello ack carries the current state so a client connecting
            # after transitions (or reconnecting) still syncs.
            self._note_state(str(event.get("state")))
        elif kind == protocol.EVT_ERROR:
            with self._state_cond:
                self._last_error_event = dict(event)
        callback = self._on_event
        if callback is not None:
            try:
                callback(event)
            except Exception:
                traceback.print_exc()

    def _note_state(self, state: str) -> None:
        with self._state_cond:
            self._service_state = state
            self._state_cond.notify_all()
        if state == protocol.STATE_FORMAL:
            self._start_chunk_thread()
        if state in (protocol.STATE_FINISHED, protocol.STATE_FATAL):
            # No more frames.jsonl rows will be appended: let the chunk
            # stream drain its tail and return.
            self._capture_finished_event.set()

    def _handle_frame(self, header: protocol.FrameHeader, payload: bytes) -> None:
        callback = self._on_frame
        if callback is not None:
            try:
                callback(header, payload)
            except Exception:
                traceback.print_exc()

    # -- observable state ---------------------------------------------------

    @property
    def service_state(self) -> str | None:
        """Last state reported by the camera service (None before hello)."""
        with self._state_cond:
            return self._service_state

    @property
    def chunk_manifests(self) -> list[dict[str, Any]]:
        """Committed online-chunk manifests so far (copy; thread-safe)."""
        with self._state_cond:
            return list(self._chunk_manifests)

    @property
    def chunk_error(self) -> BaseException | None:
        """Error that stopped the chunk-stream thread, if any."""
        return self._chunk_error

    def wait_for_state(
        self, states: str | Sequence[str], *, timeout_s: float
    ) -> str:
        """Block until the service reports one of ``states``.

        Raises RuntimeError on service death, STATE_FATAL (unless requested),
        or a chunk-stream failure; TimeoutError on deadline.
        """
        targets = {states} if isinstance(states, str) else set(states)
        deadline = time.monotonic() + float(timeout_s)
        with self._state_cond:
            while True:
                if self._service_state in targets:
                    return str(self._service_state)
                if self._chunk_error is not None:
                    raise RuntimeError(
                        f"chunk stream failed: {self._chunk_error!r}"
                    ) from self._chunk_error
                if (
                    self._service_state == protocol.STATE_FATAL
                    and protocol.STATE_FATAL not in targets
                ):
                    raise RuntimeError(
                        f"camera service reported fatal: {self._last_error_event}"
                    )
                service = self._service
                if service is not None and service.poll() is not None:
                    raise RuntimeError(
                        "camera service exited while waiting for state(s) "
                        f"{sorted(targets)} (return code {service.returncode})"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"timed out after {timeout_s:.0f}s waiting for state(s) "
                        f"{sorted(targets)}; last state: {self._service_state}"
                    )
                self._state_cond.wait(timeout=min(remaining, 1.0))

    # -- chunk streaming (started at STATE_FORMAL) --------------------------

    def _start_chunk_thread(self) -> None:
        with self._lock:
            if self._chunk_thread is not None or self._shutdown_done:
                return
            self._chunk_thread = threading.Thread(
                target=self._run_chunk_stream,
                name="v7-chunk-stream",
                daemon=True,
            )
            self._chunk_thread.start()

    def _capture_finished(self) -> bool:
        if self._capture_finished_event.is_set():
            return True
        service = self._service
        return service is not None and service.poll() is not None

    def _run_chunk_stream(self) -> None:
        """Mirror the demo_v6_2/main.py:279-299 ChunkStreamSession wiring."""
        args = self._args
        config = self._config
        try:
            chunk_stream = ChunkStreamSession(
                self.capture_dir,
                base_path=self.base_path,
                case_prefix=str(args.case_prefix),
                chunk_size=config.chunk_frame_count,
                volume_sample_size_m=config.volume_sample_size_m,
                fps=int(round(float(args.replay_fps))),
                max_chunks=args.max_chunks,
                capture_finished=self._capture_finished,
                require_shape_prior=bool(args.shape_prior_warmup),
                points_npz=self.points_npz_path,
                shape_prior_wait_timeout_s=float(
                    args.shape_prior_chunk_wait_timeout_s
                ),
                poll_interval_s=float(args.chunk_poll_interval_s),
                surface_points=load_optional_points(args.surface_points_npy),
                interior_points=load_optional_points(args.interior_points_npy),
                on_chunk_written=self._note_chunk_written,
                before_poll=self._ensure_phystwin_shen_running,
                asap_augment=bool(args.asap_augment),
                asap_mesh_path=args.asap_mesh_path,
            )
            chunk_stream.run()
        except BaseException as error:  # noqa: BLE001 - surfaced to callers
            self._chunk_error = error
            if self._status is not None:
                self._status.emit(STAGE_FATAL, str(error), ok=False)
            traceback.print_exc()
            with self._state_cond:
                self._state_cond.notify_all()

    def _note_chunk_written(self, manifest: dict[str, Any]) -> None:
        """demo_v6_2/main.py on_chunk_written minus the window visualizer."""
        with self._state_cond:
            self._chunk_manifests.append(dict(manifest))
            self._state_cond.notify_all()
        if self._status is not None:
            self._status.emit(
                STAGE_CHUNK_COMMITTED,
                f"chunk {manifest.get('chunk_index', manifest.get('chunk_name', '?'))}",
                chunk_index=manifest.get("chunk_index"),
                track_status=manifest.get("track_process_status"),
            )
        # Safety net: a chunk can only commit after the shape prior is ready.
        self._ensure_phystwin_shen_running()
        callback = self._on_chunk_written_cb
        if callback is not None:
            try:
                callback(manifest)
            except Exception:
                traceback.print_exc()

    def wait_chunk_stream(self, *, timeout_s: float) -> list[dict[str, Any]]:
        """Join the chunk thread; re-raise its failure; return manifests."""
        thread = self._chunk_thread
        if thread is not None:
            thread.join(timeout=float(timeout_s))
            if thread.is_alive():
                raise TimeoutError(
                    f"chunk stream did not finish within {timeout_s:.0f}s"
                )
        if self._chunk_error is not None:
            raise RuntimeError(
                f"chunk stream failed: {self._chunk_error!r}"
            ) from self._chunk_error
        return self.chunk_manifests

    # -- downstream (phystwin_shen) trigger ---------------------------------

    def _ensure_phystwin_shen_running(self) -> None:
        """demo_v6_2/main.py ``_ensure_phystwin_shen_running`` semantics.

        Launch once when points.npz (the chunk-0 unified sampling product)
        exists, then enforce live health on every poll. No-op when the
        downstream mode is disabled.
        """
        if not self._config.phystwin_shen_enabled:
            return
        with self._phystwin_lock:
            if self._phystwin_stopped:
                return
            launch = self._phystwin_launch
            if launch is not None:
                launch.assert_pipeline_output_relay_healthy()
                return_code = launch.pipeline_process.poll()
                if return_code not in (None, 0):
                    raise PhystwinShenLaunchError(
                        "Phystwin_shen full pipeline exited before the Demo "
                        f"stream finished (return code {return_code})"
                    )
                return
            if bool(self._args.shape_prior_warmup):
                if not self.points_npz_path.is_file():
                    return
                trigger = "shape_prior_points_ready"
            else:
                trigger = "warmup_disabled_immediate"
            launch = launch_phystwin_shen(
                self._config.phystwin_shen_settings,
                python_prefix=python_command_prefix(
                    self._args.phystwin_shen_conda_env
                ),
                log_dir=self.base_path / "phystwin_shen",
                trigger=trigger,
                wall_time_origin_s=self._run_start_monotonic_s,
            )
            self._phystwin_launch = launch
            print(
                "[demo_v7] phystwin_shen started "
                f"({trigger}); viewers: {launch.settings.viewer_urls}"
            )
            if self._status is not None:
                self._status.emit(STAGE_DOWNSTREAM_START, f"phystwin_shen ({trigger})")

    def _stop_phystwin(self) -> None:
        with self._phystwin_lock:
            self._phystwin_stopped = True
            launch = self._phystwin_launch
            self._phystwin_launch = None
        if launch is None:
            return
        try:
            stop_process(
                launch.pipeline_process,
                process_group_id=launch.process_group_id,
            )
        finally:
            launch.finish_pipeline_output_relay()

    # -- shutdown -----------------------------------------------------------

    def shutdown(self, *, chunk_join_timeout_s: float = 600.0) -> None:
        """Tear the whole run down; idempotent and callable from any thread.

        Order matters: ask the service to exit (so the headless capture
        writer finalizes and frames.jsonl closes cleanly), then let the
        chunk stream drain its tail, then close the IPC clients and stop the
        downstream pipeline.
        """
        with self._lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True
        control = self._control
        if control is not None:
            try:
                control.send_command({"cmd": protocol.CMD_SHUTDOWN})
            except Exception:
                pass
        service = self._service
        if service is not None:
            # The service's CMD_SHUTDOWN contract drains + finalizes an
            # in-flight FORMAL run before exiting (StagedRuntime bounds that
            # work with a 120s deadline); SIGTERM at 15s would truncate the
            # phystwin-strict product mid-finalize. Idle states exit fast.
            drain_s = (
                130.0 if self.service_state == protocol.STATE_FORMAL else 15.0
            )
            try:
                service.wait(timeout=drain_s)
            except subprocess.TimeoutExpired:
                pass
            stop_process(service)
        self._capture_finished_event.set()
        thread = self._chunk_thread
        if thread is not None:
            thread.join(timeout=float(chunk_join_timeout_s))
        for client in (self._control, self._frames):
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
        self._control = None
        self._frames = None
        self._stop_phystwin()
        if self._status is not None:
            failed = self._chunk_error is not None
            self._status.emit(
                STAGE_FATAL if failed else STAGE_RUN_FINISHED,
                str(self._chunk_error) if failed else "shutdown",
                ok=not failed,
                chunk_count=len(self.chunk_manifests),
            )
