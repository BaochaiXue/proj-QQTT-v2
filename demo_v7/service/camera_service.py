#!/usr/bin/env python3
"""demo_v7 camera-service subprocess entry.

Mirrors the demo_v6_2/main_data_processing.py bootstrap (persistent
TORCHINDUCTOR_CACHE_DIR, repo sys.path pin, import-time stamp) and reuses
``demo_v6_2.mdp.cli.build_parser()`` via ``parse_known_args`` for the v6.2
flag subset; only ``--socket-dir``, ``--self-check``,
``--channel-max-hz-json`` and ``--shape-prior-backend`` are v7 flags. The
v6.2 GUI windows are force-disabled — demo_v7's GUI owns every window and the
service streams frames over the frames socket instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Startup-tax visibility: everything between this stamp and the runtime's
# first status emit (module imports, __init__, prewarm spawn) would otherwise
# be invisible (v6.2 main_data_processing mirror).
_PROCESS_START_PERF_S = time.perf_counter()

# torch.compile artifact cache. The torch default lives under /tmp and is
# wiped on reboot, which re-imposes the ~10s cold EdgeTAM precompile on the
# first run after every boot; a persistent dir pays it once per torch/model
# change instead. setdefault keeps an operator override authoritative.
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(Path.home() / ".cache" / "qqtt_torchinductor"),
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR in sys.path:
    sys.path.remove(REPO_ROOT_STR)
sys.path.insert(0, REPO_ROOT_STR)


from demo_v6_2.mdp.cli import (  # noqa: E402
    build_parser as build_v62_parser,
    validate_and_normalize_args,
)


def _build_v7_parser() -> argparse.ArgumentParser:
    """Build the parser for the v7-only flags (v6.2 flags pass through)."""
    parser = argparse.ArgumentParser(
        prog="camera_service",
        description="demo_v7 camera-service subprocess (v6.2 flags pass through).",
        add_help=False,
    )
    parser.add_argument(
        "--socket-dir",
        type=Path,
        default=None,
        help="Directory for control.sock/frames.sock (service binds, GUI connects).",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Construct the runtime with a stub session and exit (no camera/GPU).",
    )
    parser.add_argument(
        "--channel-max-hz-json",
        type=str,
        default=None,
        help=(
            "JSON object {channel: max_hz} with the orchestrator-resolved "
            "preview.channel_max_hz caps (defaults to protocol.CHANNEL_MAX_HZ)."
        ),
    )
    parser.add_argument(
        "--shape-prior-backend",
        type=str,
        default=None,
        help=(
            "Shape-prior generation backend: sam3d (v6.2 default), trellis2, "
            "or none (skip the shape-prior chain entirely)."
        ),
    )
    parser.add_argument(
        "--shape-prior-upscale",
        type=str,
        default=None,
        help=(
            "Upscale (SD x4) stage toggle: on (v6.2 default) or off "
            "(crop-only passthrough; faster warmup)."
        ),
    )
    parser.add_argument(
        "--gaussian-backend",
        type=str,
        default=None,
        help=(
            "Gaussian-splats generator: triposplat (default) or none "
            "(display-only feature off)."
        ),
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        default=None,
        help=(
            "Record every published RGB-D frame into this directory as a "
            "data_collect-format fake-live case (dir must not exist or be "
            "empty)."
        ),
    )
    return parser


def _parse_channel_max_hz(text: str) -> dict[str, float]:
    """Parse + validate the --channel-max-hz-json payload (fail fast)."""
    from demo_v7.ipc import protocol  # noqa: PLC0415

    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("must be a JSON object of {channel: max_hz}")
    channel_max_hz: dict[str, float] = {}
    for channel, hz in loaded.items():
        if channel not in protocol.FRAME_CHANNELS:
            raise ValueError(
                f"unknown channel {channel!r}; expected one of "
                f"{protocol.FRAME_CHANNELS}"
            )
        try:
            hz_value = float(hz)
        except (TypeError, ValueError):
            raise ValueError(
                f"channel {channel!r} max hz must be a number, got {hz!r}"
            ) from None
        if hz_value <= 0.0:
            raise ValueError(f"channel {channel!r} max hz must be positive")
        channel_max_hz[str(channel)] = hz_value
    return channel_max_hz


def _self_check() -> int:
    """Construct StagedRuntime with a stub session and drive the ack path.

    No camera opens, no model loads (track/pcd/tracker all none keeps the
    perception preloader leg-free), no GPU touched; sockets bind in a temp
    dir and the hello / bad-transition / unknown-command acks are asserted.
    """
    import tempfile

    from demo_v6_2.mdp.session import CameraSession
    from demo_v7.ipc import protocol
    from demo_v7.service.staged_runtime import StagedRuntime

    parser = build_v62_parser()
    args, _unknown = parser.parse_known_args(
        [
            "--track-mode",
            "none",
            "--pcd-mode",
            "none",
            "--tracker-backend",
            "none",
            "--no-shape-prior-warmup",
        ]
    )
    args.warmup_rgb_preview = False
    args.live_dataprocess_viewer = False
    with tempfile.TemporaryDirectory(prefix="camera-service-self-check-") as tmp:
        tmp_path = Path(tmp)
        args.headless_capture_dir = tmp_path / "capture"
        runtime = StagedRuntime(
            args,
            socket_dir=tmp_path / "sockets",
            session=CameraSession(),  # stub: never opened
        )
        try:
            hello = runtime._on_command({"cmd": protocol.CMD_HELLO})
            assert hello is not None and hello["ok"], hello
            assert hello["state"] == protocol.STATE_STARTING, hello
            assert hello["version"] == protocol.PROTOCOL_VERSION, hello
            assert hello["cmd"] == protocol.CMD_HELLO, hello
            bad_transition = runtime._on_command(
                {"cmd": protocol.CMD_CAPTURE_FRAME0}
            )
            assert bad_transition is not None and not bad_transition["ok"], (
                bad_transition
            )
            unknown = runtime._on_command({"cmd": "not-a-command"})
            assert unknown is not None and not unknown["ok"], unknown
            assert runtime.state == protocol.STATE_STARTING
        finally:
            runtime.close()
    print("[camera-service] self-check ok", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the command-line entry point."""
    print(
        "[startup] camera service imports took "
        f"{time.perf_counter() - _PROCESS_START_PERF_S:.2f}s",
        flush=True,
    )
    v7_parser = _build_v7_parser()
    v7_args, rest = v7_parser.parse_known_args(argv)
    if v7_args.self_check:
        return _self_check()
    parser = build_v62_parser()
    args, unknown = parser.parse_known_args(rest)
    if unknown:
        # parse_known_args tolerance is for forward-compat orchestrator flags;
        # say what was dropped so a typo never fails silently.
        print(f"[camera-service] ignoring unknown args: {unknown}", flush=True)
    if v7_args.socket_dir is None:
        print("camera_service: error: --socket-dir is required", file=sys.stderr)
        return 2
    # Only pass channel_max_hz through when the orchestrator resolved one so
    # a flagless (direct CLI) run keeps the StagedRuntime protocol default.
    runtime_kwargs: dict[str, object] = {}
    if v7_args.channel_max_hz_json is not None:
        try:
            runtime_kwargs["channel_max_hz"] = _parse_channel_max_hz(
                v7_args.channel_max_hz_json
            )
        except ValueError as exc:
            print(
                f"camera_service: error: --channel-max-hz-json {exc}",
                file=sys.stderr,
            )
            return 2
    # v7 owns every window: the v6.2 warm-up preview and live viewer must
    # stay off regardless of forwarded flags (frames stream over the socket).
    args.warmup_rgb_preview = False
    args.live_dataprocess_viewer = False
    # demo_v7 NEVER uses the canonical-mesh cache (owner rule 2026-08-05):
    # the shape prior is generated live from THIS run's frame 0, every run —
    # no cache read, no cache write. None disables the v6.2 cache entirely
    # (mesh_cache.normalize_object_id semantics), overriding any forwarded
    # --shape-prior-object.
    if args.shape_prior_object is not None:
        print(
            "[camera-service] shape-prior mesh cache DISABLED (demo_v7 "
            f"generates live; ignoring cache id {args.shape_prior_object!r})",
            flush=True,
        )
    args.shape_prior_object = None
    # Shape-prior generate backend (GUI selector). Validate + fail fast here:
    # the trellis2 worker would otherwise die opaquely inside the prewarm
    # pool, and a "none" run must never leave the v6.2 warmup flag on.
    from demo_v7.service import backend_options  # noqa: PLC0415

    try:
        shape_prior_backend = backend_options.normalize_backend(
            v7_args.shape_prior_backend
        )
        if shape_prior_backend == backend_options.BACKEND_TRELLIS2:
            backend_options.ensure_trellis2_available()
        shape_prior_use_upscale = backend_options.normalize_upscale(
            v7_args.shape_prior_upscale
        )
    except (ValueError, FileNotFoundError) as exc:
        print(
            f"camera_service: error: shape-prior options: {exc}", file=sys.stderr
        )
        return 2
    if shape_prior_backend == backend_options.BACKEND_NONE and bool(
        args.shape_prior_warmup
    ):
        print(
            "[camera-service] shape-prior backend 'none': forcing "
            "--no-shape-prior-warmup",
            flush=True,
        )
        args.shape_prior_warmup = False
    if shape_prior_backend != backend_options.BACKEND_NONE and not bool(
        args.shape_prior_warmup
    ):
        # Truthful echo the other way round: with the warmup chain off no
        # generator runs, so the hello ack / GUI must not claim one.
        print(
            f"[camera-service] --no-shape-prior-warmup: backend "
            f"{shape_prior_backend!r} -> 'none' (nothing generates)",
            flush=True,
        )
        shape_prior_backend = backend_options.BACKEND_NONE
    runtime_kwargs["shape_prior_backend"] = shape_prior_backend
    runtime_kwargs["shape_prior_use_upscale"] = shape_prior_use_upscale
    # Gaussian generator (GUI selector). Unknown ids fail fast; a missing
    # TripoSplat install degrades to "none" with a truthful echo — the
    # feature is display-only and must never block a run.
    from demo_v7.service import gaussian_options  # noqa: PLC0415

    try:
        gaussian_backend = gaussian_options.normalize_gaussian_backend(
            v7_args.gaussian_backend
        )
    except ValueError as exc:
        print(f"camera_service: error: gaussian options: {exc}", file=sys.stderr)
        return 2
    if (
        gaussian_backend != gaussian_options.GAUSSIAN_NONE
        and shape_prior_backend == backend_options.BACKEND_NONE
    ):
        print(
            "[camera-service] shape-prior backend 'none': gaussian -> 'none' "
            "(no masked image / no world alignment without the chain)",
            flush=True,
        )
        gaussian_backend = gaussian_options.GAUSSIAN_NONE
    if gaussian_backend == gaussian_options.GAUSSIAN_TRIPOSPLAT:
        try:
            gaussian_options.ensure_triposplat_available()
        except FileNotFoundError as exc:
            print(
                f"[camera-service] TripoSplat unavailable -> gaussian 'none': "
                f"{exc}",
                flush=True,
            )
            gaussian_backend = gaussian_options.GAUSSIAN_NONE
    runtime_kwargs["gaussian_backend"] = gaussian_backend
    runtime_kwargs["record_dir"] = (
        Path(v7_args.record_dir) if v7_args.record_dir else None
    )
    from demo_v7.service.staged_runtime import StagedRuntime  # noqa: PLC0415

    try:
        validate_and_normalize_args(args)
        return StagedRuntime(
            args, socket_dir=Path(v7_args.socket_dir), **runtime_kwargs
        ).run()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        # Startup errors (camera/device selection, arg validation) never reach
        # the worker-thread fatal hook, so surface them on the live status
        # band too (v6.2 main_data_processing mirror).
        from demo_v6_2.pipeline_status import (  # noqa: PLC0415
            STAGE_FATAL,
            PipelineStatusWriter,
        )

        capture_dir = args.headless_capture_dir
        PipelineStatusWriter(
            Path(capture_dir).parent if capture_dir is not None else None,
            "camera",
        ).emit(STAGE_FATAL, f"startup: {exc}", ok=False, exc_type=type(exc).__name__)
        print(f"camera_service: error: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
