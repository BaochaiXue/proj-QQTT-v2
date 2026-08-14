#!/usr/bin/env python3
"""Scripted headless end-to-end drive of the demo_v7 camera service.

Runs the whole button flow with no GUI, exactly as the README's 界面流程 but
over the control socket: wait PREVIEW -> capture frame-0 -> confirm -> wait
REVIEW (masks + shape-prior artifacts must exist on disk) -> reposition ->
start formal -> wait for two committed online chunks -> stop -> clean
FINISHED + points.npz. Downstream mode is forced to ``disabled`` so the run
needs no Phystwin_shen checkout.

Requires the real pipeline (GPU + models + a data_collect fake-live case);
it is an integration gate, not a unit test. Prints PASS/FAIL and exits 0/1.

Usage::

    python demo_v7/tests/drive_fake_live.py \
        [--fake-live-case data_collect/...] [--base-path outputs] [...]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from pathlib import Path

# Repo-root bootstrap (same contract as demo_v6_2/main.py) so this script can
# be launched from any working directory.
_REPO_ROOT_STR = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)

from demo_v7.ipc import protocol
from demo_v7.orchestration.session import OrchestratorSession

REQUIRED_ARTIFACT_KINDS = (
    protocol.ARTIFACT_KIND_MASKS,
    protocol.ARTIFACT_KIND_SHAPE_PRIOR,
)
TARGET_CHUNKS = 2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fake-live-case",
        type=Path,
        default=None,
        help="data_collect case; defaults to the demo_v6_2 configured case.",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=None,
        help="Run output dir; defaults to the demo_v6_2 configured base path.",
    )
    parser.add_argument(
        "--shape-prior-backend",
        choices=("sam3d", "trellis2", "none"),
        default=None,
        help=(
            "Generation backend under test; none flips the shape-prior/"
            "points.npz assertions to the skip semantics."
        ),
    )
    parser.add_argument(
        "--shape-prior-upscale",
        choices=("on", "off"),
        default=None,
        help=(
            "Upscale (SD x4) stage toggle under test; off runs the "
            "crop-only passthrough stage (faster warmup)."
        ),
    )
    parser.add_argument(
        "--gaussian-backend",
        choices=("triposplat", "none"),
        default=None,
        help=(
            "Gaussian generator under test (default: session default = "
            "triposplat); none skips the gaussian gates."
        ),
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help=(
            "Tee the run into this data_collect-format directory (recorder "
            "E2E: the produced dir must itself replay as a fake-live case)."
        ),
    )
    parser.add_argument(
        "--formal-after-wrap-s",
        type=float,
        default=None,
        help=(
            "Fake-live 摆位 emulation: wait for the recording to wrap, then "
            "start formal this many seconds in — the recording position "
            "where the scene matches the frame-0 pose again. Needed when the "
            "warmup outlasts the choreography (e.g. trellis2's ~110s: by "
            "then the recorded hands hold a different pose, EdgeTAM's "
            "frame-0-mask seeding mismatches, and chunk-0 controller "
            "selection finds 0 valid candidates). A human operator does the "
            "same by eye with the reposition overlay."
        ),
    )
    # Generous defaults: first-run model preloads and the SAM3D shape-prior
    # chain can each take minutes on a cold machine.
    parser.add_argument(
        "--target-chunks",
        type=int,
        default=TARGET_CHUNKS,
        help="Committed chunks to wait for in FORMAL (longer = more manipulation frames captured).",
    )
    parser.add_argument("--preview-timeout-s", type=float, default=900.0)
    parser.add_argument("--frame0-timeout-s", type=float, default=120.0)
    parser.add_argument("--review-timeout-s", type=float, default=3600.0)
    parser.add_argument("--reposition-timeout-s", type=float, default=120.0)
    parser.add_argument("--formal-timeout-s", type=float, default=300.0)
    parser.add_argument("--chunks-timeout-s", type=float, default=1200.0)
    parser.add_argument("--finish-timeout-s", type=float, default=600.0)
    return parser


class DriveObserver:
    """Collects events from the session's IPC threads for the main script."""

    def __init__(self, start_s: float) -> None:
        self._start_s = start_s
        self._lock = threading.Lock()
        self.artifacts: dict[str, dict[str, str]] = {}
        self.artifact_events: dict[str, int] = {}
        self.failed_acks: list[dict] = []
        self.replay_exhausted = False
        self.gaussian_frames = 0
        self.first_gaussian_payload: bytes | None = None
        self.last_gaussian_payload: bytes | None = None
        # Wall time (monotonic) of the newest pre-formal wrap event; the
        # drive uses it to start formal at a chosen recording position.
        self.replay_wrapped_at_s: float | None = None

    def log(self, message: str) -> None:
        print(f"[drive +{time.monotonic() - self._start_s:8.1f}s] {message}", flush=True)

    def on_event(self, event: dict) -> None:
        kind = event.get("event")
        if kind == protocol.EVT_STATE:
            self.log(f"state -> {event.get('state')} {event.get('detail', '')}")
        elif kind == protocol.EVT_PROGRESS:
            ok = "ok" if event.get("ok", True) else "FAILED"
            self.log(
                f"progress [{event.get('stage')}] {event.get('detail', '')} ({ok})"
            )
        elif kind == protocol.EVT_ARTIFACTS:
            paths = {
                str(name): str(path)
                for name, path in dict(event.get("paths") or {}).items()
            }
            with self._lock:
                kind_key = str(event.get("kind"))
                self.artifacts.setdefault(kind_key, {}).update(paths)
                self.artifact_events[kind_key] = (
                    self.artifact_events.get(kind_key, 0) + 1
                )
            self.log(f"artifacts [{event.get('kind')}]: {sorted(paths)}")
        elif kind == protocol.EVT_ACK:
            if not event.get("ok", False):
                with self._lock:
                    self.failed_acks.append(dict(event))
                self.log(f"ACK FAILED: {event}")
            else:
                self.log(f"ack {event.get('cmd')}")
        elif kind == protocol.EVT_ERROR:
            self.log(f"service error [{event.get('where')}]: {event.get('message')}")
        elif kind == protocol.EVT_REPLAY_EXHAUSTED:
            self.replay_exhausted = True
            if event.get("wrapped", False):
                with self._lock:
                    self.replay_wrapped_at_s = time.monotonic()
            self.log("replay exhausted")
        elif kind == protocol.EVT_FORMAL_STATS:
            pass  # periodic; too chatty for the drive log
        else:
            self.log(f"event: {event}")

    def on_frame(self, header, payload) -> None:
        if header.channel == protocol.CH_GAUSSIAN:
            with self._lock:
                self.gaussian_frames += 1
                if self.first_gaussian_payload is None:
                    self.first_gaussian_payload = bytes(payload)
                self.last_gaussian_payload = bytes(payload)

    def artifact_event_count(self, kind: str) -> int:
        with self._lock:
            return self.artifact_events.get(kind, 0)

    def check_acks(self) -> None:
        with self._lock:
            if self.failed_acks:
                raise RuntimeError(f"command ack failed: {self.failed_acks[0]}")

    def artifact_kinds(self) -> set[str]:
        with self._lock:
            return set(self.artifacts)

    def artifact_paths(self, kind: str) -> dict[str, str]:
        with self._lock:
            return dict(self.artifacts.get(kind, {}))


def wait_for_artifacts(
    observer: DriveObserver,
    *,
    timeout_s: float,
    required: tuple[str, ...] = REQUIRED_ARTIFACT_KINDS,
) -> None:
    """Artifacts events can land moments after the REVIEW state flips."""
    deadline = time.monotonic() + timeout_s
    while True:
        missing = set(required) - observer.artifact_kinds()
        if not missing:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"missing artifact events after REVIEW: {sorted(missing)}"
            )
        time.sleep(0.5)
    for kind in required:
        paths = observer.artifact_paths(kind)
        if not paths:
            raise RuntimeError(f"artifact kind {kind!r} reported no paths")
        for name, path in paths.items():
            if not Path(path).exists():
                raise RuntimeError(
                    f"artifact {kind}/{name} missing on disk: {path}"
                )


def wait_for_chunks(
    session: OrchestratorSession,
    observer: DriveObserver,
    *,
    target: int,
    timeout_s: float,
) -> int:
    """Poll online_data/chunks until ``target`` chunk pickles are committed."""
    deadline = time.monotonic() + timeout_s
    last_reported = -1
    while True:
        observer.check_acks()
        if session.chunk_error is not None:
            raise RuntimeError(f"chunk stream failed: {session.chunk_error!r}")
        count = len(sorted(session.online_chunks_dir.glob("chunk_*.pkl")))
        if count != last_reported:
            observer.log(f"committed online chunks: {count}/{target}")
            last_reported = count
        if count >= target:
            return count
        if session.service_state == protocol.STATE_FATAL:
            raise RuntimeError("camera service went fatal while streaming chunks")
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"timed out after {timeout_s:.0f}s waiting for {target} chunks "
                f"(have {count})"
            )
        time.sleep(1.0)


def drive(args: argparse.Namespace) -> None:
    start_s = time.monotonic()
    observer = DriveObserver(start_s)
    prior_disabled = args.shape_prior_backend == "none"
    session = OrchestratorSession(
        source="fake-live",
        fake_live_case=args.fake_live_case,
        base_path=args.base_path,
        downstream_mode="disabled",
        shape_prior_backend=args.shape_prior_backend,
        shape_prior_upscale=args.shape_prior_upscale,
        record_dir=args.record_dir,
        gaussian_backend=args.gaussian_backend,
        on_event=observer.on_event,
        on_frame=observer.on_frame,
    )
    observer.log(f"base_path={session.base_path} socket_dir={session.socket_dir}")
    try:
        observer.log("stage 1/8: starting camera service")
        session.start()

        observer.log("stage 2/8: waiting for PREVIEW (source + preloads)")
        session.wait_for_state(
            protocol.STATE_PREVIEW, timeout_s=args.preview_timeout_s
        )
        observer.check_acks()

        observer.log("stage 3/8: capturing frame-0")
        session.send_command({"cmd": protocol.CMD_CAPTURE_FRAME0})
        session.wait_for_state(
            protocol.STATE_FRAME0_PENDING, timeout_s=args.frame0_timeout_s
        )
        observer.check_acks()

        observer.log("stage 4/8: confirming frame-0, waiting for REVIEW "
                     "(SAM3.1 masks + shape-prior chain)")
        session.send_command({"cmd": protocol.CMD_CONFIRM_FRAME0})
        session.wait_for_state(
            protocol.STATE_REVIEW, timeout_s=args.review_timeout_s
        )
        observer.check_acks()
        if prior_disabled:
            # backend none: no shape-prior artifacts by design; the observed
            # frame-0 points npz (frame0 kind) is the review payload.
            wait_for_artifacts(
                observer,
                timeout_s=30.0,
                required=(protocol.ARTIFACT_KIND_MASKS,),
            )
            if protocol.ARTIFACT_KIND_SHAPE_PRIOR in observer.artifact_kinds():
                raise RuntimeError(
                    "backend none must not emit shape_prior artifacts"
                )
        else:
            wait_for_artifacts(observer, timeout_s=30.0)
        observer.log("review artifacts verified on disk "
                     f"({', '.join(sorted(observer.artifact_kinds()))})")

        gaussian_on = (
            not prior_disabled
            and args.gaussian_backend != "none"
            and __import__("os").environ.get("DEMO_V7_GAUSSIAN_SPLATS", "1") != "0"
        )
        if gaussian_on:
            observer.log("stage 4b: waiting for gaussian generation + alignment")
            deadline = time.monotonic() + 420.0
            while observer.artifact_event_count(protocol.ARTIFACT_KIND_GAUSSIAN) < 1:
                observer.check_acks()
                if time.monotonic() >= deadline:
                    raise RuntimeError("timed out waiting for gaussian artifacts")
                time.sleep(1.0)
            wait_for_artifacts(
                observer,
                timeout_s=10.0,
                required=(protocol.ARTIFACT_KIND_GAUSSIAN,),
            )
            observer.log("stage 4c: 换seed re-roll (seed=123)")
            session.send_command(
                {"cmd": protocol.CMD_REGEN_GAUSSIAN, "seed": 123}
            )
            deadline = time.monotonic() + 420.0
            while observer.artifact_event_count(protocol.ARTIFACT_KIND_GAUSSIAN) < 2:
                observer.check_acks()
                if time.monotonic() >= deadline:
                    raise RuntimeError("timed out waiting for the re-rolled gaussian")
                time.sleep(1.0)
            observer.log("gaussian re-roll verified")

        observer.log("stage 5/8: entering reposition")
        session.send_command({"cmd": protocol.CMD_BEGIN_REPOSITION})
        session.wait_for_state(
            protocol.STATE_REPOSITION, timeout_s=args.reposition_timeout_s
        )
        observer.check_acks()

        if args.formal_after_wrap_s is not None:
            observer.log(
                "stage 6/8 (摆位 emulation): waiting for a replay wrap, then "
                f"+{args.formal_after_wrap_s:.0f}s to re-match the frame-0 pose"
            )
            wrap_deadline = time.monotonic() + args.chunks_timeout_s
            while observer.replay_wrapped_at_s is None:
                observer.check_acks()
                if session.service_state == protocol.STATE_FATAL:
                    raise RuntimeError("camera service went fatal pre-formal")
                if time.monotonic() >= wrap_deadline:
                    raise RuntimeError("timed out waiting for a replay wrap")
                time.sleep(0.5)
            target = observer.replay_wrapped_at_s + args.formal_after_wrap_s
            while time.monotonic() < target:
                observer.check_acks()
                time.sleep(0.2)

        observer.log("stage 6/8: starting formal tracking")
        session.send_command({"cmd": protocol.CMD_START_FORMAL})
        session.wait_for_state(
            protocol.STATE_FORMAL, timeout_s=args.formal_timeout_s
        )
        observer.check_acks()

        target_chunks = int(args.target_chunks)
        observer.log(f"stage 7/8: waiting for {target_chunks} committed chunks "
                     f"under {session.online_chunks_dir}")
        wait_for_chunks(
            session,
            observer,
            target=target_chunks,
            timeout_s=args.chunks_timeout_s,
        )

        observer.log("stage 8/8: stopping formal, waiting for clean FINISHED")
        if session.service_state == protocol.STATE_FORMAL:
            session.send_command({"cmd": protocol.CMD_STOP_FORMAL})
        session.wait_for_state(
            protocol.STATE_FINISHED, timeout_s=args.finish_timeout_s
        )
        observer.check_acks()
        manifests = session.wait_chunk_stream(timeout_s=args.finish_timeout_s)
        if len(manifests) < target_chunks:
            raise RuntimeError(
                f"chunk stream drained with only {len(manifests)} manifests"
            )
        if gaussian_on:
            if observer.gaussian_frames < 1:
                raise RuntimeError(
                    "no CH_GAUSSIAN frames were published during FORMAL"
                )
            observer.log(
                f"gaussian live channel verified ({observer.gaussian_frames} frames)"
            )
            check_gaussian_follow(session, observer, args.gaussian_backend)
        if prior_disabled:
            # No shape prior -> ChunkStreamSession never writes the
            # downstream-trigger points.npz (require_shape_prior=False).
            if session.points_npz_path.is_file():
                raise RuntimeError(
                    "backend none must not write points.npz: "
                    f"{session.points_npz_path}"
                )
            observer.log(
                f"clean finish: {len(manifests)} chunks, observed-only "
                "(no points.npz by design)"
            )
        else:
            if not session.points_npz_path.is_file():
                raise RuntimeError(
                    f"points.npz missing after chunk 0: {session.points_npz_path}"
                )
            observer.log(
                f"clean finish: {len(manifests)} chunks, points.npz at "
                f"{session.points_npz_path}"
            )
    finally:
        observer.log("shutting down session")
        session.shutdown()


def check_gaussian_follow(
    session, observer: DriveObserver, gaussian_backend: str | None
) -> None:
    """Assert the live gaussian actually FOLLOWED the tracked object.

    Surface: <base_path>/gaussian/gaussian_live_stats.json, written
    periodically by the service's gaussian worker (its stdout is not
    capturable here). bone2splat p50 is the follow metric: bones ride the
    real object surface, so a stuck or misaligned gaussian shows up as a
    large median bone->nearest-splat distance. Frame dumps land next to it
    for eyeball checks.
    """
    gaussian_dir = session.base_path / "gaussian"
    for name, payload in (
        ("live_first_frame.jpg", observer.first_gaussian_payload),
        ("live_last_frame.jpg", observer.last_gaussian_payload),
    ):
        if payload:
            (gaussian_dir / name).write_bytes(payload)
    stats_path = gaussian_dir / "gaussian_live_stats.json"
    if not stats_path.is_file():
        raise RuntimeError(
            f"gaussian follow stats missing: {stats_path} (formal ran "
            f"{observer.gaussian_frames} frames but the worker never "
            "reported — deform likely failed)"
        )
    stats = json.loads(stats_path.read_text())
    observer.log(f"gaussian follow stats: {stats}")
    if stats.get("failed", False):
        raise RuntimeError(f"gaussian live deform/render failed mid-run: {stats}")
    if not stats.get("rest_seeded", False):
        raise RuntimeError(
            "gaussian bones were not seeded from the formal seq-0 rest "
            f"pose: {stats}"
        )
    p50 = float(stats.get("bone2splat_p50_cm", 1e9))
    if p50 > 6.0:
        raise RuntimeError(
            f"gaussian is not following the tracked object: bone->splat "
            f"p50 {p50}cm (limit 6.0): {stats}"
        )
    bones_moved = float(stats.get("bones_moved_cm", 0.0))
    splats_moved = float(stats.get("splats_moved_cm", 0.0))
    if bones_moved > 2.0 and splats_moved <= 0.0:
        raise RuntimeError(
            f"tracked bones moved {bones_moved}cm but the splats never "
            f"moved: {stats}"
        )
    observer.log("gaussian follow verified")
    check_self_align_attempt(gaussian_dir, observer)


def check_self_align_attempt(gaussian_dir, observer: DriveObserver) -> None:
    """The background self-align upgrade must have run and recorded its
    decision (swap or keep — both fine; silence is a regression)."""
    import os

    if os.environ.get("DEMO_V7_GAUSSIAN_SELF_ALIGN", "1") == "0":
        return
    provenance_path = gaussian_dir / "gaussian_provenance.json"
    if not provenance_path.is_file():
        raise RuntimeError(f"gaussian provenance missing: {provenance_path}")
    alignment = json.loads(provenance_path.read_text()).get("alignment", {})
    record = alignment.get("self_align")
    if not record:
        raise RuntimeError(
            "self-align upgrade never recorded a decision in provenance "
            f"(method={alignment.get('method')})"
        )
    observer.log(
        f"self-align verified: {record.get('decision')} "
        f"(improvement {record.get('improvement')}, "
        f"scores {record.get('scores')})"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        drive(args)
    except BaseException as error:
        traceback.print_exc()
        print(f"FAIL: {error}", flush=True)
        return 1
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
