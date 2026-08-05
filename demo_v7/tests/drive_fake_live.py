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
    # Generous defaults: first-run model preloads and the SAM3D shape-prior
    # chain can each take minutes on a cold machine.
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
        self.failed_acks: list[dict] = []
        self.replay_exhausted = False

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
                self.artifacts.setdefault(str(event.get("kind")), {}).update(paths)
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
            self.log("replay exhausted")
        elif kind == protocol.EVT_FORMAL_STATS:
            pass  # periodic; too chatty for the drive log
        else:
            self.log(f"event: {event}")

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


def wait_for_artifacts(observer: DriveObserver, *, timeout_s: float) -> None:
    """Artifacts events can land moments after the REVIEW state flips."""
    deadline = time.monotonic() + timeout_s
    while True:
        missing = set(REQUIRED_ARTIFACT_KINDS) - observer.artifact_kinds()
        if not missing:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"missing artifact events after REVIEW: {sorted(missing)}"
            )
        time.sleep(0.5)
    for kind in REQUIRED_ARTIFACT_KINDS:
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
    session = OrchestratorSession(
        source="fake-live",
        fake_live_case=args.fake_live_case,
        base_path=args.base_path,
        downstream_mode="disabled",
        on_event=observer.on_event,
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
        wait_for_artifacts(observer, timeout_s=30.0)
        observer.log("review artifacts verified on disk "
                     f"({', '.join(sorted(observer.artifact_kinds()))})")

        observer.log("stage 5/8: entering reposition")
        session.send_command({"cmd": protocol.CMD_BEGIN_REPOSITION})
        session.wait_for_state(
            protocol.STATE_REPOSITION, timeout_s=args.reposition_timeout_s
        )
        observer.check_acks()

        observer.log("stage 6/8: starting formal tracking")
        session.send_command({"cmd": protocol.CMD_START_FORMAL})
        session.wait_for_state(
            protocol.STATE_FORMAL, timeout_s=args.formal_timeout_s
        )
        observer.check_acks()

        observer.log(f"stage 7/8: waiting for {TARGET_CHUNKS} committed chunks "
                     f"under {session.online_chunks_dir}")
        wait_for_chunks(
            session,
            observer,
            target=TARGET_CHUNKS,
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
        if len(manifests) < TARGET_CHUNKS:
            raise RuntimeError(
                f"chunk stream drained with only {len(manifests)} manifests"
            )
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
