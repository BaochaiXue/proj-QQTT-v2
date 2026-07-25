"""Camera-free perception preloads started before the camera opens.

Every model a frame-0 consumer needs — the EdgeTAM streaming runtime
(segmentation), the SAM3.1 frame-0 seeder, and the TAPNext++ tracker
checkpoint — loads on its own daemon thread at ``run()`` entry, while the
camera is still closed. The live capture worker holds frame-0 designation
behind ``wait_frame0_consumers_ready`` so the operator's hold-still window
opens only when segmentation and tracking can consume frame 0 immediately.

Failure semantics mirror the old inline loads exactly: each stage worker
joins its own leg and the leg's exception re-raises there — the same worker
and fatal-latch route as before, just earlier. A failed leg still counts as
"done" for the readiness barrier; the consuming worker then re-raises and the
fatal latch sets ``stop_event``, which the barrier loop also watches.
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from demo_v6_2.mdp import warmup
from demo_v6_2.mdp.cli import RunMode
from demo_v6_2.mdp.constants import (
    DEFAULT_EDGETAM_COMPILE_MODE,
    DEFAULT_EDGETAM_MODEL_ID,
)
from demo_v6_2.utils.concurrency import (
    HEAVY_IMPORT_LOCK,
    elapsed_ms as _elapsed_ms,
)


@dataclass(frozen=True)
class EdgetamRuntime:
    """Loaded EdgeTAM streaming runtime plus its load-timing breakdown."""

    hf_stream: Any
    torch_module: Any
    dtype: Any
    model: Any
    processor: Any
    timing_ms: dict[str, float]


def load_edgetam_runtime(args: argparse.Namespace) -> EdgetamRuntime:
    """Import, load, and compile-wrap the EdgeTAM model (no camera inputs)."""
    from demo_v6_2.mdp.segmentation import _load_hf_streaming_runtime

    init_start_s = time.perf_counter()
    runtime_load_start_s = time.perf_counter()
    with HEAVY_IMPORT_LOCK:
        hf_stream = _load_hf_streaming_runtime()
    torch_module = hf_stream.torch
    if (
        str(args.device).startswith("cuda")
        and not torch_module.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA device requested but torch.cuda.is_available() is false"
        )
    dtype = hf_stream._dtype_from_name(args.dtype)
    runtime_load_end_s = time.perf_counter()
    model_load_start_s = time.perf_counter()
    model = hf_stream.EdgeTamVideoModel.from_pretrained(DEFAULT_EDGETAM_MODEL_ID).to(
        args.device,
        dtype=dtype,
    )
    model.eval()
    model_load_end_s = time.perf_counter()
    compile_start_s = time.perf_counter()
    model, compile_metadata = hf_stream._apply_compile_mode(
        model, DEFAULT_EDGETAM_COMPILE_MODE
    )
    compile_end_s = time.perf_counter()
    processor_load_start_s = time.perf_counter()
    processor = hf_stream.Sam2VideoProcessor.from_pretrained(DEFAULT_EDGETAM_MODEL_ID)
    processor_load_end_s = time.perf_counter()
    timing_ms = {
        "runtime_import_ms": _elapsed_ms(runtime_load_start_s, runtime_load_end_s),
        "model_load_ms": _elapsed_ms(model_load_start_s, model_load_end_s),
        "compile_ms": _elapsed_ms(compile_start_s, compile_end_s),
        "processor_load_ms": _elapsed_ms(
            processor_load_start_s, processor_load_end_s
        ),
        "total_ms": _elapsed_ms(init_start_s, processor_load_end_s),
    }
    print(
        "[edgetam] "
        f"model={DEFAULT_EDGETAM_MODEL_ID} device={args.device} dtype={args.dtype} "
        f"track_mode={args.track_mode} compile_mode={DEFAULT_EDGETAM_COMPILE_MODE} "
        f"applied={compile_metadata.get('applied_targets', [])}",
        flush=True,
    )
    return EdgetamRuntime(
        hf_stream=hf_stream,
        torch_module=torch_module,
        dtype=dtype,
        model=model,
        processor=processor,
        timing_ms=timing_ms,
    )


class _PreloadLeg:
    """One daemon-thread load whose failure re-raises at ``join``."""

    def __init__(self, name: str, fn: Callable[[], Any]) -> None:
        """Initialize _PreloadLeg."""
        self.name = str(name)
        self._fn = fn
        self.done = threading.Event()
        self._result: Any = None
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the leg's daemon thread; safe to call at most once."""
        if self._thread is not None:
            raise RuntimeError(f"preload leg {self.name!r} already started")
        self._thread = threading.Thread(
            target=self._run, name=f"preload-{self.name}", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            self._result = self._fn()
        except BaseException as exc:  # re-raised on the consuming worker
            self._error = exc
        finally:
            self.done.set()

    def join(self) -> Any:
        """Block until the load finishes; re-raise its failure; return result."""
        if self._thread is None:
            raise RuntimeError(f"preload leg {self.name!r} was never started")
        self._thread.join()
        if self._error is not None:
            raise self._error
        return self._result


class PerceptionPreloader:
    """Owns the camera-free model-load legs and the frame-0 readiness barrier.

    Legs mirror the worker-spawn conditions in ``_start_threads``: the EdgeTAM
    and SAM3.1 legs exist only when a seg worker will run, the tracker leg only
    when a tracker worker will run. ``start()`` is called once at ``run()``
    entry — after the FFS engine constructor on ffs runs, so its global
    ``torch.compile`` disposition still precedes the EdgeTAM compile wrap.
    """

    def __init__(self, *, args: argparse.Namespace, mode: RunMode) -> None:
        """Initialize PerceptionPreloader."""
        self.args = args
        self.mode = mode
        self._started = False
        self._legs: list[_PreloadLeg] = []
        self._edgetam: _PreloadLeg | None = None
        self._tracker: _PreloadLeg | None = None
        # Set by the seg worker once BOTH its joins (and the optional
        # precompile forward) are done and it is about to wait for frame 0 —
        # the barrier waits for this too, so a precompile that outlasts the
        # SAM3.1 load never eats into the hold-still window.
        self._seg_frame0_ready = threading.Event()
        # SAM3.1 keeps its dedicated preload thread (same runtime-cache
        # population and re-raise-at-wait semantics as before).
        self._sam31: warmup.Sam31PreloadThread | None = None
        if args.track_mode != "none":
            self._edgetam = _PreloadLeg(
                "edgetam", lambda: load_edgetam_runtime(self.args)
            )
            self._legs.append(self._edgetam)
            if mode.object_tracking_enabled or mode.controller_tracking_enabled:
                self._sam31 = warmup.Sam31PreloadThread(device=str(args.device))
        if mode.lossless_enabled or mode.tracker_enabled:
            from demo_v6_2.mdp.tracker import build_tracker_adapter

            self._tracker = _PreloadLeg(
                "tracker", lambda: build_tracker_adapter(self.args)
            )
            self._legs.append(self._tracker)

    @property
    def has_frame0_consumers(self) -> bool:
        """Return whether any frame-0 consumer leg exists (barrier applies)."""
        return bool(self._legs) or self._sam31 is not None

    def start(self) -> None:
        """Start every leg; must be called exactly once before workers run."""
        if self._started:
            raise RuntimeError("perception preload already started")
        self._started = True
        if self._sam31 is not None:
            self._sam31.start()
        for leg in self._legs:
            leg.start()

    def mark_seg_frame0_ready(self) -> None:
        """Seg worker signal: models joined, ready to consume frame 0 now."""
        self._seg_frame0_ready.set()

    def wait_frame0_consumers_ready(self, timeout: float) -> bool:
        """Wait until every frame-0 consumer can take the frame immediately.

        That means every leg finished loading AND, when a seg worker exists,
        it reported ``mark_seg_frame0_ready`` (joins + optional precompile
        done, about to wait for frame 0). A failed leg unblocks the barrier
        instead: its error re-raises on the consuming worker, whose fatal
        record sets ``stop_event`` — the barrier caller watches that event,
        so nothing hangs.
        """
        deadline = time.perf_counter() + float(timeout)
        if self._sam31 is not None:
            remaining = deadline - time.perf_counter()
            if not self._sam31.wait_done(max(0.0, remaining)):
                return False
        for leg in self._legs:
            remaining = deadline - time.perf_counter()
            if not leg.done.wait(max(0.0, remaining)):
                return False
        if any(leg._error is not None for leg in self._legs):
            return True
        if self._edgetam is not None:
            remaining = deadline - time.perf_counter()
            if not self._seg_frame0_ready.wait(max(0.0, remaining)):
                return False
        return True

    def join_edgetam(self) -> EdgetamRuntime:
        """Return the loaded EdgeTAM runtime (re-raises a load failure)."""
        if self._edgetam is None:
            raise RuntimeError("EdgeTAM preload leg was not configured")
        return self._edgetam.join()

    def join_sam31(self) -> dict[str, float] | None:
        """Return SAM3.1 preload timings, or None when the leg is absent."""
        if self._sam31 is None:
            return None
        return self._sam31.wait_for_model()

    def join_tracker(self) -> Any:
        """Return the preloaded tracker adapter (re-raises a load failure)."""
        if self._tracker is None:
            raise RuntimeError("tracker preload leg was not configured")
        return self._tracker.join()


__all__ = [
    "EdgetamRuntime",
    "PerceptionPreloader",
    "load_edgetam_runtime",
]
