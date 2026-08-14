"""Stdin GO/EXIT handshake for pre-warmed one-shot shape-prior stage workers.

A stage started with ``--wait-signal`` loads its models first, then blocks on
stdin until the orchestrator (``shape_prior_warmup.ShapePriorLocalClient``)
writes one line:

- ``GO``   -> run the stage compute exactly as the cold path would, then exit.
- ``EXIT`` -> exit without running (orchestrated shutdown before frame 0).
- ``PRERENDER {json}`` -> optional pre-GO directive (align only): do
  speculative work whose inputs are already on disk, then keep waiting.
  Only delivered to stages that pass an ``on_directive`` handler.
- EOF      -> same as ``EXIT``; also covers orchestrator death, because the
  parent's end of the stdin pipe closes when its process exits.

The worker exits after a single request either way, which releases its whole
CUDA context back to the GPU.
"""

from __future__ import annotations

import sys
from typing import Callable

GO_SIGNAL = "GO"
EXIT_SIGNAL = "EXIT"
PRERENDER_DIRECTIVE_PREFIX = "PRERENDER "


def wait_for_go(
    stage_name: str,
    *,
    on_directive: Callable[[str], None] | None = None,
) -> bool:
    """Block until the orchestrator signals; True means run the stage.

    ``on_directive`` receives the payload of each ``PRERENDER`` line (the text
    after the prefix) and the wait continues; without a handler such a line is
    a protocol error, like any other unexpected input.
    """
    print(f"[prewarm] {stage_name}: models loaded, waiting for GO", flush=True)
    while True:
        line = sys.stdin.readline()
        if not line or line.strip() == EXIT_SIGNAL:
            print(f"[prewarm] {stage_name}: exiting without run", flush=True)
            return False
        stripped = line.strip()
        if stripped == GO_SIGNAL:
            return True
        if on_directive is not None and stripped.startswith(
            PRERENDER_DIRECTIVE_PREFIX
        ):
            on_directive(stripped[len(PRERENDER_DIRECTIVE_PREFIX) :])
            continue
        raise ValueError(
            f"prewarm worker {stage_name!r} received unexpected signal {line!r}"
        )
