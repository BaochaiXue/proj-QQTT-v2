"""Stdin GO/EXIT handshake for pre-warmed one-shot shape-prior stage workers.

A stage started with ``--wait-signal`` loads its models first, then blocks on
stdin until the orchestrator (``shape_prior_warmup.ShapePriorLocalClient``)
writes one line:

- ``GO``   -> run the stage compute exactly as the cold path would, then exit.
- ``EXIT`` -> exit without running (orchestrated shutdown before frame 0).
- EOF      -> same as ``EXIT``; also covers orchestrator death, because the
  parent's end of the stdin pipe closes when its process exits.

The worker exits after a single request either way, which releases its whole
CUDA context back to the GPU.
"""

from __future__ import annotations

import sys

GO_SIGNAL = "GO"
EXIT_SIGNAL = "EXIT"


def wait_for_go(stage_name: str) -> bool:
    """Block until the orchestrator signals; True means run the stage."""
    print(f"[prewarm] {stage_name}: models loaded, waiting for GO", flush=True)
    line = sys.stdin.readline()
    if not line or line.strip() == EXIT_SIGNAL:
        print(f"[prewarm] {stage_name}: exiting without run", flush=True)
        return False
    if line.strip() != GO_SIGNAL:
        raise ValueError(
            f"prewarm worker {stage_name!r} received unexpected signal {line!r}"
        )
    return True
