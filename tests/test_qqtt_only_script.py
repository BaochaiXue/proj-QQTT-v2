from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


SCRIPT = Path("scripts/run_single_proj_qqtt_only_fake_live.sh")


class QqttOnlyScriptTest(unittest.TestCase):
    def test_script_is_bash_syntax_valid(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_script_uses_demo_v4_and_not_realtime_phystwin(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("demo_v4/realtime_futurephystwin_chunks.py", text)
        self.assertNotIn("demo_v5/realtime_futurephystwin_chunks.py", text)
        self.assertNotIn("train_online_zero_then_first.py", text)
        self.assertNotIn("optimize_online_cma.py", text)

    def test_script_auto_releases_managed_worker_after_shape_prior_chunk(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("QQTT_ONLY_AUTO_RELEASE_WORKER", text)
        self.assertIn("shape_prior_backed_chunk_ready", text)
        self.assertIn("shape_prior/points.npz", text)
        self.assertIn("shape_prior_complete", text)
        self.assertIn("shape_prior_target_counts_met", text)
        self.assertIn("shape-prior-backed-chunk-ready", text)
        self.assertNotIn('\nexec "${cmd[@]}" "$@"', text)

    def test_managed_worker_is_stopped_before_demo_process_exits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            events_path = root / "events.txt"
            base_path = root / "out"
            conda_stub = bin_dir / "conda"
            conda_stub.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    args="$*"
                    if [[ "${args}" == *"services/shape_prior_remote/server.py"* ]]; then
                      echo "[shape-prior-worker] ready bind=tcp://127.0.0.1:7103"
                      trap 'echo worker_terminated >> "${QQTT_STUB_EVENTS}"; exit 0' TERM INT
                      while true; do sleep 1; done
                    fi
                    if [[ "${args}" == *"demo_v4/realtime_futurephystwin_chunks.py"* ]]; then
                      echo demo_started >> "${QQTT_STUB_EVENTS}"
                      mkdir -p "${QQTT_ONLY_BASE_PATH}/${QQTT_ONLY_CASE_PREFIX}_demo32_capture_stub/shape_prior"
                      mkdir -p "${QQTT_ONLY_BASE_PATH}/${QQTT_ONLY_CASE_PREFIX}_chunk_0001"
                      printf 'points' > "${QQTT_ONLY_BASE_PATH}/${QQTT_ONLY_CASE_PREFIX}_demo32_capture_stub/shape_prior/points.npz"
                      printf '{"shape_prior_complete": true, "shape_prior_target_counts_met": true}\\n' > "${QQTT_ONLY_BASE_PATH}/${QQTT_ONLY_CASE_PREFIX}_chunk_0001/manifest.json"
                      sleep 2
                      echo demo_done >> "${QQTT_STUB_EVENTS}"
                      exit 0
                    fi
                    echo "unexpected conda stub invocation: ${args}" >&2
                    exit 42
                    """
                ),
                encoding="utf-8",
            )
            conda_stub.chmod(0o755)
            env = {
                **dict(os.environ),
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "QQTT_STUB_EVENTS": str(events_path),
                "QQTT_ONLY_BASE_PATH": str(base_path),
                "QQTT_ONLY_CASE_PREFIX": "demo_v4_qqtt_only",
                "QQTT_ONLY_MANAGE_WORKER": "1",
                "QQTT_ONLY_AUTO_RELEASE_WORKER": "1",
                "QQTT_ONLY_WORKER_RELEASE_POLL_S": "0.1",
                "QQTT_ONLY_WORKER_PRELOAD_MODELS": "0",
                "QQTT_ONLY_WORKER_WARMUP_MODELS": "0",
                "QQTT_ONLY_WORKER_DEBUG": "0",
            }
            result = subprocess.run(
                [str(SCRIPT)],
                check=True,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=20,
            )
            events = events_path.read_text(encoding="utf-8").splitlines()
            self.assertIn("shape-prior-backed chunk ready", result.stdout)
            self.assertIn("managed shape-prior worker was released", result.stdout)
            self.assertLess(events.index("worker_terminated"), events.index("demo_done"))


if __name__ == "__main__":
    unittest.main()
