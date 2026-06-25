#!/usr/bin/env bash
set -euo pipefail

# Run the single_proj_qqtt fake-live camera -> final_data/chunks path only.
# This wrapper intentionally uses Demo v4 and never starts demo_v5 or
# realtime_phystwin optimization.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "demo_v4/realtime_futurephystwin_chunks.py" ]]; then
  echo "[qqtt-only] must run from the single_proj_qqtt checkout" >&2
  exit 2
fi

if [[ "${PWD}" == *"/realtime_phystwin"* ]]; then
  echo "[qqtt-only] refusing to run from realtime_phystwin" >&2
  exit 2
fi

MAIN_ENV="${QQTT_ONLY_MAIN_ENV:-demo_2_max}"
SHAPE_ENV="${QQTT_ONLY_SHAPE_ENV:-phystwin-max}"
ENDPOINT="${QQTT_ONLY_SHAPE_PRIOR_ENDPOINT:-tcp://127.0.0.1:7103}"
CASE_PREFIX="${QQTT_ONLY_CASE_PREFIX:-demo_v4_qqtt_only}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_PATH="${QQTT_ONLY_BASE_PATH:-result/demo_v4/qqtt_only_fake_live_${STAMP}}"
REPLAY_FPS="${QQTT_ONLY_REPLAY_FPS:-5}"
CHUNK_SECONDS="${QQTT_ONLY_CHUNK_SECONDS:-7}"
CAPTURE_EXTRA_SECONDS="${QQTT_ONLY_CAPTURE_EXTRA_SECONDS:-120}"
REALTIME_GPU_MODE="${QQTT_ONLY_REALTIME_GPU_MODE:-single}"
WARMUP_GPU_MODE="${QQTT_ONLY_WARMUP_GPU_MODE:-dual}"
DEPTH_BACKEND="${QQTT_ONLY_DEPTH_BACKEND:-native-realsense}"
MANAGE_WORKER="${QQTT_ONLY_MANAGE_WORKER:-auto}"
WORKER_CUDA_VISIBLE_DEVICES="${QQTT_ONLY_WORKER_CUDA_VISIBLE_DEVICES:-1}"
WORKER_DEVICE="${QQTT_ONLY_WORKER_DEVICE:-cuda:0}"
WORKER_PRELOAD_MODELS="${QQTT_ONLY_WORKER_PRELOAD_MODELS:-1}"
WORKER_WARMUP_MODELS="${QQTT_ONLY_WORKER_WARMUP_MODELS:-1}"
WORKER_DEBUG="${QQTT_ONLY_WORKER_DEBUG:-1}"
AUTO_RELEASE_WORKER="${QQTT_ONLY_AUTO_RELEASE_WORKER:-1}"
WORKER_RELEASE_POLL_S="${QQTT_ONLY_WORKER_RELEASE_POLL_S:-2}"
LOG_DIR="${QQTT_ONLY_LOG_DIR:-${BASE_PATH}/logs}"

mkdir -p "${LOG_DIR}"

endpoint_port() {
  python - "$1" <<'PY'
from urllib.parse import urlparse
import sys

parsed = urlparse(sys.argv[1])
print(parsed.port or "")
PY
}

is_endpoint_listening() {
  local port
  port="$(endpoint_port "$1")"
  if [[ -z "${port}" ]]; then
    return 1
  fi
  ss -ltn | awk '{print $4}' | grep -Eq "(:|\\])${port}$"
}

worker_pid=""
demo_pid=""
worker_was_released="0"

stop_managed_worker() {
  local reason="${1:-cleanup}"
  if [[ -n "${worker_pid}" ]]; then
    echo "[qqtt-only] stopping managed shape-prior worker pid=${worker_pid} reason=${reason}"
    kill -- "-${worker_pid}" >/dev/null 2>&1 || true
    wait "${worker_pid}" >/dev/null 2>&1 || true
    worker_pid=""
    worker_was_released="1"
  fi
}

cleanup() {
  if [[ -n "${demo_pid}" ]] && kill -0 "${demo_pid}" >/dev/null 2>&1; then
    echo "[qqtt-only] stopping Demo v4 pid=${demo_pid}"
    kill "${demo_pid}" >/dev/null 2>&1 || true
    wait "${demo_pid}" >/dev/null 2>&1 || true
  fi
  stop_managed_worker "cleanup"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_worker_if_needed() {
  local should_start="0"
  case "${MANAGE_WORKER}" in
    1|true|yes)
      should_start="1"
      ;;
    0|false|no)
      should_start="0"
      ;;
    auto)
      if is_endpoint_listening "${ENDPOINT}"; then
        echo "[qqtt-only] using existing shape-prior worker at ${ENDPOINT}"
        should_start="0"
      else
        should_start="1"
      fi
      ;;
    *)
      echo "[qqtt-only] unsupported QQTT_ONLY_MANAGE_WORKER=${MANAGE_WORKER}" >&2
      exit 2
      ;;
  esac

  if [[ "${should_start}" != "1" ]]; then
    return
  fi

  local worker_log="${LOG_DIR}/shape_prior_worker.log"
  local worker_cmd=(
    conda run -n "${SHAPE_ENV}" --no-capture-output
    python services/shape_prior_remote/server.py
    --bind "${ENDPOINT}"
    --device "${WORKER_DEVICE}"
  )
  if [[ "${WORKER_PRELOAD_MODELS}" == "1" || "${WORKER_PRELOAD_MODELS}" == "true" ]]; then
    worker_cmd+=(--preload-models)
  fi
  if [[ "${WORKER_WARMUP_MODELS}" == "1" || "${WORKER_WARMUP_MODELS}" == "true" ]]; then
    worker_cmd+=(--warmup-models)
  fi
  if [[ "${WORKER_DEBUG}" == "1" || "${WORKER_DEBUG}" == "true" ]]; then
    worker_cmd+=(--debug)
  fi

  echo "[qqtt-only] starting managed shape-prior worker at ${ENDPOINT}"
  echo "[qqtt-only] worker log: ${worker_log}"
  setsid env CUDA_VISIBLE_DEVICES="${WORKER_CUDA_VISIBLE_DEVICES}" "${worker_cmd[@]}" >"${worker_log}" 2>&1 &
  worker_pid="$!"

  echo "[qqtt-only] waiting for shape-prior worker readiness"
  for _ in $(seq 1 900); do
    if grep -q "\[shape-prior-worker\].*ready" "${worker_log}" 2>/dev/null; then
      echo "[qqtt-only] shape-prior worker ready"
      return
    fi
    if ! kill -0 "${worker_pid}" >/dev/null 2>&1; then
      echo "[qqtt-only] shape-prior worker exited before ready" >&2
      tail -n 80 "${worker_log}" >&2 || true
      exit 1
    fi
    sleep 1
  done

  echo "[qqtt-only] timed out waiting for shape-prior worker readiness" >&2
  tail -n 80 "${worker_log}" >&2 || true
  exit 1
}

auto_release_enabled() {
  case "${AUTO_RELEASE_WORKER}" in
    1|true|yes)
      return 0
      ;;
    0|false|no)
      return 1
      ;;
    *)
      echo "[qqtt-only] unsupported QQTT_ONLY_AUTO_RELEASE_WORKER=${AUTO_RELEASE_WORKER}" >&2
      exit 2
      ;;
  esac
}

shape_prior_backed_chunk_ready() {
  python - "${BASE_PATH}" "${CASE_PREFIX}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

base = Path(sys.argv[1])
case_prefix = sys.argv[2]

has_points = False
for path in base.glob("**/shape_prior/points.npz"):
    try:
        if path.is_file() and path.stat().st_size > 0:
            has_points = True
            break
    except OSError:
        continue
if not has_points:
    raise SystemExit(1)

for manifest_path in sorted(base.glob(f"{case_prefix}_chunk_*/manifest.json")):
    if ".publishing" in manifest_path.parts:
        continue
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if payload.get("shape_prior_complete") or payload.get("shape_prior_target_counts_met"):
        print(str(manifest_path))
        raise SystemExit(0)

raise SystemExit(1)
PY
}

maybe_release_worker_after_shape_prior_chunk() {
  if [[ -z "${worker_pid}" ]]; then
    return
  fi
  if ! auto_release_enabled; then
    return
  fi
  if ! kill -0 "${worker_pid}" >/dev/null 2>&1; then
    worker_pid=""
    return
  fi

  local ready_manifest
  if ready_manifest="$(shape_prior_backed_chunk_ready 2>/dev/null)"; then
    echo "[qqtt-only] shape-prior-backed chunk ready: ${ready_manifest}"
    stop_managed_worker "shape-prior-backed-chunk-ready"
  fi
}

start_worker_if_needed

cmd=(
  conda run -n "${MAIN_ENV}" --no-capture-output
  python demo_v4/realtime_futurephystwin_chunks.py
  --input-source fake-live
  --depth-backend "${DEPTH_BACKEND}"
  --replay-fps "${REPLAY_FPS}"
  --chunk-seconds "${CHUNK_SECONDS}"
  --capture-extra-seconds "${CAPTURE_EXTRA_SECONDS}"
  --realtime-gpu-mode "${REALTIME_GPU_MODE}"
  --warmup-gpu-mode "${WARMUP_GPU_MODE}"
  --shape-prior-endpoint "${ENDPOINT}"
  --futurephystwin-base-path "${BASE_PATH}"
  --case-prefix "${CASE_PREFIX}"
)

if [[ -n "${QQTT_ONLY_MAX_CHUNKS:-}" ]]; then
  cmd+=(--max-chunks "${QQTT_ONLY_MAX_CHUNKS}")
fi

echo "[qqtt-only] running single_proj_qqtt only"
echo "[qqtt-only] output: ${BASE_PATH}"
echo "[qqtt-only] command: ${cmd[*]} $*"
if [[ -n "${worker_pid}" ]] && auto_release_enabled; then
  echo "[qqtt-only] managed worker will auto-stop after the first shape-prior-backed chunk"
fi

set +e
"${cmd[@]}" "$@" &
demo_pid="$!"
demo_exit=0
while true; do
  demo_status="$(ps -p "${demo_pid}" -o stat= 2>/dev/null | tr -d '[:space:]')"
  if [[ -z "${demo_status}" || "${demo_status}" == Z* ]]; then
    wait "${demo_pid}"
    demo_exit="$?"
    demo_pid=""
    break
  fi
  maybe_release_worker_after_shape_prior_chunk
  sleep "${WORKER_RELEASE_POLL_S}"
done
set -e

if [[ "${worker_was_released}" == "1" ]]; then
  echo "[qqtt-only] managed shape-prior worker was released before wrapper exit"
fi
exit "${demo_exit}"
