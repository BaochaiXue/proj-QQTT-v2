#!/usr/bin/env python3
"""Web frontend for the Demo v6.2 chunk viewer (downstream demo_visualizer).

Serves the online chunk stream as a local web page, phystwin_shen-style
(register a port, open a browser):

  NOW   the latest committed chunk auto-playing at the output cadence;
  PAST  one row per earlier chunk with every frame as a clickable tile,
        newest first, lazy-loaded.

Frames are rendered server-side by the same ``viz_panels.render_chunk_frame``
the window viewer uses — rainbow object points keyed to first-frame height
with visibility filtering (masked-out points are dropped) plus fixed-color
controller markers — so the page shows exactly what the window viewer shows.
On top of that the page adds run context a window cannot: per-chunk
``track_process_status`` and ASAP badges, output/source frame spans, commit
age, and the live ``pipeline_status.jsonl`` headline while warm-up runs.

Endpoints:
  /                          the page
  /api/state                 run + chunk index JSON (client polls this)
  /render/<chunk>/<frame>.jpg[?w=N]   rendered frame (optionally resized)
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR in sys.path:
    sys.path.remove(REPO_ROOT_STR)
sys.path.insert(0, REPO_ROOT_STR)

from demo_v6_2.orchestration.main_config import (  # noqa: E402
    DEFAULT_VISUALIZER_CONTROLLER_RADIUS,
    DEFAULT_VISUALIZER_OBJECT_COLOR_MODE,
    DEFAULT_VISUALIZER_OBJECT_RADIUS,
    DEFAULT_VISUALIZER_OBJECT_STRIDE,
    DEFAULT_VISUALIZER_WEB_HOST,
    DEFAULT_VISUALIZER_WEB_PORT,
)
from demo_v6_2.pipeline_status import (  # noqa: E402
    STAGE_FATAL,
    STAGE_LABELS,
    read_status_events,
)
from demo_v6_2.visualization.viz_camera_model import (  # noqa: E402
    CameraModel,
    _require_cv2,
    infer_case_dir,
    load_camera_model,
    load_pickle,
    normalize_online_dir,
    read_json,
)
from demo_v6_2.visualization.viz_input_timeline import (  # noqa: E402
    _chunk_frame_count,
    list_available_chunk_paths,
)
from demo_v6_2.visualization.viz_panels import (  # noqa: E402
    parse_bgr_color,
    render_chunk_frame,
)

CHUNK_CACHE_ITEMS = 32
RENDER_CACHE_ITEMS = 384
JPEG_QUALITY = 87
THUMB_WIDTH = 320
DEFAULT_PLAYBACK_FPS = 5.0
APP_ID = "demo_v6_2_chunk_viewer"
SUPERSEDE_DEADLINE_S = 6.0


class WebViewerState:
    """Chunk discovery, lazy camera loading, and cached frame rendering."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize WebViewerState."""
        self.args = args
        self.online_dir = normalize_online_dir(args.online_dir)
        self.case_dir = infer_case_dir(self.online_dir, args.case_dir)
        self._lock = threading.RLock()
        self._camera: CameraModel | None = None
        self._meta: dict[int, dict[str, Any]] = {}
        self._chunk_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._render_cache: OrderedDict[tuple[int, int, int | None], bytes] = (
            OrderedDict()
        )

    # -- camera / fps ---------------------------------------------------------

    def _camera_candidates(self) -> list[Path]:
        """Order camera-source dirs so a real extrinsic wins.

        The aggregate data dir may carry only metadata intrinsics (no
        calibrate.pkl), in which case load_camera_model silently falls back to
        an identity camera-to-world and every projected point lands off the
        object. Prefer whichever dir actually has an extrinsic source.
        """

        def rank(path: Path) -> int:
            if (path / "calibrate.pkl").is_file():
                return 0
            metadata = read_json(path / "metadata.json")
            if any(
                key in metadata for key in ("camera_to_world_c2w", "camera_to_world")
            ):
                return 1
            return 2

        return sorted((self.case_dir, self.online_dir), key=rank)

    def camera(self) -> CameraModel | None:
        """Return the camera model, retrying until calibration files appear."""
        with self._lock:
            if self._camera is not None:
                return self._camera
            for case_path in self._camera_candidates():
                try:
                    self._camera = load_camera_model(
                        case_path, cam_idx=int(self.args.cam_idx)
                    )
                    return self._camera
                except Exception:
                    continue
            return None

    def playback_fps(self) -> float:
        """Return the output playback FPS (CLI override, metadata, then 5)."""
        if self.args.fps is not None:
            return float(self.args.fps)
        camera = self.camera()
        if camera is not None and camera.metadata_fps:
            return float(camera.metadata_fps)
        return DEFAULT_PLAYBACK_FPS

    # -- chunk discovery ------------------------------------------------------

    def _chunk_meta(self, chunk: dict[str, Any], path: Path) -> dict[str, Any]:
        """Build the per-chunk metadata row served by /api/state."""
        source_frames = [int(v) for v in chunk.get("source_frame_indices", [])]
        start_frame = int(chunk.get("start_frame", 0))
        asap = chunk.get("asap_surface_points")
        # The streamed archive names color files by output frame number while
        # chunks reference recording-source ids; bridge them per chunk so
        # render_chunk_frame finds the right background without an input
        # timeline.
        background_paths: dict[int, Path] = {}
        color_dir = self.online_dir / "color" / str(int(self.args.cam_idx))
        for local, source_frame in enumerate(source_frames):
            candidate = color_dir / f"{start_frame + local}.png"
            if candidate.is_file():
                background_paths[source_frame] = candidate
        return {
            "id": int(chunk.get("chunk_id", -1)),
            "frames": _chunk_frame_count(chunk),
            "start_frame": start_frame,
            "end_frame": int(chunk.get("end_frame", start_frame)),
            "source_start": min(source_frames) if source_frames else None,
            "source_end": max(source_frames) if source_frames else None,
            "status": str(chunk.get("track_process_status") or "?"),
            "asap": bool(asap is not None and np.size(asap) > 0),
            "mtime_s": float(path.stat().st_mtime),
            "path": path,
            "background_paths": background_paths,
        }

    def scan_chunks(self) -> None:
        """Load newly committed chunks once; partial files retry next poll."""
        with self._lock:
            known = {meta["path"].resolve() for meta in self._meta.values()}
            for path in list_available_chunk_paths(
                self.online_dir, start_chunk=int(self.args.start_chunk)
            ):
                if path.resolve() in known:
                    continue
                try:
                    chunk = dict(load_pickle(path))
                except Exception:
                    continue
                meta = self._chunk_meta(chunk, path)
                self._meta[meta["id"]] = meta
                self._remember_chunk(meta["id"], chunk)

    def _remember_chunk(self, chunk_id: int, chunk: dict[str, Any]) -> None:
        """LRU-cache one loaded chunk."""
        self._chunk_cache[chunk_id] = chunk
        self._chunk_cache.move_to_end(chunk_id)
        while len(self._chunk_cache) > CHUNK_CACHE_ITEMS:
            self._chunk_cache.popitem(last=False)

    def _chunk(self, chunk_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return (chunk, meta), reloading evicted chunks from disk."""
        meta = self._meta.get(int(chunk_id))
        if meta is None:
            raise KeyError(f"unknown chunk {chunk_id}")
        cached = self._chunk_cache.get(int(chunk_id))
        if cached is None:
            cached = dict(load_pickle(meta["path"]))
            self._remember_chunk(int(chunk_id), cached)
        else:
            self._chunk_cache.move_to_end(int(chunk_id))
        return cached, meta

    # -- state + rendering ----------------------------------------------------

    def _pipeline_headline(self) -> dict[str, Any]:
        """Summarize pipeline_status.jsonl the way the window band does."""
        events = read_status_events(self.online_dir.parent, tail=200)
        latest = events[-1] if events else None
        fatal = next(
            (
                event
                for event in reversed(events)
                if not event.get("ok", True) or event.get("stage") == STAGE_FATAL
            ),
            None,
        )
        headline = fatal if fatal is not None else latest
        if headline is None:
            return {"label": "waiting for pipeline...", "detail": "", "fatal": False,
                    "source": "", "age_s": None}
        stage_key = str(headline.get("stage", ""))
        event_t = headline.get("t")
        return {
            "label": STAGE_LABELS.get(stage_key, stage_key or "—"),
            "detail": str(headline.get("detail", "")),
            "fatal": fatal is not None,
            "source": str(headline.get("source", "")),
            "age_s": None if event_t is None else max(0.0, time.time() - float(event_t)),
        }

    def state_payload(self) -> dict[str, Any]:
        """Build the /api/state JSON."""
        self.scan_chunks()
        with self._lock:
            manifest = read_json(self.online_dir / "manifest.json")
            now_s = time.time()
            chunks = [
                {
                    "id": meta["id"],
                    "frames": meta["frames"],
                    "start_frame": meta["start_frame"],
                    "end_frame": meta["end_frame"],
                    "source_start": meta["source_start"],
                    "source_end": meta["source_end"],
                    "status": meta["status"],
                    "asap": meta["asap"],
                    "age_s": max(0.0, now_s - meta["mtime_s"]),
                }
                for _, meta in sorted(self._meta.items())
            ]
            return {
                "app": APP_ID,
                "ready": self.camera() is not None,
                "case_name": manifest.get("case_name"),
                "chunk_size": manifest.get("chunk_size"),
                "num_frames_total": manifest.get("num_frames_total"),
                "fps": self.playback_fps(),
                "latest_chunk": chunks[-1]["id"] if chunks else -1,
                "pipeline": self._pipeline_headline(),
                "chunks": chunks,
            }

    def render_jpeg(self, chunk_id: int, frame: int, width: int | None) -> bytes:
        """Render one chunk frame (window-viewer semantics) to JPEG bytes."""
        cv2 = _require_cv2()
        with self._lock:
            key = (int(chunk_id), int(frame), width)
            cached = self._render_cache.get(key)
            if cached is not None:
                self._render_cache.move_to_end(key)
                return cached
            camera = self.camera()
            if camera is None:
                raise RuntimeError("camera calibration is not available yet")
            chunk, meta = self._chunk(int(chunk_id))
            if not 0 <= int(frame) < int(meta["frames"]):
                raise KeyError(f"chunk {chunk_id} has no frame {frame}")
            image = render_chunk_frame(
                chunk,
                local_frame=int(frame),
                case_dir=self.case_dir,
                camera=camera,
                cam_idx=int(self.args.cam_idx),
                use_background=not bool(self.args.no_background),
                show_invisible_object_points=bool(
                    self.args.show_invisible_object_points
                ),
                object_stride=int(self.args.object_stride),
                object_radius=int(self.args.object_radius),
                controller_radius=int(self.args.controller_radius),
                object_color_mode=str(self.args.object_color_mode),
                controller_color=self.args.controller_color,
                fps=self.playback_fps(),
                background_frame_paths=meta["background_paths"],
            )
            if width is not None and 0 < int(width) < image.shape[1]:
                scale = int(width) / image.shape[1]
                image = cv2.resize(
                    image, (int(width), max(1, int(round(image.shape[0] * scale))))
                )
            ok, encoded = cv2.imencode(
                ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            payload = encoded.tobytes()
            self._render_cache[key] = payload
            self._render_cache.move_to_end(key)
            while len(self._render_cache) > RENDER_CACHE_ITEMS:
                self._render_cache.popitem(last=False)
            return payload


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Demo v6.2 chunk viewer</title>
  <style>
    :root { --tile-w: 240px; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #101216; color: #d8dbe2;
           font: 13px/1.45 system-ui, sans-serif; }
    #toolbar { position: sticky; top: 0; z-index: 5; display: flex;
               align-items: center; gap: 12px; padding: 8px 14px;
               background: #171a21; border-bottom: 1px solid #262b36; }
    #toolbar button { background: #232936; color: #d8dbe2; border: 1px solid
                      #333b4d; border-radius: 4px; padding: 4px 12px;
                      cursor: pointer; }
    #toolbar button.active { background: #1f6f43; border-color: #2c9c5e; }
    #status { color: #8b93a5; margin-left: auto; }
    #pipeline { padding: 6px 14px; background: #1c2230; color: #aab2c4; }
    #pipeline.fatal { background: #5b1a1a; color: #ffd7d7; }
    main { padding: 12px 14px 40px; }
    h2 { font-size: 12px; letter-spacing: .12em; text-transform: uppercase;
         color: #7d8698; margin: 18px 0 8px; }
    #nowImg { max-width: min(96vw, 1100px); border: 1px solid #2a3140;
              border-radius: 4px; display: block; }
    #nowCaption { color: #9aa3b5; margin-top: 6px; }
    #nowEmpty { padding: 40px; border: 1px dashed #333b4d; border-radius: 6px;
                color: #8b93a5; max-width: 640px; }
    .row { margin-bottom: 14px; }
    .rowhead { display: flex; gap: 8px; align-items: baseline;
               color: #9aa3b5; margin-bottom: 4px; }
    .rowhead b { color: #d8dbe2; font-family: ui-monospace, monospace; }
    .badge { font-size: 11px; padding: 1px 7px; border-radius: 8px;
             border: 1px solid #333b4d; }
    .badge.ok { color: #79d69a; border-color: #2c5c3e; }
    .badge.warn { color: #e8c46a; border-color: #6b5a26; }
    .badge.asap { color: #7db4ff; border-color: #2c4a75; }
    .tiles { display: flex; gap: 6px; overflow-x: auto; padding-bottom: 4px; }
    .tiles img { width: var(--tile-w); border: 1px solid #262b36;
                 border-radius: 3px; cursor: pointer; flex: 0 0 auto; }
    .tiles img.sel { border-color: #2c9c5e; }
  </style>
</head>
<body>
  <div id="toolbar">
    <button id="liveBtn" class="active" title="Follow the latest chunk">LIVE</button>
    <button id="playBtn" title="Play/pause frame cycling">Pause</button>
    <label>zoom <input id="zoom" type="range" min="120" max="480" step="20" value="240"></label>
    <span id="fpsLabel"></span>
    <div id="status">connecting…</div>
  </div>
  <div id="pipeline"></div>
  <main>
    <h2>now</h2>
    <div id="nowEmpty">waiting for the first final_data chunk…</div>
    <img id="nowImg" style="display:none" alt="latest chunk frame">
    <div id="nowCaption"></div>
    <h2>past</h2>
    <div id="history"></div>
  </main>
  <script>
    const S = {
      chunks: new Map(), order: [], latest: -1, live: true, playing: true,
      sel: { chunk: -1, frame: 0 }, fps: 5, timer: null,
    };
    const els = {
      liveBtn: document.getElementById('liveBtn'),
      playBtn: document.getElementById('playBtn'),
      zoom: document.getElementById('zoom'),
      fpsLabel: document.getElementById('fpsLabel'),
      status: document.getElementById('status'),
      pipeline: document.getElementById('pipeline'),
      nowImg: document.getElementById('nowImg'),
      nowEmpty: document.getElementById('nowEmpty'),
      nowCaption: document.getElementById('nowCaption'),
      history: document.getElementById('history'),
    };

    function badge(text, cls) {
      return `<span class="badge ${cls}">${text}</span>`;
    }
    function rowHeadHtml(c) {
      const statusCls = c.status === 'normal' ? 'ok' : 'warn';
      const asap = c.asap ? badge('ASAP', 'asap') : '';
      const age = c.age_s == null ? '' : `${Math.round(c.age_s)}s ago`;
      return `<b>chunk ${String(c.id).padStart(6, '0')}</b>` +
        badge(c.status, statusCls) + asap +
        `<span>out ${c.start_frame}–${c.end_frame - 1}` +
        ` · src ${c.source_start}–${c.source_end} · ${age}</span>`;
    }
    function makeRow(c) {
      const row = document.createElement('div');
      row.className = 'row';
      row.id = `row-${c.id}`;
      const head = document.createElement('div');
      head.className = 'rowhead';
      head.innerHTML = rowHeadHtml(c);
      const tiles = document.createElement('div');
      tiles.className = 'tiles';
      for (let f = 0; f < c.frames; f += 1) {
        const img = document.createElement('img');
        img.loading = 'lazy';
        img.src = `/render/${c.id}/${f}.jpg?w=__THUMB_W__`;
        img.onclick = () => selectFrame(c.id, f);
        tiles.appendChild(img);
      }
      row.appendChild(head);
      row.appendChild(tiles);
      return row;
    }
    function selectFrame(chunkId, frame) {
      S.live = false;
      S.sel = { chunk: chunkId, frame: frame };
      els.liveBtn.classList.remove('active');
      showFrame();
      markSelection();
    }
    function markSelection() {
      document.querySelectorAll('.tiles img.sel').forEach((el) =>
        el.classList.remove('sel'));
      if (S.live) return;
      const row = document.getElementById(`row-${S.sel.chunk}`);
      if (row) {
        const img = row.querySelectorAll('img')[S.sel.frame];
        if (img) img.classList.add('sel');
      }
    }
    function showFrame() {
      const c = S.chunks.get(S.sel.chunk);
      if (!c) return;
      const url = `/render/${S.sel.chunk}/${S.sel.frame}.jpg`;
      const im = new Image();
      im.onload = () => {
        els.nowImg.src = url;
        els.nowImg.style.display = 'block';
        els.nowEmpty.style.display = 'none';
      };
      im.src = url;
      const mode = S.live ? 'LIVE' : 'history';
      els.nowCaption.textContent =
        `${mode} · chunk ${String(S.sel.chunk).padStart(6, '0')} · frame ` +
        `${S.sel.frame + 1}/${c.frames} · status ${c.status}` +
        (c.asap ? ' · ASAP' : '');
    }
    function tick() {
      if (!S.playing) return;
      if (S.live) {
        if (S.latest < 0) return;
        S.sel.chunk = S.latest;
      }
      const c = S.chunks.get(S.sel.chunk);
      if (!c) return;
      S.sel.frame = (S.sel.frame + 1) % c.frames;
      showFrame();
      if (!S.live) markSelection();
    }
    function armTimer() {
      if (S.timer) clearInterval(S.timer);
      S.timer = setInterval(tick, Math.max(60, 1000 / Math.max(0.5, S.fps)));
    }
    async function poll() {
      let data;
      try {
        const resp = await fetch('/api/state');
        data = await resp.json();
      } catch (err) {
        els.status.textContent = 'server unreachable';
        return;
      }
      if (Math.abs((data.fps || 5) - S.fps) > 1e-6) {
        S.fps = data.fps || 5;
        armTimer();
      }
      els.fpsLabel.textContent = `${S.fps.toFixed(1)} fps`;
      const p = data.pipeline || {};
      els.pipeline.className = p.fatal ? 'fatal' : '';
      els.pipeline.textContent = `PIPELINE: ${p.label || ''}` +
        (p.detail ? ` — ${p.detail}` : '') +
        (p.age_s == null ? '' : `  (${p.source} ${Math.round(p.age_s)}s ago)`);
      for (const c of data.chunks || []) {
        if (!S.chunks.has(c.id)) {
          S.chunks.set(c.id, c);
          S.order.push(c.id);
          els.history.prepend(makeRow(c));
        } else {
          S.chunks.set(c.id, c);
          const row = document.getElementById(`row-${c.id}`);
          if (row) row.querySelector('.rowhead').innerHTML = rowHeadHtml(c);
        }
      }
      S.latest = data.latest_chunk;
      els.status.textContent = data.latest_chunk >= 0
        ? `${(data.chunks || []).length} chunks · latest ${data.latest_chunk}` +
          (data.case_name ? ` · ${data.case_name}` : '')
        : 'no chunks yet';
      if (S.latest < 0) {
        els.nowEmpty.textContent =
          `waiting for the first final_data chunk… (${p.label || 'starting'})`;
      }
    }
    els.liveBtn.onclick = () => {
      S.live = true;
      S.playing = true;
      els.playBtn.textContent = 'Pause';
      els.liveBtn.classList.add('active');
      markSelection();
    };
    els.playBtn.onclick = () => {
      S.playing = !S.playing;
      els.playBtn.textContent = S.playing ? 'Pause' : 'Play';
    };
    els.zoom.oninput = () => {
      document.documentElement.style.setProperty('--tile-w', `${els.zoom.value}px`);
    };
    armTimer();
    poll();
    setInterval(poll, 1000);
  </script>
</body>
</html>
"""

PAGE = HTML.replace("__THUMB_W__", str(THUMB_WIDTH)).encode("utf-8")


def make_handler(state: WebViewerState, *, quiet: bool):
    """Build the request handler bound to one viewer state."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "DemoV62ChunkViewer/1.0"

        def log_message(self, fmt: str, *log_args: Any) -> None:
            if not quiet:
                super().log_message(fmt, *log_args)

        def send_payload(
            self, payload: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_payload(data, "application/json", status=status)

        def do_GET(self) -> None:  # noqa: N802 (http.server API name)
            parsed = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            try:
                if parsed.path == "/":
                    self.send_payload(PAGE, "text/html; charset=utf-8")
                    return
                if parsed.path == "/api/state":
                    self.send_json(state.state_payload())
                    return
                if parsed.path == "/api/shutdown":
                    # A newer viewer instance supersedes this one: the run
                    # leaves the viewer alive for post-run review, so the next
                    # run's viewer must be able to reclaim the port.
                    self.send_json({"ok": True})
                    threading.Thread(
                        target=self.server.shutdown, daemon=True
                    ).start()
                    return
                if parsed.path.startswith("/render/") and parsed.path.endswith(
                    ".jpg"
                ):
                    parts = parsed.path.split("/")
                    if len(parts) != 4:
                        raise ValueError("expected /render/<chunk>/<frame>.jpg")
                    chunk_id = int(parts[2])
                    frame = int(Path(parts[3]).stem)
                    width = int(query["w"]) if "w" in query else None
                    state.scan_chunks()
                    payload = state.render_jpeg(chunk_id, frame, width)
                    self.send_payload(payload, "image/jpeg")
                    return
                self.send_json(
                    {"error": "not found"}, status=HTTPStatus.NOT_FOUND
                )
            except KeyError as exc:
                self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self.send_json(
                    {"error": f"{type(exc).__name__}: {exc}"},
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )

    return Handler


def build_parser() -> argparse.ArgumentParser:
    """Build the web viewer CLI parser (defaults come from config/default.yaml)."""
    parser = argparse.ArgumentParser(
        description="Serve Demo v6.2 online chunks as a local web page."
    )
    parser.add_argument("--online-dir", type=Path, required=True,
                        help="Path to online_data or its chunks directory.")
    parser.add_argument("--case-dir", type=Path, default=None,
                        help="Path to data. Inferred from --online-dir when omitted.")
    parser.add_argument("--host", default=DEFAULT_VISUALIZER_WEB_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_VISUALIZER_WEB_PORT)
    parser.add_argument("--cam-idx", type=int, default=0)
    parser.add_argument("--fps", type=float, default=None,
                        help="Playback FPS. Defaults to metadata fps, then 5.")
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--object-stride", type=int,
                        default=DEFAULT_VISUALIZER_OBJECT_STRIDE)
    parser.add_argument("--object-radius", type=int,
                        default=DEFAULT_VISUALIZER_OBJECT_RADIUS)
    parser.add_argument("--controller-radius", type=int,
                        default=DEFAULT_VISUALIZER_CONTROLLER_RADIUS)
    parser.add_argument("--object-color-mode",
                        choices=("rainbow", "green", "object-colors"),
                        default=DEFAULT_VISUALIZER_OBJECT_COLOR_MODE)
    parser.add_argument("--controller-color", type=parse_bgr_color,
                        default=parse_bgr_color("0,0,255"))
    parser.add_argument("--show-invisible-object-points", action="store_true")
    parser.add_argument("--no-background", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _ask_previous_instance_to_exit(host: str, port: int) -> bool:
    """Shut down an earlier viewer instance holding the port, if it is ours."""
    import urllib.request

    base = f"http://{host}:{int(port)}"
    try:
        with urllib.request.urlopen(f"{base}/api/state", timeout=2) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return False
    # Accept the explicit marker, plus marker-less payloads with our state
    # shape (an instance from before the marker existed).
    ours = payload.get("app") == APP_ID or {
        "latest_chunk",
        "pipeline",
        "chunks",
    } <= set(payload)
    if not ours:
        return False
    try:
        with urllib.request.urlopen(f"{base}/api/shutdown", timeout=2) as resp:
            resp.read()
    except Exception:
        pass
    return True


def _bind_server(host: str, port: int, handler) -> ThreadingHTTPServer:
    """Bind the port, superseding a leftover viewer from a previous run."""
    asked_previous = False
    deadline = time.monotonic() + SUPERSEDE_DEADLINE_S
    while True:
        try:
            return ThreadingHTTPServer((host, int(port)), handler)
        except OSError as exc:
            if not asked_previous:
                asked_previous = True
                if not _ask_previous_instance_to_exit(host, port):
                    raise RuntimeError(
                        f"port {port} is in use by another application: {exc}"
                    ) from exc
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"previous viewer did not release port {port}: {exc}"
                ) from exc
            time.sleep(0.3)


def main(argv: list[str] | None = None) -> int:
    """Run the command-line entry point."""
    args = build_parser().parse_args(argv)
    state = WebViewerState(args)
    handler = make_handler(state, quiet=bool(args.quiet))
    server = _bind_server(str(args.host), int(args.port), handler)
    url = f"http://{args.host}:{int(args.port)}/"
    print(f"[web-viewer] serving {state.online_dir}", flush=True)
    print(f"[web-viewer] open {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web-viewer] stopped", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
