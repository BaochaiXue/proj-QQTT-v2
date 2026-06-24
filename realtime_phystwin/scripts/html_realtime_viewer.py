#!/usr/bin/env python3
"""Serve an HTML realtime viewer for online training snapshots."""

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import pickle
from urllib.parse import parse_qs, urlparse

import numpy as np


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Online Training Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101214;
      --panel: #191d21;
      --line: #343a40;
      --text: #edf0f2;
      --muted: #aeb6bf;
      --accent: #4fb477;
      --tile-width: 360px;
      --tile-height: 204px;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      height: 100%;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 13px;
    }
    body {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
    }
    #toolbar {
      display: flex;
      align-items: center;
      gap: 12px;
      min-height: 48px;
      padding: 8px 12px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      white-space: nowrap;
      overflow-x: auto;
    }
    #viewport {
      min-height: 0;
      overflow: auto;
      background: #0b0d0f;
    }
    #grid {
      width: max-content;
      padding: 10px;
    }
    #topRows {
      position: sticky;
      top: 0;
      z-index: 4;
      padding-bottom: 10px;
      background: #0b0d0f;
      border-bottom: 1px solid #20262c;
    }
    #historyRows {
      padding-top: 10px;
    }
    .control {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
    }
    button, select, input {
      color: var(--text);
      background: #22272c;
      border: 1px solid var(--line);
      border-radius: 4px;
      height: 28px;
    }
    button {
      padding: 0 10px;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    input[type="number"] { width: 64px; padding: 0 6px; }
    input[type="range"] { width: 180px; }
    #status {
      color: var(--muted);
      margin-left: auto;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .row {
      display: grid;
      grid-template-columns: 104px repeat(var(--window-count), var(--tile-width));
      align-items: stretch;
    }
    .row + .row { margin-top: 8px; }
    .rowLabel {
      position: sticky;
      left: 0;
      z-index: 3;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: var(--tile-height);
      background: #15191d;
      border: 1px solid var(--line);
      color: var(--muted);
      writing-mode: vertical-rl;
      text-orientation: mixed;
    }
    .topRow .rowLabel,
    .historyRow .rowLabel {
      color: var(--text);
      background: #161d19;
      flex-direction: column;
      gap: 8px;
      padding: 8px;
      writing-mode: horizontal-tb;
      text-orientation: mixed;
    }
    .rowPlay {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
    }
    .rowPlay input {
      width: 16px;
      height: 16px;
    }
    canvas {
      width: var(--tile-width);
      height: var(--tile-height);
      display: block;
      background: #030405;
      border: 1px solid #22282e;
    }
    .emptyTile {
      width: var(--tile-width);
      height: var(--tile-height);
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      border: 1px solid #22282e;
      background: #050607;
    }
  </style>
</head>
<body>
  <div id="toolbar">
    <button id="playBtn" title="Play/pause frames">Pause</button>
    <label class="control">frame <input id="frameSlider" type="range" min="0" max="0" value="0"></label>
    <label class="control">fps <input id="fpsInput" type="number" min="1" max="60" value="8"></label>
    <label class="control">zoom <input id="zoomInput" type="range" min="180" max="900" step="20" value="360"></label>
    <label class="control">points
      <select id="pointMode">
        <option value="surface" selected>surface</option>
        <option value="original">original</option>
        <option value="all">all</option>
      </select>
    </label>
    <label class="control">stride <input id="pointStrideInput" type="number" min="1" max="100" value="5"></label>
    <button id="playVisibleHistoryBtn" title="Play currently visible history rows">Play visible</button>
    <button id="stopHistoryBtn" title="Stop all history rows">Stop history</button>
    <button id="refreshBtn" title="Refresh manifest">Refresh</button>
    <div id="status">waiting</div>
  </div>
  <main id="viewport">
    <div id="grid">
      <div id="topRows"></div>
      <div id="historyRows"></div>
    </div>
  </main>

  <script>
    const state = {
      manifest: null,
      iterations: [],
      latestIteration: null,
      frame: 0,
      segmentLen: 1,
      windowCount: 1,
      playing: true,
      cache: new Map(),
      imageCache: new Map(),
      rows: new Map(),
      visibleRows: new Set(),
      playRows: new Set(['top:origin', 'top:latest', 'top:first']),
      observer: null,
      manifestSignature: '',
      historyStride: 10,
      renderingTop: false,
      pendingTop: false,
    };

    const els = {
      status: document.getElementById('status'),
      viewport: document.getElementById('viewport'),
      topRows: document.getElementById('topRows'),
      historyRows: document.getElementById('historyRows'),
      frameSlider: document.getElementById('frameSlider'),
      fpsInput: document.getElementById('fpsInput'),
      zoomInput: document.getElementById('zoomInput'),
      pointMode: document.getElementById('pointMode'),
      pointStrideInput: document.getElementById('pointStrideInput'),
      playBtn: document.getElementById('playBtn'),
      playVisibleHistoryBtn: document.getElementById('playVisibleHistoryBtn'),
      stopHistoryBtn: document.getElementById('stopHistoryBtn'),
    };

    function setStatus(text) {
      els.status.textContent = text;
    }

    async function fetchJson(url) {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(await response.text());
      }
      return await response.json();
    }

    async function loadImage(url) {
      if (state.imageCache.has(url)) {
        return state.imageCache.get(url);
      }
      const promise = new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = url;
      });
      state.imageCache.set(url, promise);
      return promise;
    }

    function updateTileSize(imageWidth = 848, imageHeight = 480) {
      const tileWidth = Number(els.zoomInput.value);
      const tileHeight = Math.max(1, Math.round(imageHeight * tileWidth / imageWidth));
      document.documentElement.style.setProperty('--tile-width', `${tileWidth}px`);
      document.documentElement.style.setProperty('--tile-height', `${tileHeight}px`);
      return { tileWidth, tileHeight };
    }

    async function getBundle(iteration) {
      const pointMode = els.pointMode.value;
      const pointStride = Math.max(1, Number(els.pointStrideInput.value) || 1);
      const key = `${iteration}:${state.frame}:${pointMode}:${pointStride}`;
      if (state.cache.has(key)) {
        return state.cache.get(key);
      }
      const url = `/api/frame_bundle?iteration=${iteration}&frame=${state.frame}&point_mode=${pointMode}&point_stride=${pointStride}`;
      const bundle = await fetchJson(url);
      if (state.cache.size > 96) {
        state.cache.clear();
      }
      state.cache.set(key, bundle);
      return bundle;
    }

    function drawPoints(ctx, points, color, radius, sx, sy) {
      if (!points || points.length === 0) {
        return;
      }
      ctx.fillStyle = color;
      const size = Math.max(1, radius * 2 + 1);
      for (const point of points) {
        const x = Math.round(point[0] * sx) - radius;
        const y = Math.round(point[1] * sy) - radius;
        ctx.fillRect(x, y, size, size);
      }
    }

    function drawLabel(ctx, text) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
      ctx.fillRect(0, 0, ctx.canvas.width, 24);
      ctx.fillStyle = '#ffffff';
      ctx.font = '13px ui-sans-serif, system-ui, sans-serif';
      ctx.fillText(text, 8, 17);
    }

    async function drawTile(canvas, tile, mode) {
      if (!tile) {
        return;
      }
      const img = await loadImage(tile.image_url);
      const { tileWidth, tileHeight } = updateTileSize(tile.image_width, tile.image_height);
      canvas.width = tileWidth;
      canvas.height = tileHeight;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, tileWidth, tileHeight);
      ctx.drawImage(img, 0, 0, tileWidth, tileHeight);
      const sx = tileWidth / tile.image_width;
      const sy = tileHeight / tile.image_height;
      if (mode !== 'raw') {
        drawPoints(ctx, tile.gt_uv, '#ff2f2f', 1, sx, sy);
        drawPoints(ctx, tile.pred_uv, '#23ff48', 2, sx, sy);
        drawPoints(ctx, tile.controller_uv, '#2f9bff', 3, sx, sy);
      }
      drawLabel(ctx, tile.label);
    }

    function makeLabel(text) {
      const labelEl = document.createElement('div');
      labelEl.className = 'rowLabel';
      labelEl.textContent = text;
      return labelEl;
    }

    function makePlayableLabel(titleText, playKey, onPlay) {
      const labelEl = document.createElement('div');
      labelEl.className = 'rowLabel';

      const title = document.createElement('div');
      title.textContent = titleText;
      labelEl.appendChild(title);

      const playLabel = document.createElement('label');
      playLabel.className = 'rowPlay';
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.checked = state.playRows.has(playKey);
      checkbox.onchange = () => {
        if (checkbox.checked) {
          state.playRows.add(playKey);
          onPlay();
        } else {
          state.playRows.delete(playKey);
        }
      };
      playLabel.appendChild(checkbox);
      playLabel.appendChild(document.createTextNode('play'));
      labelEl.appendChild(playLabel);

      return labelEl;
    }

    function makeHistoryLabel(iteration) {
      return makePlayableLabel(
        `iter ${iteration}`,
        `history:${iteration}`,
        () => renderHistoryRow(iteration, true)
      );
    }

    function makeTopLabel(label, playKey) {
      return makePlayableLabel(label, playKey, () => renderTopRows(true));
    }

    function makeEmptyTile(text = 'waiting') {
      const empty = document.createElement('div');
      empty.className = 'emptyTile';
      empty.textContent = text;
      return empty;
    }

    function createRow(label, windowCount, className = '') {
      const row = document.createElement('section');
      row.className = `row ${className}`.trim();
      row.style.setProperty('--window-count', String(Math.max(1, windowCount)));
      if (row.classList.contains('topRow')) {
        row.appendChild(makeTopLabel(label, row.dataset.playKey));
      } else {
        row.appendChild(makeLabel(label));
      }
      for (let i = 0; i < Math.max(1, windowCount); i += 1) {
        row.appendChild(makeEmptyTile());
      }
      return row;
    }

    function populateRow(row, label, tiles, mode) {
      row.replaceChildren();
      row.style.setProperty('--window-count', String(Math.max(1, tiles.length)));
      if (row.classList.contains('historyRow')) {
        row.appendChild(makeHistoryLabel(Number(row.dataset.iteration)));
      } else if (row.classList.contains('topRow')) {
        row.appendChild(makeTopLabel(label, row.dataset.playKey));
      } else {
        row.appendChild(makeLabel(label));
      }
      for (const tile of tiles) {
        if (!tile) {
          row.appendChild(makeEmptyTile('missing'));
          continue;
        }
        const canvas = document.createElement('canvas');
        row.appendChild(canvas);
        drawTile(canvas, tile, mode).catch((error) => {
          console.error(error);
          canvas.replaceWith(makeEmptyTile('draw failed'));
        });
      }
    }

    function createObserver() {
      if (state.observer) {
        state.observer.disconnect();
      }
      state.observer = new IntersectionObserver((entries) => {
        for (const entry of entries) {
          const iteration = Number(entry.target.dataset.iteration);
          if (entry.isIntersecting) {
            state.visibleRows.add(iteration);
            renderHistoryRow(iteration);
          } else {
            state.visibleRows.delete(iteration);
          }
        }
      }, {
        root: els.viewport,
        rootMargin: '900px 0px',
        threshold: 0.01,
      });
    }

    function buildHistoryRows() {
      createObserver();
      state.rows.clear();
      state.visibleRows.clear();
      els.historyRows.innerHTML = '';
      const historyIterations = [...state.iterations]
        .sort((a, b) => a - b)
        .filter((iteration) => (
          iteration !== state.latestIteration
          && iteration % state.historyStride === 0
        ));
      state.playRows = new Set(
        [...state.playRows].filter((playKey) => {
          if (String(playKey).startsWith('top:')) {
            return true;
          }
          if (!String(playKey).startsWith('history:')) {
            return false;
          }
          const iteration = Number(String(playKey).split(':')[1]);
          return historyIterations.includes(iteration);
        })
      );
      for (const iteration of historyIterations) {
        const row = createRow(`iter ${iteration}`, state.windowCount, 'historyRow');
        row.dataset.iteration = String(iteration);
        row.dataset.renderKey = '';
        state.rows.set(iteration, row);
        els.historyRows.appendChild(row);
        state.observer.observe(row);
      }
    }

    async function renderTopRows(onlyChecked = false) {
      if (state.latestIteration === null) {
        return;
      }
      if (state.renderingTop) {
        state.pendingTop = true;
        return;
      }
      state.renderingTop = true;
      state.pendingTop = false;
      try {
        const bundle = await getBundle(state.latestIteration);
        state.segmentLen = Math.max(1, bundle.segment_len || 1);
        state.frame %= state.segmentLen;
        state.windowCount = Math.max(1, bundle.windows.length);
        els.frameSlider.max = state.segmentLen - 1;
        els.frameSlider.value = state.frame;

        const rawTiles = bundle.windows.map((window) => window.current);
        const latestTiles = bundle.windows.map((window) => window.current);
        const firstTiles = bundle.windows.map((window) => window.first_seen);
        let originRow = document.getElementById('originRow');
        let latestRow = document.getElementById('latestRow');
        let firstRow = document.getElementById('firstSeenRow');
        if (!originRow || !latestRow || !firstRow) {
          els.topRows.innerHTML = '';
          originRow = createRow('origin', state.windowCount, 'topRow');
          latestRow = createRow('latest', state.windowCount, 'topRow');
          firstRow = createRow('first-seen', state.windowCount, 'topRow');
          originRow.id = 'originRow';
          latestRow.id = 'latestRow';
          firstRow.id = 'firstSeenRow';
          originRow.dataset.playKey = 'top:origin';
          latestRow.dataset.playKey = 'top:latest';
          firstRow.dataset.playKey = 'top:first';
          els.topRows.appendChild(originRow);
          els.topRows.appendChild(latestRow);
          els.topRows.appendChild(firstRow);
        }
        if (!onlyChecked || state.playRows.has('top:origin')) {
          populateRow(originRow, 'origin', rawTiles, 'raw');
        }
        if (!onlyChecked || state.playRows.has('top:latest')) {
          populateRow(latestRow, `latest ${bundle.iteration}`, latestTiles, 'overlay');
        }
        if (!onlyChecked || state.playRows.has('top:first')) {
          populateRow(firstRow, 'first-seen', firstTiles, 'overlay');
        }
        setStatus(`latest=${bundle.iteration} frame=${state.frame}/${state.segmentLen - 1} history_rows=${state.rows.size} step=${state.historyStride} windows=${bundle.windows.length}`);
      } catch (error) {
        console.error(error);
        setStatus(error.message);
      } finally {
        state.renderingTop = false;
        if (state.pendingTop) {
          renderTopRows();
        }
      }
    }

    async function renderHistoryRow(iteration, force = false) {
      const row = state.rows.get(iteration);
      if (!row) {
        return;
      }
      const renderKey = `${iteration}:${state.frame}:${els.pointMode.value}:${els.pointStrideInput.value}:${els.zoomInput.value}`;
      if (!force && row.dataset.renderKey === renderKey) {
        return;
      }
      row.dataset.renderKey = renderKey;
      try {
        const bundle = await getBundle(iteration);
        if (row.dataset.renderKey !== renderKey) {
          return;
        }
        const tiles = bundle.windows.map((window) => window.current);
        populateRow(row, `iter ${bundle.iteration}`, tiles, 'overlay');
      } catch (error) {
        console.error(error);
        populateRow(row, `iter ${iteration}`, [], 'overlay');
      }
    }

    function renderVisibleHistoryRows(onlyChecked = false) {
      for (const iteration of state.visibleRows) {
        if (onlyChecked && !state.playRows.has(`history:${iteration}`)) {
          continue;
        }
        renderHistoryRow(iteration);
      }
    }

    function refreshAllRows() {
      renderTopRows();
      renderVisibleHistoryRows();
    }

    function refreshPlayingRows() {
      renderTopRows(true);
      renderVisibleHistoryRows(true);
    }

    async function loadManifest() {
      const manifest = await fetchJson('/api/manifest');
      const iterations = manifest.iterations || [];
      if (iterations.length === 0 && manifest.latest_iteration !== null) {
        iterations.push(manifest.latest_iteration);
      }
      const latestIteration = manifest.latest_iteration !== null
        ? manifest.latest_iteration
        : (iterations.length > 0 ? iterations[iterations.length - 1] : null);
      if (latestIteration === null) {
        setStatus('no iteration snapshots yet');
        return;
      }

      state.manifest = manifest;
      state.iterations = iterations;
      state.latestIteration = latestIteration;
      state.segmentLen = Math.max(1, manifest.segment_len || state.segmentLen);
      state.windowCount = Math.max(1, (manifest.window_starts || []).length || state.windowCount);
      updateTileSize(manifest.image_width || 848, manifest.image_height || 480);

      const signature = `${latestIteration}:${iterations.join(',')}:${state.windowCount}`;
      if (signature !== state.manifestSignature) {
        state.manifestSignature = signature;
        buildHistoryRows();
        refreshAllRows();
      } else {
        renderTopRows();
      }
    }

    function tick() {
      if (state.playing && state.segmentLen > 0) {
        state.frame = (state.frame + 1) % state.segmentLen;
        refreshPlayingRows();
      }
      const fps = Math.max(1, Number(els.fpsInput.value) || 8);
      window.setTimeout(tick, Math.round(1000 / fps));
    }

    document.getElementById('refreshBtn').onclick = () => loadManifest();
    els.playBtn.onclick = () => {
      state.playing = !state.playing;
      els.playBtn.textContent = state.playing ? 'Pause' : 'Play';
    };
    els.playVisibleHistoryBtn.onclick = () => {
      for (const iteration of state.visibleRows) {
        state.playRows.add(`history:${iteration}`);
        renderHistoryRow(iteration, true);
      }
    };
    els.stopHistoryBtn.onclick = () => {
      state.playRows = new Set(
        [...state.playRows].filter((playKey) => String(playKey).startsWith('top:'))
      );
      refreshAllRows();
    };
    els.frameSlider.oninput = () => {
      state.playing = false;
      els.playBtn.textContent = 'Play';
      state.frame = Number(els.frameSlider.value);
      refreshAllRows();
    };
    els.zoomInput.oninput = () => {
      refreshAllRows();
    };
    els.pointMode.onchange = () => {
      state.cache.clear();
      refreshAllRows();
    };
    els.pointStrideInput.oninput = () => {
      state.cache.clear();
      refreshAllRows();
    };

    loadManifest();
    window.setInterval(loadManifest, 2000);
    tick();
  </script>
</body>
</html>
"""


def scalar_int(value):
    return int(np.asarray(value).reshape(-1)[0])


def parse_bool(value, default=False):
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "on"}


def load_camera(base_path, case_name, cam_idx):
    case_dir = Path(base_path) / case_name
    with open(case_dir / "calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    with open(case_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    c2w = np.asarray(c2ws[int(cam_idx)], dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    intrinsic = np.asarray(metadata["intrinsics"][int(cam_idx)], dtype=np.float64)
    width, height = metadata["WH"]
    fps = metadata.get("fps")
    return intrinsic, w2c, int(width), int(height), fps


def project_points(points, intrinsic, w2c, image_width, image_height, min_depth=1e-6):
    if points.size == 0:
        return []
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.shape[0] == 0:
        return []

    points_h = np.concatenate(
        [points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    cam = (w2c @ points_h.T).T[:, :3]
    z = cam[:, 2]
    valid = np.isfinite(cam).all(axis=1) & (z > min_depth)
    cam = cam[valid]
    z = z[valid]
    if cam.shape[0] == 0:
        return []

    u = intrinsic[0, 0] * cam[:, 0] / z + intrinsic[0, 2]
    v = intrinsic[1, 1] * cam[:, 1] / z + intrinsic[1, 2]
    uv = np.stack([u, v], axis=1)
    in_frame = (
        np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < image_width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < image_height)
    )
    return np.rint(uv[in_frame]).astype(np.int32).tolist()


def stride_points(points, point_stride):
    point_stride = max(1, int(point_stride))
    if points.size == 0 or point_stride == 1:
        return points
    return points[::point_stride]


def select_points(points, num_original, num_surface, num_all, mode, point_stride=1):
    if mode == "original":
        points = points[: int(num_original)]
    elif mode == "surface":
        points = points[: int(num_surface)]
    else:
        points = points[: int(num_all)]
    return stride_points(points, point_stride)


class NpzCache:
    def __init__(self, max_items=64):
        self.max_items = max(1, int(max_items))
        self.cache = {}

    def load(self, path):
        path = Path(path)
        mtime_ns = path.stat().st_mtime_ns
        key = str(path)
        cached = self.cache.get(key)
        if cached is not None and cached[0] == mtime_ns:
            return cached[1]

        with np.load(path, allow_pickle=False) as data:
            loaded = {name: data[name].copy() for name in data.files}
        self.cache[key] = (mtime_ns, loaded)
        if len(self.cache) > self.max_items:
            oldest = next(iter(self.cache))
            self.cache.pop(oldest, None)
        return loaded


class ViewerState:
    def __init__(self, args):
        self.args = args
        self.base_path = Path(args.base_path)
        self.case_name = args.case_name
        self.realtime_dir = (
            Path(args.realtime_dir)
            if args.realtime_dir is not None
            else Path(args.experiments_dir) / args.case_name / "realtime"
        )
        self.npz_cache = NpzCache(args.npz_cache_size)

    def manifest_path(self):
        return self.realtime_dir / "manifest.json"

    def iterations_dir(self):
        return self.realtime_dir / "iterations"

    def first_seen_dir(self):
        return self.realtime_dir / "first_seen"

    def latest_path(self):
        return self.realtime_dir / "latest_window.npz"

    def iteration_path(self, iteration):
        return self.iterations_dir() / f"iter_{int(iteration):06d}.npz"

    def load_npz(self, path):
        return self.npz_cache.load(path)

    def discover_iterations(self):
        iterations_dir = self.iterations_dir()
        if not iterations_dir.is_dir():
            return []
        iterations = []
        for path in iterations_dir.glob("iter_*.npz"):
            try:
                iterations.append(int(path.stem.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        return sorted(iterations)

    def load_manifest(self):
        manifest = {}
        if self.manifest_path().exists():
            with open(self.manifest_path(), "r") as f:
                manifest = json.load(f)

        iterations = self.discover_iterations()
        latest_iteration = manifest.get("latest_iteration")
        latest_path = self.latest_path()
        latest_data = None
        if latest_path.exists():
            latest_data = self.load_npz(latest_path)
            latest_iteration = scalar_int(latest_data["iteration"])
            if not iterations and latest_iteration is not None:
                iterations = [int(latest_iteration)]

        _, _, image_width, image_height, metadata_fps = load_camera(
            self.args.base_path, self.case_name, self.args.cam_idx
        )
        if latest_data is not None:
            window_starts = latest_data["window_starts"].astype(int).tolist()
            segment_len = scalar_int(latest_data["segment_len"])
        else:
            window_starts = manifest.get("window_starts", [])
            segment_len = int(manifest.get("segment_len", 0) or 0)

        return {
            "case_name": self.case_name,
            "latest_iteration": latest_iteration,
            "iterations": iterations,
            "window_starts": window_starts,
            "segment_len": segment_len,
            "fps": manifest.get("fps", metadata_fps),
            "image_width": image_width,
            "image_height": image_height,
            "realtime_dir": str(self.realtime_dir),
        }

    def resolve_iteration_file(self, iteration):
        iteration = int(iteration)
        path = self.iteration_path(iteration)
        if path.exists():
            return path
        latest_path = self.latest_path()
        if latest_path.exists():
            latest = self.load_npz(latest_path)
            if scalar_int(latest["iteration"]) == iteration:
                return latest_path
        raise FileNotFoundError(f"iteration snapshot not found: {iteration}")

    def image_path(self, cam_idx, frame_idx):
        path = (
            self.base_path
            / self.case_name
            / "color"
            / str(int(cam_idx))
            / f"{int(frame_idx)}.png"
        )
        if not path.exists():
            raise FileNotFoundError(f"image not found: {path}")
        return path

    def image_url(self, cam_idx, frame_idx):
        return f"/frame/{int(cam_idx)}/{int(frame_idx)}.png"

    def iteration_summary(self, iteration):
        data = self.load_npz(self.resolve_iteration_file(iteration))
        return {
            "iteration": scalar_int(data["iteration"]),
            "window_starts": data["window_starts"].astype(int).tolist(),
            "frame_indices": data["frame_indices"].astype(int).tolist(),
            "online_frame_indices": data["online_frame_indices"].astype(int).tolist(),
            "segment_len": scalar_int(data["segment_len"]),
        }

    def _tile_from_arrays(
        self,
        data,
        window_idx,
        frame_idx,
        cam_idx,
        point_mode,
        point_stride,
        label_prefix,
        iteration=None,
    ):
        intrinsic, w2c, image_width, image_height, _ = load_camera(
            self.args.base_path, self.case_name, cam_idx
        )
        num_original = scalar_int(data["num_original_points"])
        num_surface = scalar_int(data["num_surface_points"])
        num_all = scalar_int(data["num_all_points"])

        pred_points = data["pred_points"]
        if pred_points.ndim == 4:
            pred = pred_points[int(window_idx), int(frame_idx)]
        else:
            pred = pred_points[int(frame_idx)]
        pred = select_points(
            pred,
            num_original,
            num_surface,
            num_all,
            point_mode,
            point_stride=point_stride,
        )

        gt_points = data.get("gt_object_points")
        if gt_points is not None and gt_points.size > 0:
            if gt_points.ndim == 4:
                gt = gt_points[int(window_idx), int(frame_idx)]
            else:
                gt = gt_points[int(frame_idx)]
            vis = data.get("object_visibilities")
            if vis is not None and vis.size > 0:
                if vis.ndim == 3:
                    visible = vis[int(window_idx), int(frame_idx)].astype(bool)
                else:
                    visible = vis[int(frame_idx)].astype(bool)
                gt = gt[visible]
            gt = stride_points(gt, point_stride)
        else:
            gt = np.empty((0, 3), dtype=np.float32)

        controller_points = data.get("controller_points")
        if controller_points is not None and controller_points.size > 0:
            if controller_points.ndim == 4:
                controller = controller_points[int(window_idx), int(frame_idx)]
            else:
                controller = controller_points[int(frame_idx)]
        else:
            controller = np.empty((0, 3), dtype=np.float32)

        frame_indices = data["frame_indices"]
        if frame_indices.ndim == 2:
            source_frame = int(frame_indices[int(window_idx), int(frame_idx)])
        else:
            source_frame = int(frame_indices[int(frame_idx)])
        window_start = (
            int(data["window_starts"][int(window_idx)])
            if "window_starts" in data
            else scalar_int(data["window_start"])
        )
        if iteration is None:
            iteration = scalar_int(data["iteration"])

        return {
            "iteration": int(iteration),
            "window_start": int(window_start),
            "frame_index": int(source_frame),
            "image_url": self.image_url(cam_idx, source_frame),
            "image_width": int(image_width),
            "image_height": int(image_height),
            "pred_uv": project_points(pred, intrinsic, w2c, image_width, image_height),
            "gt_uv": project_points(gt, intrinsic, w2c, image_width, image_height),
            "controller_uv": project_points(
                controller, intrinsic, w2c, image_width, image_height
            ),
            "label": (
                f"{label_prefix} iter={int(iteration)} "
                f"start={int(window_start)} frame={int(source_frame)}"
            ),
        }

    def frame_bundle(self, iteration, frame_idx, cam_idx, point_mode, point_stride):
        data = self.load_npz(self.resolve_iteration_file(iteration))
        segment_len = scalar_int(data["segment_len"])
        frame_idx = int(frame_idx) % int(segment_len)
        point_stride = max(1, int(point_stride))
        window_starts = data["window_starts"].astype(np.int64)
        windows = []
        for window_idx, start in enumerate(window_starts):
            current = self._tile_from_arrays(
                data=data,
                window_idx=window_idx,
                frame_idx=frame_idx,
                cam_idx=cam_idx,
                point_mode=point_mode,
                point_stride=point_stride,
                label_prefix="iter",
                iteration=scalar_int(data["iteration"]),
            )
            first = None
            first_path = self.first_seen_dir() / f"window_{int(start):06d}.npz"
            if first_path.exists():
                first_data = self.load_npz(first_path)
                first_frame = frame_idx % int(first_data["pred_points"].shape[0])
                first = self._tile_from_arrays(
                    data=first_data,
                    window_idx=0,
                    frame_idx=first_frame,
                    cam_idx=cam_idx,
                    point_mode=point_mode,
                    point_stride=point_stride,
                    label_prefix="first",
                    iteration=scalar_int(
                        first_data["first_iteration"]
                        if "first_iteration" in first_data
                        else first_data["iteration"]
                    ),
                )

            windows.append(
                {
                    "window_index": int(window_idx),
                    "window_start": int(start),
                    "current": current,
                    "first_seen": first,
                }
            )

        return {
            "iteration": scalar_int(data["iteration"]),
            "frame": int(frame_idx),
            "segment_len": int(segment_len),
            "windows": windows,
        }


def parse_query(query):
    return {key: values[-1] for key, values in parse_qs(query).items()}


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        server_version = "PhysTwinHTMLViewer/1.0"

        def log_message(self, fmt, *args):
            if not state.args.quiet:
                super().log_message(fmt, *args)

        def send_bytes(self, payload, content_type, status=HTTPStatus.OK):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def send_json(self, payload, status=HTTPStatus.OK):
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_bytes(data, "application/json", status=status)

        def send_error_json(self, status, message):
            self.send_json({"error": str(message)}, status=status)

        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_query(parsed.query)
            try:
                if parsed.path == "/":
                    self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
                    return
                if parsed.path == "/api/manifest":
                    self.send_json(state.load_manifest())
                    return
                if parsed.path == "/api/iteration":
                    iteration = int(query.get("iteration"))
                    self.send_json(state.iteration_summary(iteration))
                    return
                if parsed.path == "/api/frame_bundle":
                    iteration = int(query.get("iteration"))
                    frame_idx = int(query.get("frame", 0))
                    cam_idx = int(query.get("cam_idx", state.args.cam_idx))
                    point_mode = query.get("point_mode", state.args.point_mode)
                    point_stride = int(query.get("point_stride", state.args.point_stride))
                    if point_mode not in {"original", "surface", "all"}:
                        raise ValueError(f"invalid point_mode: {point_mode}")
                    if point_stride < 1:
                        raise ValueError(f"invalid point_stride: {point_stride}")
                    self.send_json(
                        state.frame_bundle(
                            iteration=iteration,
                            frame_idx=frame_idx,
                            cam_idx=cam_idx,
                            point_mode=point_mode,
                            point_stride=point_stride,
                        )
                    )
                    return
                if parsed.path.startswith("/frame/") and parsed.path.endswith(".png"):
                    parts = parsed.path.split("/")
                    if len(parts) != 4:
                        raise ValueError("expected /frame/<cam>/<frame>.png")
                    cam_idx = int(parts[2])
                    frame_idx = int(Path(parts[3]).stem)
                    image_path = state.image_path(cam_idx, frame_idx)
                    self.send_bytes(image_path.read_bytes(), "image/png")
                    return
                self.send_error_json(HTTPStatus.NOT_FOUND, "not found")
            except FileNotFoundError as exc:
                self.send_error_json(HTTPStatus.NOT_FOUND, exc)
            except Exception as exc:
                self.send_error_json(HTTPStatus.BAD_REQUEST, exc)

    return Handler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", default="data/different_types")
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--experiments_dir", default="experiments_online")
    parser.add_argument("--realtime_dir", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cam_idx", type=int, default=0)
    parser.add_argument(
        "--point_mode",
        choices=("original", "surface", "all"),
        default="surface",
    )
    parser.add_argument("--point_stride", type=int, default=5)
    parser.add_argument("--npz_cache_size", type=int, default=96)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    state = ViewerState(args)
    handler = make_handler(state)
    server = ThreadingHTTPServer((args.host, int(args.port)), handler)
    url = f"http://{args.host}:{int(args.port)}/"
    print(f"[html-viewer] serving {state.realtime_dir}")
    print(f"[html-viewer] open {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[html-viewer] stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
