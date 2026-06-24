import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from qqtt.utils import logger, cfg


TIME_KEYS = (
    "object_points",
    "object_colors",
    "object_visibilities",
    "object_motions_valid",
    "controller_points",
    "asap_object_points_filled",
    "asap_surface_points",
    "asap_interior_points",
)

STATIC_KEYS = (
    "surface_points",
    "interior_points",
)


class OnlineChunkReader:
    """Polls a chunk directory written by the fake or real online tracker."""

    def __init__(self, online_dir, chunk_pattern="chunk_{chunk_id:06d}.pkl"):
        self.online_dir = Path(online_dir)
        self.chunks_dir = self.online_dir / "chunks"
        self.manifest_path = self.online_dir / "manifest.json"
        self.chunk_pattern = chunk_pattern
        self.last_loaded_chunk = -1
        self.last_manifest = None

    @property
    def status(self):
        if self.last_manifest is None:
            return None
        return self.last_manifest.get("status")

    @property
    def is_finished(self):
        return self.status == "finished"

    def read_manifest(self):
        if not self.manifest_path.exists():
            return None
        try:
            with open(self.manifest_path, "r") as f:
                manifest = json.load(f)
        except json.JSONDecodeError:
            logger.warning(
                f"[Online-Reader]: manifest is not valid JSON yet: {self.manifest_path}"
            )
            return None
        self.last_manifest = manifest
        return manifest

    def _chunk_path(self, chunk_id):
        return self.chunks_dir / self.chunk_pattern.format(chunk_id=int(chunk_id))

    def load_new_chunks(self):
        manifest = self.read_manifest()
        if manifest is None:
            return []

        latest_chunk = int(manifest.get("latest_committed_chunk", -1))
        if latest_chunk <= self.last_loaded_chunk:
            return []

        chunks = []
        for chunk_id in range(self.last_loaded_chunk + 1, latest_chunk + 1):
            chunk_path = self._chunk_path(chunk_id)
            if not chunk_path.exists():
                raise FileNotFoundError(
                    f"Manifest committed chunk {chunk_id}, but {chunk_path} is missing"
                )
            with open(chunk_path, "rb") as f:
                chunk = pickle.load(f)
            chunks.append(chunk)
            self.last_loaded_chunk = chunk_id
        return chunks

    def wait_for_manifest(self, poll_sec=1.0):
        while True:
            manifest = self.read_manifest()
            if manifest is not None:
                return manifest
            logger.info(f"[Online-Reader]: waiting for {self.manifest_path}")
            time.sleep(float(poll_sec))


class OnlineFrameBuffer:
    """Growing frame buffer with a RealData-like tensor interface."""

    def __init__(self, static_data_path=None, device=None):
        self.static_data_path = static_data_path
        self.device = cfg.device if device is None else device
        self._arrays = {key: [] for key in TIME_KEYS}
        self._static = {key: None for key in STATIC_KEYS}
        self._source_frame_indices = []
        self._frame_count = 0
        self._synced_frame_count = -1
        self._loaded_any_chunk = False

        self._load_static_fallback(static_data_path)
        self._clear_tensor_attrs()

    @property
    def frame_len(self):
        return int(self._frame_count)

    @property
    def num_frames(self):
        return int(self._frame_count)

    def _clear_tensor_attrs(self):
        self.object_points = None
        self.object_colors = None
        self.original_object_colors = None
        self.object_visibilities = None
        self.object_motions_valid = None
        self.controller_points = None
        self.asap_object_points_filled = None
        self.asap_surface_points = None
        self.asap_interior_points = None
        self.structure_points = None
        self.num_original_points = None
        self.num_surface_points = None
        self.num_all_points = None
        self.source_frame_indices = None

    def _load_static_fallback(self, static_data_path):
        if static_data_path is None:
            return
        static_path = Path(static_data_path)
        if not static_path.exists():
            logger.warning(
                f"[Online-Buffer]: static data path does not exist: {static_path}"
            )
            return

        with open(static_path, "rb") as f:
            data = pickle.load(f)
        for key in STATIC_KEYS:
            value = data.get(key)
            if value is not None:
                self._static[key] = value

    def _set_static_from_chunk(self, chunk):
        for key in STATIC_KEYS:
            value = chunk.get(key)
            if value is not None:
                if self._static[key] is not None and self._static[key].shape != value.shape:
                    raise ValueError(
                        f"Static key {key} changed shape from "
                        f"{self._static[key].shape} to {value.shape}"
                    )
                self._static[key] = value

    def _validate_chunk_range(self, chunk):
        start_frame = int(chunk["start_frame"])
        end_frame = int(chunk["end_frame"])
        if start_frame != self._frame_count:
            raise ValueError(
                "Online chunks must be contiguous from frame 0. "
                f"Expected start_frame={self._frame_count}, got {start_frame}."
            )
        if end_frame <= start_frame:
            raise ValueError(
                f"Invalid chunk frame range [{start_frame}, {end_frame})"
            )
        return start_frame, end_frame

    def _validate_chunk_shapes(self, chunk, chunk_len):
        required = (
            "object_points",
            "object_colors",
            "object_visibilities",
            "object_motions_valid",
            "controller_points",
        )
        for key in required:
            if key not in chunk:
                raise KeyError(f"Online chunk missing required key: {key}")

        for key in TIME_KEYS:
            value = chunk.get(key)
            if value is None:
                continue
            if int(value.shape[0]) != chunk_len:
                raise ValueError(
                    f"Chunk key {key} has {value.shape[0]} frames, expected {chunk_len}"
                )
            if self._arrays[key]:
                prev_shape = self._arrays[key][0].shape[1:]
                if value.shape[1:] != prev_shape:
                    raise ValueError(
                        f"Chunk key {key} changed per-frame shape from "
                        f"{prev_shape} to {value.shape[1:]}"
                    )

    def append_chunks(self, chunks):
        if len(chunks) == 0:
            return 0

        frames_added = 0
        for chunk in chunks:
            start_frame, end_frame = self._validate_chunk_range(chunk)
            chunk_len = end_frame - start_frame
            self._set_static_from_chunk(chunk)
            self._validate_chunk_shapes(chunk, chunk_len)

            source_frame_indices = chunk.get("source_frame_indices")
            if source_frame_indices is None:
                source_frame_indices = np.arange(start_frame, end_frame, dtype=np.int64)
            else:
                source_frame_indices = np.asarray(source_frame_indices, dtype=np.int64)
            if int(source_frame_indices.shape[0]) != chunk_len:
                raise ValueError(
                    "Chunk source_frame_indices has "
                    f"{source_frame_indices.shape[0]} frames, expected {chunk_len}"
                )

            for key in TIME_KEYS:
                value = chunk.get(key)
                if value is not None:
                    self._arrays[key].append(value)
            self._source_frame_indices.append(source_frame_indices)
            self._frame_count = end_frame
            frames_added += chunk_len
            self._loaded_any_chunk = True

        logger.info(
            f"[Online-Buffer]: appended {len(chunks)} chunks, "
            f"frames_added={frames_added}, total_frames={self._frame_count}"
        )
        return frames_added

    def _concat_key(self, key):
        values = self._arrays[key]
        if len(values) == 0:
            return None
        return np.concatenate(values, axis=0)

    def _require_static_points(self):
        surface_points = self._static["surface_points"]
        interior_points = self._static["interior_points"]
        if surface_points is None or interior_points is None:
            raise ValueError(
                "OnlineFrameBuffer needs surface_points and interior_points from "
                "chunks or static_data_path before trainer initialization."
            )
        return surface_points, interior_points

    def sync_to_device(self, device=None):
        if device is not None:
            self.device = device
        if self._frame_count == self._synced_frame_count:
            return False
        if not self._loaded_any_chunk:
            raise RuntimeError("Cannot sync an empty online frame buffer")

        surface_points, interior_points = self._require_static_points()
        object_points = self._concat_key("object_points")
        object_colors = self._concat_key("object_colors")
        object_visibilities = self._concat_key("object_visibilities")
        object_motions_valid = self._concat_key("object_motions_valid")
        controller_points = self._concat_key("controller_points")
        asap_object_points_filled = self._concat_key("asap_object_points_filled")
        asap_surface_points = self._concat_key("asap_surface_points")
        asap_interior_points = self._concat_key("asap_interior_points")
        if len(self._source_frame_indices) > 0:
            source_frame_indices = np.concatenate(self._source_frame_indices, axis=0)
        else:
            source_frame_indices = np.arange(object_points.shape[0], dtype=np.int64)

        for key, value in (
            ("asap_object_points_filled", asap_object_points_filled),
            ("asap_surface_points", asap_surface_points),
            ("asap_interior_points", asap_interior_points),
        ):
            if value is not None and value.shape[0] != object_points.shape[0]:
                raise ValueError(
                    f"Optional online key {key} is only present for "
                    f"{value.shape[0]} of {object_points.shape[0]} frames"
                )

        if asap_object_points_filled is None:
            asap_object_points_filled = object_points
        if asap_surface_points is None:
            asap_surface_points = np.repeat(
                surface_points[None], object_points.shape[0], axis=0
            )
        if asap_interior_points is None:
            asap_interior_points = np.repeat(
                interior_points[None], object_points.shape[0], axis=0
            )

        self.num_original_points = int(object_points.shape[1])
        self.num_surface_points = int(self.num_original_points + surface_points.shape[0])
        self.num_all_points = int(self.num_surface_points + interior_points.shape[0])

        structure_points = np.concatenate(
            [object_points[0], asap_surface_points[0], asap_interior_points[0]], axis=0
        )

        self.structure_points = torch.tensor(
            structure_points, dtype=torch.float32, device=self.device
        )
        self.object_points = torch.tensor(
            object_points, dtype=torch.float32, device=self.device
        )
        self.asap_object_points_filled = torch.tensor(
            asap_object_points_filled, dtype=torch.float32, device=self.device
        )
        self.asap_surface_points = torch.tensor(
            asap_surface_points, dtype=torch.float32, device=self.device
        )
        self.asap_interior_points = torch.tensor(
            asap_interior_points, dtype=torch.float32, device=self.device
        )
        self.original_object_colors = torch.tensor(
            object_colors, dtype=torch.float32, device=self.device
        )
        self.object_colors = self.original_object_colors
        self.object_visibilities = torch.tensor(
            object_visibilities, dtype=torch.bool, device=self.device
        )
        self.object_motions_valid = torch.tensor(
            object_motions_valid, dtype=torch.bool, device=self.device
        )
        self.controller_points = torch.tensor(
            controller_points, dtype=torch.float32, device=self.device
        )
        self.source_frame_indices = torch.tensor(
            source_frame_indices, dtype=torch.long, device=self.device
        )
        self._synced_frame_count = int(self._frame_count)
        return True
