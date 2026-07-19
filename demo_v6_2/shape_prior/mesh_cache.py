"""Persistent reuse of the canonical shape-prior ``object.glb`` mesh (v1).

The only cached asset is the raw SAM3D mesh (``shape/object.glb``). Alignment,
metric scale, surface/interior points and every downstream product are still
recomputed per run from the current frame-0 observation, so caching the mesh
never changes the alignment algorithm or any downstream data contract.

Cache identity is the operator-supplied ``object`` id (a specific physical
instance plus asset version), NOT the SAM3.1 ``object_prompt``. Changing the
prompt never invalidates an existing entry; a new asset uses a new ``object``
version instead.

Entry layout::

    <cache_root>/schema_v1/<object_id>/
        object.glb
        manifest.json

Status is resolved from config + disk before the shape-prior workers pre-warm:
``disabled`` (object is null), ``miss`` (no entry) or ``hit`` (a complete,
valid entry). A present-but-invalid entry raises immediately -- there is no
"cache failed, regenerate on the fly" path.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterator


SCHEMA_VERSION = 1
SCHEMA_DIR_NAME = "schema_v1"
MESH_FILENAME = "object.glb"
MANIFEST_FILENAME = "manifest.json"
GENERATOR_TYPE = "sam3d"
ASSET_STATUS = "generated"

CACHE_STATUS_DISABLED = "disabled"
CACHE_STATUS_MISS = "miss"
CACHE_STATUS_HIT = "hit"

# Object-id strings that look like a disabled sentinel; rejected so a stray
# "none"/"null"/"" is never silently treated as a real cache identity. Only a
# real YAML ``null`` (Python ``None``) disables the cache.
_RESERVED_OBJECT_IDS = frozenset({"none", "null", "nil", "false", "true"})
_FORBIDDEN_OBJECT_ID_SUBSTRINGS = ("/", "\\", "..", "\x00")
_OBJECT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ShapePriorMeshCacheError(RuntimeError):
    """Raised on cache corruption, publish conflict, or invalid configuration."""


def normalize_object_id(value: Any) -> str | None:
    """Return the validated cache identity, or ``None`` when the cache is off.

    ``None`` (YAML ``null``) disables the cache. Any other value must be a safe
    single directory-name identity: non-empty, no path separators, no ``..``,
    not absolute, no whitespace, and not a disabled-looking sentinel string.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ShapePriorMeshCacheError(
            "shape_prior.object must be a string or YAML null"
        )
    object_id = value
    stripped = object_id.strip()
    if not stripped:
        raise ShapePriorMeshCacheError(
            "shape_prior.object must be YAML null to disable the cache, not an "
            "empty string"
        )
    if object_id != stripped or any(ch.isspace() for ch in object_id):
        raise ShapePriorMeshCacheError(
            f"shape_prior.object {object_id!r} must not contain whitespace"
        )
    if object_id.lower() in _RESERVED_OBJECT_IDS:
        raise ShapePriorMeshCacheError(
            f"shape_prior.object {object_id!r} is reserved; use YAML null to "
            "disable the cache or a concrete instance id like 'sloth_plush_01_v1'"
        )
    if any(token in object_id for token in _FORBIDDEN_OBJECT_ID_SUBSTRINGS):
        raise ShapePriorMeshCacheError(
            f"shape_prior.object {object_id!r} must be a single safe directory "
            "name (no '/', '\\', '..', or null bytes)"
        )
    if _OBJECT_ID_PATTERN.fullmatch(object_id) is None:
        raise ShapePriorMeshCacheError(
            f"shape_prior.object {object_id!r} may contain only ASCII letters, "
            "digits, '.', '_', and '-', and must start with a letter or digit"
        )
    if os.path.isabs(object_id) or Path(object_id).name != object_id:
        raise ShapePriorMeshCacheError(
            f"shape_prior.object {object_id!r} must be a single directory name, "
            "not a path"
        )
    return object_id


def validate_cache_root(cache_root: str | Path, *, forbidden_root: Path) -> Path:
    """Return the resolved cache root, or raise if it lives under the run output.

    Keeping the persistent cache outside ``forbidden_root`` (the run's output /
    base path) guarantees the run's output-cleanup never deletes cache entries.
    """
    resolved = Path(cache_root).expanduser().resolve()
    forbidden = Path(forbidden_root).expanduser().resolve()
    if resolved == forbidden or forbidden in resolved.parents:
        raise ShapePriorMeshCacheError(
            f"shape_prior.cache_root {resolved} must not live under the run "
            f"output directory {forbidden}"
        )
    return resolved


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 of a file's contents."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_mesh_glb(path: str | Path) -> None:
    """Raise ``ShapePriorMeshCacheError`` unless ``path`` is a usable mesh GLB.

    Uses the same loader the downstream align/sample stages use, so a GLB that
    passes here is guaranteed loadable by them. No bbox/scale/watertight/metric
    thresholds are applied: the raw SAM3D mesh has not yet been metric-aligned
    to the current observation.
    """
    mesh_path = Path(path)
    if not mesh_path.is_file():
        raise ShapePriorMeshCacheError(f"cache mesh missing: {mesh_path}")
    if mesh_path.stat().st_size == 0:
        raise ShapePriorMeshCacheError(f"cache mesh is empty: {mesh_path}")
    import trimesh  # noqa: PLC0415

    from demo_v6_2.utils.align_util import as_mesh  # noqa: PLC0415

    try:
        mesh = as_mesh(trimesh.load(str(mesh_path), force="mesh"))
    except Exception as exc:  # noqa: BLE001 - any loader failure is corruption
        raise ShapePriorMeshCacheError(
            f"cache mesh failed to load: {mesh_path}: {exc}"
        ) from exc
    if mesh is None or getattr(mesh, "vertices", None) is None:
        raise ShapePriorMeshCacheError(f"cache mesh has no geometry: {mesh_path}")
    if len(mesh.vertices) < 1:
        raise ShapePriorMeshCacheError(f"cache mesh has no vertices: {mesh_path}")
    if getattr(mesh, "faces", None) is None or len(mesh.faces) < 1:
        raise ShapePriorMeshCacheError(f"cache mesh has no faces: {mesh_path}")
    import numpy as np  # noqa: PLC0415

    if not np.isfinite(np.asarray(mesh.vertices, dtype=np.float64)).all():
        raise ShapePriorMeshCacheError(
            f"cache mesh has non-finite vertices: {mesh_path}"
        )


@dataclass(frozen=True)
class CacheResolution:
    """Startup-resolved cache decision for one run (before prewarm)."""

    status: str  # disabled | miss | hit
    object_id: str | None
    cache_root: Path | None
    entry_dir: Path | None
    mesh_path: Path | None
    manifest: dict[str, Any] | None

    @property
    def enabled(self) -> bool:
        """Return whether the cache is enabled for this run."""
        return self.status != CACHE_STATUS_DISABLED

    @property
    def hit(self) -> bool:
        """Return whether a valid entry will be reused (skip mesh generation)."""
        return self.status == CACHE_STATUS_HIT


class ShapePriorMeshCache:
    """Resolve, validate, publish and materialize the canonical ``object.glb``."""

    def __init__(
        self,
        *,
        object_id: str | None,
        cache_root: str | Path,
    ) -> None:
        """Initialize a cache handle from validated identity and root values."""
        self.object_id = normalize_object_id(object_id)
        self.cache_root = Path(cache_root).expanduser().resolve()

    @property
    def schema_dir(self) -> Path:
        """Return the schema-versioned root under the cache root."""
        return self.cache_root / SCHEMA_DIR_NAME

    @property
    def entry_dir(self) -> Path | None:
        """Return the entry directory for this object id, or None if disabled."""
        if self.object_id is None:
            return None
        return self.schema_dir / self.object_id

    def _ensure_publishable_schema_dir(self) -> None:
        """Create and write-probe the schema directory for an enabled miss."""
        if self.object_id is None:
            raise ShapePriorMeshCacheError(
                "cannot prepare publication with a disabled cache"
            )
        try:
            self.schema_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ShapePriorMeshCacheError(
                f"cannot create shape-prior cache schema directory "
                f"{self.schema_dir}: {exc}"
            ) from exc
        if not self.schema_dir.is_dir():
            raise ShapePriorMeshCacheError(
                f"cache schema path exists but is not a directory: {self.schema_dir}"
            )

        probe_path: Path | None = None
        try:
            fd, probe_name = tempfile.mkstemp(
                prefix=".write-probe-",
                dir=str(self.schema_dir),
            )
            probe_path = Path(probe_name)
            os.close(fd)
        except OSError as exc:
            raise ShapePriorMeshCacheError(
                f"shape-prior cache schema directory is not writable: "
                f"{self.schema_dir}: {exc}"
            ) from exc
        finally:
            if probe_path is not None and probe_path.exists():
                probe_path.unlink()

    @contextmanager
    def _publish_lock(self) -> Iterator[None]:
        """Serialize publication for this object id across local processes."""
        if self.object_id is None:
            raise ShapePriorMeshCacheError("cannot lock a disabled cache")
        lock_path = self.schema_dir / f".{self.object_id}.lock"
        try:
            lock_handle = lock_path.open("a+b")
        except OSError as exc:
            raise ShapePriorMeshCacheError(
                f"cannot open shape-prior cache publish lock {lock_path}: {exc}"
            ) from exc
        with lock_handle:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            except OSError as exc:
                raise ShapePriorMeshCacheError(
                    f"cannot acquire shape-prior cache publish lock {lock_path}: {exc}"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _load_manifest(self, entry_dir: Path) -> dict[str, Any]:
        """Load and structurally validate the entry manifest, or raise corrupt."""
        manifest_path = entry_dir / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise ShapePriorMeshCacheError(
                f"cache entry missing manifest: {manifest_path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ShapePriorMeshCacheError(
                f"cache manifest is not valid JSON: {manifest_path}: {exc}"
            ) from exc
        if not isinstance(manifest, dict):
            raise ShapePriorMeshCacheError(
                f"cache manifest must be a JSON object: {manifest_path}"
            )
        schema_version = manifest.get("schema_version")
        if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
            raise ShapePriorMeshCacheError(
                f"cache manifest schema_version mismatch at {manifest_path}: "
                f"{schema_version!r} != {SCHEMA_VERSION}"
            )
        if manifest.get("object_id") != self.object_id:
            raise ShapePriorMeshCacheError(
                f"cache manifest object_id {manifest.get('object_id')!r} does not "
                f"match request {self.object_id!r} at {manifest_path}"
            )
        prompt = manifest.get("object_prompt_at_generation")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ShapePriorMeshCacheError(
                f"cache manifest object_prompt_at_generation must be a non-empty "
                f"string at {manifest_path}"
            )
        if manifest.get("asset_status") != ASSET_STATUS:
            raise ShapePriorMeshCacheError(
                f"cache manifest asset_status must be {ASSET_STATUS!r} at "
                f"{manifest_path}"
            )
        if manifest.get("mesh_file") != MESH_FILENAME:
            raise ShapePriorMeshCacheError(
                f"cache manifest mesh_file must be {MESH_FILENAME!r} at {manifest_path}"
            )
        mesh_sha256 = manifest.get("mesh_sha256")
        if (
            not isinstance(mesh_sha256, str)
            or _SHA256_PATTERN.fullmatch(mesh_sha256) is None
        ):
            raise ShapePriorMeshCacheError(
                f"cache manifest mesh_sha256 must be 64 lowercase hex characters "
                f"at {manifest_path}"
            )
        created_at = manifest.get("created_at_utc")
        if not isinstance(created_at, str):
            raise ShapePriorMeshCacheError(
                f"cache manifest created_at_utc must be an ISO-8601 UTC string at "
                f"{manifest_path}"
            )
        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ShapePriorMeshCacheError(
                f"cache manifest created_at_utc is invalid at {manifest_path}: "
                f"{created_at!r}"
            ) from exc
        if created.tzinfo is None or created.utcoffset() != timedelta(0):
            raise ShapePriorMeshCacheError(
                f"cache manifest created_at_utc must include UTC timezone at "
                f"{manifest_path}"
            )
        generator = manifest.get("generator")
        if not isinstance(generator, dict):
            raise ShapePriorMeshCacheError(
                f"cache manifest generator must be an object at {manifest_path}"
            )
        if generator.get("type") != GENERATOR_TYPE:
            raise ShapePriorMeshCacheError(
                f"cache manifest generator.type must be {GENERATOR_TYPE!r} at "
                f"{manifest_path}"
            )
        seed = generator.get("seed")
        if type(seed) is not int:
            raise ShapePriorMeshCacheError(
                f"cache manifest generator.seed must be an integer at {manifest_path}"
            )
        return manifest

    def validate_entry(self, entry_dir: Path) -> dict[str, Any]:
        """Validate a complete entry (manifest + mesh + hash), or raise corrupt."""
        manifest = self._load_manifest(entry_dir)
        mesh_path = entry_dir / MESH_FILENAME
        if not mesh_path.is_file():
            raise ShapePriorMeshCacheError(f"cache mesh missing: {mesh_path}")
        if mesh_path.stat().st_size == 0:
            raise ShapePriorMeshCacheError(f"cache mesh is empty: {mesh_path}")
        actual_sha = sha256_file(mesh_path)
        if actual_sha != str(manifest["mesh_sha256"]):
            raise ShapePriorMeshCacheError(
                f"cache mesh hash mismatch at {mesh_path}: manifest "
                f"{manifest['mesh_sha256']} != actual {actual_sha}"
            )
        validate_mesh_glb(mesh_path)
        return manifest

    def resolve(self) -> CacheResolution:
        """Resolve the cache status from config + disk (raises on corruption).

        Called once at startup before the shape-prior workers pre-warm.
        """
        if self.object_id is None:
            return CacheResolution(
                status=CACHE_STATUS_DISABLED,
                object_id=None,
                cache_root=None,
                entry_dir=None,
                mesh_path=None,
                manifest=None,
            )
        entry_dir = self.entry_dir
        assert entry_dir is not None
        if not entry_dir.exists():
            # A miss is committed to publishing after generation. Probe the
            # destination now so an invalid/unwritable root fails before the
            # expensive SAM3D stage starts.
            self._ensure_publishable_schema_dir()
            return CacheResolution(
                status=CACHE_STATUS_MISS,
                object_id=self.object_id,
                cache_root=self.cache_root,
                entry_dir=entry_dir,
                mesh_path=None,
                manifest=None,
            )
        if not entry_dir.is_dir():
            raise ShapePriorMeshCacheError(
                f"cache entry path exists but is not a directory: {entry_dir}"
            )
        manifest = self.validate_entry(entry_dir)
        return CacheResolution(
            status=CACHE_STATUS_HIT,
            object_id=self.object_id,
            cache_root=self.cache_root,
            entry_dir=entry_dir,
            mesh_path=entry_dir / MESH_FILENAME,
            manifest=manifest,
        )

    def publish(
        self,
        *,
        source_glb: str | Path,
        object_prompt_at_generation: str,
        generator_seed: int,
    ) -> dict[str, Any]:
        """Atomically publish a freshly generated mesh as this object's entry.

        The mesh is validated, hashed and written with its manifest into a temp
        directory under ``schema_v1``; only then is the whole directory renamed
        onto the final object id. Never overwrites an existing entry (conflict
        raises), so a run cannot clobber another asset version.
        """
        if self.object_id is None:
            raise ShapePriorMeshCacheError("cannot publish with a disabled cache")
        if not isinstance(object_prompt_at_generation, str):
            raise ShapePriorMeshCacheError(
                "object_prompt_at_generation must be a string"
            )
        prompt = object_prompt_at_generation.strip()
        if not prompt:
            raise ShapePriorMeshCacheError(
                "object_prompt_at_generation must be non-empty"
            )
        if type(generator_seed) is not int:
            raise ShapePriorMeshCacheError("generator_seed must be an integer")
        source = Path(source_glb)
        validate_mesh_glb(source)
        entry_dir = self.entry_dir
        assert entry_dir is not None
        self._ensure_publishable_schema_dir()
        mesh_sha256 = sha256_file(source)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "object_id": self.object_id,
            "object_prompt_at_generation": prompt,
            "asset_status": ASSET_STATUS,
            "mesh_file": MESH_FILENAME,
            "mesh_sha256": mesh_sha256,
            "created_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "generator": {
                "type": GENERATOR_TYPE,
                "seed": generator_seed,
            },
        }
        with self._publish_lock():
            if entry_dir.exists():
                raise ShapePriorMeshCacheError(
                    f"cache entry already exists, refusing to overwrite: {entry_dir}"
                )
            tmp_dir = Path(
                tempfile.mkdtemp(
                    prefix=f".tmp-{self.object_id}-",
                    dir=str(self.schema_dir),
                )
            )
            try:
                tmp_mesh = tmp_dir / MESH_FILENAME
                shutil.copyfile(source, tmp_mesh)
                if sha256_file(tmp_mesh) != mesh_sha256:
                    raise ShapePriorMeshCacheError(
                        "cache mesh copy hash mismatch during publish"
                    )
                (tmp_dir / MANIFEST_FILENAME).write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                self.validate_entry(tmp_dir)
                try:
                    os.rename(tmp_dir, entry_dir)
                except OSError as exc:
                    raise ShapePriorMeshCacheError(
                        f"cache publish rename failed: {entry_dir}: {exc}"
                    ) from exc
            finally:
                if tmp_dir.exists():
                    _remove_tree(tmp_dir)
        return manifest

    def materialize(self, *, resolution: CacheResolution, dest_glb: str | Path) -> str:
        """Copy the cached mesh into a run-local ``object.glb`` and return its sha.

        The copy goes through a same-directory temp file whose hash is verified
        against the manifest before an atomic replace, so the run is
        self-contained and unaffected by later manual cache edits. No symlink.
        """
        if resolution.status != CACHE_STATUS_HIT:
            raise ShapePriorMeshCacheError(
                "materialize requires a cache hit resolution"
            )
        assert resolution.mesh_path is not None and resolution.manifest is not None
        expected_sha = str(resolution.manifest["mesh_sha256"])
        dest = Path(dest_glb)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".tmp-{MESH_FILENAME}-", dir=str(dest.parent)
        )
        tmp_path = Path(tmp_name)
        try:
            os.close(fd)
            shutil.copyfile(resolution.mesh_path, tmp_path)
            actual_sha = sha256_file(tmp_path)
            if actual_sha != expected_sha:
                raise ShapePriorMeshCacheError(
                    f"materialized cache mesh hash mismatch: manifest {expected_sha} "
                    f"!= actual {actual_sha}"
                )
            os.replace(tmp_path, dest)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return expected_sha


def _remove_tree(path: Path) -> None:
    """Best-effort recursive delete of a temp publish directory."""
    shutil.rmtree(path, ignore_errors=True)


__all__ = [
    "CACHE_STATUS_DISABLED",
    "CACHE_STATUS_HIT",
    "CACHE_STATUS_MISS",
    "MESH_FILENAME",
    "SCHEMA_DIR_NAME",
    "SCHEMA_VERSION",
    "CacheResolution",
    "ShapePriorMeshCache",
    "ShapePriorMeshCacheError",
    "normalize_object_id",
    "sha256_file",
    "validate_cache_root",
    "validate_mesh_glb",
]
