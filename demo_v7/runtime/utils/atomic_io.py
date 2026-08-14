"""Atomic file writers shared by Demo v6.2 tools and online consumers."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
import pickle
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping


@contextmanager
def atomic_open(path: str | Path) -> Iterator[BinaryIO]:
    """Open a fsync'd temp file that replaces ``path`` on successful exit.

    Readers polling ``path`` never see partial bytes, and the finished file
    survives a power loss once the context exits.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(target.name + ".tmp")
    with tmp_path.open("wb") as handle:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, target)


def atomic_pickle_dump(value: Any, path: str | Path) -> None:
    """Write a pickle through a temp file so readers never see partial bytes."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(target.name + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, target)


def atomic_json_dump(value: Mapping[str, Any], path: str | Path) -> None:
    """Write JSON through a temp file so readers can poll safely."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(target.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, target)
