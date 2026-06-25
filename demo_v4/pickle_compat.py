from __future__ import annotations

import copyreg
from pathlib import Path
import pickle
from typing import Any, BinaryIO

import numpy as np


def _reduce_ndarray_legacy_numpy(arr: np.ndarray) -> tuple[Any, ...]:
    contiguous = np.ascontiguousarray(arr)
    return (
        np.ndarray,
        (
            tuple(int(value) for value in contiguous.shape),
            contiguous.dtype,
            contiguous.tobytes(order="C"),
        ),
    )


class LegacyNumpyPickler(pickle.Pickler):
    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[np.ndarray] = _reduce_ndarray_legacy_numpy


def dump_pickle_legacy_numpy(obj: Any, handle: BinaryIO) -> None:
    LegacyNumpyPickler(handle, protocol=pickle.HIGHEST_PROTOCOL).dump(obj)


def atomic_pickle_dump_legacy_numpy(obj: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(target.name + ".tmp")
    with tmp_path.open("wb") as handle:
        dump_pickle_legacy_numpy(obj, handle)
        handle.flush()
        import os

        os.fsync(handle.fileno())
    tmp_path.replace(target)
