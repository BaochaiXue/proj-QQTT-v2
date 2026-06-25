from __future__ import annotations

import json
import os
from pathlib import Path
import pickle


def atomic_pickle_dump(value, path):
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, target)


def atomic_json_dump(value, path):
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True); handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, target)
