"""Small file-loading helpers shared by Demo v6.1 tools."""
from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Any


def load_pickle(path: str | Path) -> Any:
    """Load a pickle artifact from disk."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a UTF-8 JSON object from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = ["load_json", "load_pickle"]
