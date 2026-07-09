"""Shared JSON Lines (.jsonl) reader for demo_v6_2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read every non-blank row of a .jsonl file as a dict.

    Blank lines are skipped and each remaining line is parsed with ``json.loads``;
    rows that fail to parse (or are not dict-convertible) are skipped. A missing
    file yields an empty list.
    """
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(dict(json.loads(text)))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return rows
