#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Demo 3 tracking backend availability.")
    parser.add_argument("--backends", type=str, default="all", help="Comma-separated backend names or 'all'.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args(argv)


def _parse_backends(spec: str) -> list[str] | None:
    normalized = str(spec).strip().lower()
    if normalized in {"", "all"}:
        return None
    return [item.strip() for item in normalized.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from qqtt.tracking.registry import check_backend_availability

    availability = check_backend_availability(_parse_backends(str(args.backends)))
    payload = {name: item.to_dict() for name, item in availability.items()}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
