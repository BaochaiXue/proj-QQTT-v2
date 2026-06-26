"""Legacy compatibility wrapper for :mod:`demo_v5.chunked_final_data_output`."""
from __future__ import annotations

from demo_v5.chunked_final_data_output import *  # noqa: F401,F403
from demo_v5.chunked_final_data_output import ChunkedFinalDataWriter


DemoV5OnlineOutputWriter = ChunkedFinalDataWriter
DemoV4OnlineOutputWriter = ChunkedFinalDataWriter


__all__ = [
    "DemoV4OnlineOutputWriter",
    "DemoV5OnlineOutputWriter",
    "ChunkedFinalDataWriter",
    "TIME_KEYS",
    "STATIC_KEYS",
    "atomic_json_dump",
    "atomic_pickle_dump",
    "build_online_chunk",
]
