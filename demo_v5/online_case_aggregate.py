"""Legacy compatibility wrapper for :mod:`demo_v5.chunked_final_data_aggregate`."""
from __future__ import annotations

from demo_v5.chunked_final_data_aggregate import *  # noqa: F401,F403
from demo_v5.chunked_final_data_aggregate import FinalDataAggregateWriter


OnlineAggregateCaseWriter = FinalDataAggregateWriter


__all__ = [
    "FinalDataAggregateWriter",
    "OnlineAggregateCaseWriter",
    "build_aggregate_case_from_chunk_cases",
    "migrate_legacy_online_static_case",
]
