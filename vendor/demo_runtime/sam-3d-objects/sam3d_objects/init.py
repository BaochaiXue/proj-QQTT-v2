# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Local runtime defaults for the FuturePhysTwin SAM3D environment.

The upstream package imports this module at package import time. Keep it small:
only set defaults that avoid incompatible optional CUDA packages while preserving
the existing Python/Torch/CUDA stack.
"""

from __future__ import annotations

import os


os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

