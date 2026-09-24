"""Default TileLang language facade.

``tilelang.language`` re-exports the CUDA/MACA dialect so that ``import
tilelang.language as T`` yields the common surface plus CUDA/MACA extensions. Other
backends are reached explicitly via ``tilelang.<backend>.language`` (which build
on ``tilelang.language.common``).
"""

from __future__ import annotations
from tilelang.maca.target import check_maca_availability

if check_maca_availability:
    from tilelang.maca.language import *  # noqa: F401,F403
    from tilelang.maca.language import __all__ as __all__  # noqa: F401

    __tilelang_dialect__ = "maca"
else:
    from tilelang.cuda.language import *  # noqa: F401,F403
    from tilelang.cuda.language import __all__ as __all__  # noqa: F401

    __tilelang_dialect__ = "cuda"

# Imported by name so static type checkers resolve the CUDA-typed signatures
# through this facade (they cannot evaluate the dynamic __all__).
from tilelang.cuda.language import (  # noqa: F401
    Kernel,
    Parallel,
    Unroll,
    atomic_add,
    copy,
    gemm,
    gemm_sp,
    reduce_absmax,
    reduce_max,
    reduce_min,
    unroll,
)
