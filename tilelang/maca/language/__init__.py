"""MACA language dialect: common TileLang plus MACA extensions."""

from __future__ import annotations

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL
from tilelang.language.annotations import annotate_l2_hit_ratio, annotate_min_blocks_per_sm  # noqa: F401
from tilelang.language.builtin import (  # noqa: F401
    annotate_consumer_reg_alloc,
    get_lane_idx,
    get_warp_idx,
    get_warp_idx_sync,
    annotate_producer_reg_dealloc,
    deallocate_tmem,
    dec_max_nreg,
    disable_warp_group_reg_alloc,
    get_warp_group_idx,
    inc_max_nreg,
    increase_descriptor_offset,
    ldg128,
    ldg256,
    ldg32,
    ldg64,
    lds128,
    lds32,
    lds64,
    match_all_sync,
    match_any_sync,
    set_max_nreg,
    shuffle_elect,
    stg128,
    stg256,
    stg32,
    stg64,
    sts128,
    sts32,
    sts64,
)
from tilelang.language.copy_op import maca_async_copy  # noqa: F401
from tilelang.language.kernel import CUDASourceCodeKernel  # noqa: F401

from tilelang.cuda.language.kernel import Kernel  # noqa: F401
from tilelang.cuda.language.reduce_op import reduce_absmax, reduce_max, reduce_min  # noqa: F401
from tilelang.language.math_intrinsics import (  # noqa: F401
    __cos,
    __exp,
    __exp10,
    __log,
    __log2,
    __log10,
    __sin,
    __tan,
    fast_rcp,
    ieee_add,
    ieee_fdiv,
    ieee_fmaf,
    ieee_frcp,
    ieee_frsqrt,
    ieee_fsqrt,
    ieee_mul,
    ieee_sub,
)
from tilelang.cuda.language.copy_op import copy, im2col  # noqa: F401
from tilelang.cuda.language.gemm_op import gemm, gemm_sp  # noqa: F401
from tilelang.cuda.language.atomic import atomic_add  # noqa: F401
from tilelang.cuda.language.loop import Parallel, Unroll, unroll  # noqa: F401
from .intrinsics import *  # noqa: F401,F403
from .intrinsics import __all__ as _INTRINSICS_ALL
from tilelang.cuda.language.math import *  # noqa: F401,F403
from tilelang.cuda.language.math import __all__ as _MATH_ALL
from .pdl import *  # noqa: F401,F403
from .pdl import __all__ as _PDL_ALL
from .print import *  # noqa: F401,F403
from .print import __all__ as _PRINT_ALL
from .random import *  # noqa: F401,F403
from .random import __all__ as _RANDOM_ALL
from .tir import *  # noqa: F401,F403
from .tir import __all__ as _TIR_ALL
from .warpgroup import *  # noqa: F401,F403
from .warpgroup import __all__ as _WARPGROUP_ALL
from .barrier import *  # noqa: F401,F403
from .barrier import __all__ as _BARRIER

_MACA_API_ALL = (
    "CUDASourceCodeKernel",
    "Kernel",
    "__cos",
    "__exp",
    "__exp10",
    "__log",
    "__log2",
    "__log10",
    "__sin",
    "__tan",
    "fast_rcp",
    "get_lane_idx",
    "get_warp_idx",
    "get_warp_idx_sync",
    "ieee_add",
    "ieee_fdiv",
    "ieee_fmaf",
    "ieee_frcp",
    "ieee_frsqrt",
    "ieee_fsqrt",
    "ieee_mul",
    "ieee_sub",
    "reduce_absmax",
    "reduce_max",
    "reduce_min",
    "Parallel",
    "Unroll",
    "atomic_add",
    "copy",
    "gemm",
    "gemm_sp",
    "im2col",
    "unroll",
    "annotate_consumer_reg_alloc",
    "annotate_l2_hit_ratio",
    "annotate_min_blocks_per_sm",
    "annotate_producer_reg_dealloc",
    "deallocate_tmem",
    "dec_max_nreg",
    "disable_warp_group_reg_alloc",
    "get_warp_group_idx",
    "inc_max_nreg",
    "increase_descriptor_offset",
    "ldg128",
    "ldg256",
    "ldg32",
    "ldg64",
    "lds128",
    "lds32",
    "lds64",
    "match_all_sync",
    "match_any_sync",
    "set_max_nreg",
    "shuffle_elect",
    "stg128",
    "stg256",
    "stg32",
    "stg64",
    "sts128",
    "sts32",
    "sts64",
    "maca_async_copy",
)

__tilelang_dialect__ = "maca"
__all__ = tuple(
    dict.fromkeys(
        (
            *_COMMON_ALL,
            *_MACA_API_ALL,
            *_INTRINSICS_ALL,
            *_MATH_ALL,
            *_PDL_ALL,
            *_PRINT_ALL,
            *_RANDOM_ALL,
            *_TIR_ALL,
            *_WARPGROUP_ALL,
            *_BARRIER,
        )
    )
)

del _COMMON_ALL, _MACA_API_ALL, _INTRINSICS_ALL, _MATH_ALL, _PDL_ALL, _PRINT_ALL, _RANDOM_ALL, _TIR_ALL, _WARPGROUP_ALL, _BARRIER
