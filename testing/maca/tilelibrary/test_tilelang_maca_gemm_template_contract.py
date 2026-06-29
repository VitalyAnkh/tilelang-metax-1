from pathlib import Path


def test_maca_dense_template_normalizes_partitioned_fragments_before_gemm():
    repo_root = Path(__file__).resolve().parents[3]
    gemm_header = repo_root / "src" / "tl_templates" / "maca" / "gemm.h"

    source = gemm_header.read_text()
    normalized = " ".join(source.split())

    assert "static CUTE_DEVICE auto remove_swizzle(Layout<Args...> const &layout)" in normalized
    assert "static CUTE_DEVICE auto remove_swizzle(ComposedLayout<Args...> const &layout)" in normalized
    assert source.count("CUTE_UNROLL") >= 2
    assert source.count("auto tCrB_view = make_tensor(tCrB.data(), remove_swizzle(tCrB.layout()));") >= 2
    assert source.count("gemm(tiled_mma, tCrA(_, _, k), tCrB_view(_, _, k), acc);") >= 2
    assert "Tensor tCrB_copy_view = thr_copy_B.retile_D(tCrB);" in source
    assert source.count("copy(tiled_copy_B, tCsB(_, _, 0), tCrB_copy_view(_, _, 0));") >= 2
    assert source.count("copy(tiled_copy_B, tCsB(_, _, k + 1), tCrB_copy_view(_, _, k + 1));") >= 2


def test_maca_dense_template_uses_maca_cute_composed_layout_accessor():
    repo_root = Path(__file__).resolve().parents[3]
    gemm_header = repo_root / "src" / "tl_templates" / "maca" / "gemm.h"

    source = gemm_header.read_text()
    composed_overload = source.split("remove_swizzle(ComposedLayout<Args...> const &layout)", 1)[1].split(
        "\n  }\n\n  CUTE_DEVICE static void body",
        1,
    )[0]

    assert "layout.layout_b()" not in source
    assert "layout.layout_fn()" in source
    assert "return layout;" not in composed_overload


def test_maca_dense_template_remains_opt_in_by_default():
    repo_root = Path(__file__).resolve().parents[3]
    gemm_mma = repo_root / "tilelang" / "maca" / "op" / "gemm" / "gemm_mma.py"

    source = gemm_mma.read_text()

    assert "def _get_maca_gemm_use_template(default: bool = False) -> bool:" in source
    assert "use_template = _get_maca_gemm_use_template(default=False)" in source
    assert source.count("use_template = _get_maca_gemm_use_template(default=False)") >= 2


def test_maca_dense_template_fragment_c_matches_cute_partition_order():
    repo_root = Path(__file__).resolve().parents[3]
    layout_source = repo_root / "src" / "layout" / "gemm_layouts.cc"

    source = layout_source.read_text()
    maca_fragment = source.split("Fragment makeGemmFragmentCMACA", 1)[1].split("Fragment makeGemmFragmentCHopper", 1)[0]

    assert "base_layout->Repeat({block_m / warp_m, block_n / warp_n}, true, false)" in maca_fragment
    assert "thread_layout->Repeat({warp_m / 16, warp_n / 16}, false, false)" in maca_fragment
    assert "base_layout->Repeat({warp_m / 16, warp_n / 16}, false, true)" not in maca_fragment


def test_maca_wsm_template_declares_supported_contract():
    repo_root = Path(__file__).resolve().parents[3]
    wsm_header = repo_root / "src" / "tl_templates" / "maca" / "gemm_wsm.h"

    source = wsm_header.read_text()

    assert "static_assert(!trans_A && !trans_B" in source
    assert "static_assert(num_warp_m == 1 && num_warp_n == 1" in source
    assert "static_assert(kPack == 8" in source
    assert "static_assert(AStrideElements % 8 == 0" in source
    assert "static_assert(sizeof(A_type) == 2 && sizeof(B_type) == 2" in source
    assert "static_assert(sizeof(C_type) == 4" in source


def test_maca_wsm_lowering_names_workspace_size():
    repo_root = Path(__file__).resolve().parents[3]
    gemm_mma = repo_root / "tilelang" / "maca" / "op" / "gemm" / "gemm_mma.py"

    source = gemm_mma.read_text()

    assert "MACA_WSM_STAGE_BYTES = 0x4000" in source
    assert "MACA_WSM_STAGE_COUNT = 4" in source
    assert "MACA_WSM_WORKSPACE_BYTES = MACA_WSM_STAGE_BYTES * MACA_WSM_STAGE_COUNT" in source
    assert "T.alloc_shared((MACA_WSM_WORKSPACE_BYTES,)" in source


def test_maca_wsm_lowering_falls_back_for_unsupported_contracts():
    repo_root = Path(__file__).resolve().parents[3]
    gemm_mma = repo_root / "tilelang" / "maca" / "op" / "gemm" / "gemm_mma.py"

    source = gemm_mma.read_text()

    assert "def _can_use_maca_gemm_wsm(" in source
    assert "not trans_a" in source
    assert "not trans_b" in source
    assert "num_warp_m == 1" in source
    assert "num_warp_n == 1" in source
    assert "k_pack == 8" in source
    assert "a_source_stride % 8 == 0" in source
    assert 'consumer_surface = "direct_tl_gemm_ss"' in source
