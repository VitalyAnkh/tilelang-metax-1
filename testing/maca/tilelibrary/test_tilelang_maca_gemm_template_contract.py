from pathlib import Path


def test_maca_dense_template_normalizes_partitioned_fragments_before_gemm():
    repo_root = Path(__file__).resolve().parents[3]
    gemm_header = repo_root / "src" / "tl_templates" / "maca" / "gemm.h"

    source = gemm_header.read_text()

    assert "static CUTE_DEVICE auto remove_swizzle(Layout<Args...> const &layout)" in source
    assert "static CUTE_DEVICE auto remove_swizzle(ComposedLayout<Args...> const &layout)" in source
    assert source.count("CUTE_UNROLL") >= 2
    assert source.count("auto tCrB_view = make_tensor(tCrB.data(), remove_swizzle(tCrB.layout()));") >= 2
    assert source.count("gemm(tiled_mma, tCrA(_, _, k), tCrB_view(_, _, k), acc);") >= 2
    assert "Tensor tCrB_copy_view = thr_copy_B.retile_D(tCrB);" in source
    assert source.count("copy(tiled_copy_B, tCsB(_, _, 0), tCrB_copy_view(_, _, 0));") >= 2
    assert source.count("copy(tiled_copy_B, tCsB(_, _, k + 1), tCrB_copy_view(_, _, k + 1));") >= 2


def test_maca_dense_template_fragment_c_matches_cute_partition_order():
    repo_root = Path(__file__).resolve().parents[3]
    layout_source = repo_root / "src" / "layout" / "gemm_layouts.cc"

    source = layout_source.read_text()
    maca_fragment = source.split("Fragment makeGemmFragmentCMACA", 1)[1].split(
        "Fragment makeGemmFragmentCHopper", 1
    )[0]

    assert (
        "base_layout->Repeat({block_m / warp_m, block_n / warp_n}, true, false)"
        in maca_fragment
    )
    assert (
        "thread_layout->Repeat({warp_m / 16, warp_n / 16}, false, false)"
        in maca_fragment
    )
    assert (
        "base_layout->Repeat({warp_m / 16, warp_n / 16}, false, true)"
        not in maca_fragment
    )
