import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GEMM_REGRESSION = REPO_ROOT / "examples/maca/gemm/regression_example_gemm.py"

EXPECTED_BENCH_GEMM_CONFIG = {
    "block_M": 128,
    "block_N": 128,
    "block_K": 128,
    "threads": 256,
    "num_stages": 0,
}

EXPECTED_BENCH_GEMM_CASES = {
    ("bench_gemm_m1664_n1024_k262144", 1664, 1024, 262144),
    ("bench_gemm_m4096_n8192_k8192", 4096, 8192, 8192),
    ("bench_gemm_m4096_n8192_k28672", 4096, 8192, 28672),
    ("bench_gemm_m8192_n1024_k8192", 8192, 1024, 8192),
}


def _literal_assignment(module, name):
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        for target in statement.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(statement.value)
    raise AssertionError(f"missing literal assignment for {name}")


def test_maca_regression_bench_gemm_cases_cover_optimization_shapes():
    module = ast.parse(GEMM_REGRESSION.read_text())

    config = _literal_assignment(module, "_BENCH_GEMM_CONFIG")
    cases = _literal_assignment(module, "_BENCH_GEMM_CASES")

    actual_config = {key: config[key] for key in EXPECTED_BENCH_GEMM_CONFIG}
    actual_cases = {(case["name"], case["M"], case["N"], case["K"]) for case in cases}

    assert actual_config == EXPECTED_BENCH_GEMM_CONFIG
    assert actual_cases == EXPECTED_BENCH_GEMM_CASES


def test_maca_regression_bench_gemm_cases_have_discoverable_wrappers():
    module = ast.parse(GEMM_REGRESSION.read_text())
    function_names = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
    cases = _literal_assignment(module, "_BENCH_GEMM_CASES")

    expected_wrappers = {f"regression_{case['name']}" for case in cases}

    assert expected_wrappers <= function_names
