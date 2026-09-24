from __future__ import annotations

from tvm.target import Target

from tilelang.backend.execution_backend import ExecutionBackendSpec


def _is_cutedsl_target(target: Target) -> bool:
    return target.kind.name == "maca" and "cutedsl" in target.keys


def _is_plain_maca_target(target: Target) -> bool:
    return target.kind.name == "maca" and "cutedsl" not in target.keys


def _is_mcrtc_available() -> bool:
    try:
        from tilelang.jit.adapter.mcrtc import is_mcrtc_available
    except ImportError:
        return False
    return bool(is_mcrtc_available)


def _is_cutedsl_available() -> bool:
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
    except ImportError:
        return False
    return True


MACA_EXECUTION_BACKENDS = [
    ExecutionBackendSpec(
        "tvm_ffi",
        supports_target=_is_plain_maca_target,
        enable_host_codegen=True,
        enable_device_compile=True,
        supports_callee_allocated_outputs=True,
    ),
    ExecutionBackendSpec("mcrtc", is_available=_is_mcrtc_available, supports_target=_is_plain_maca_target),
    ExecutionBackendSpec("cython", supports_target=_is_plain_maca_target),
    ExecutionBackendSpec("cutedsl", is_available=_is_cutedsl_available, supports_target=_is_cutedsl_target),
]
