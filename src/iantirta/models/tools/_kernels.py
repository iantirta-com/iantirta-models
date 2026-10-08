
from functools import lru_cache
from packaging import version

from ._deps import _is_package_available, _make_compile_constant

KERNELS_MIN_VERSION = "0.17.0"
KERNELS_MAX_VERSION = "0.18.0"


@lru_cache
@_make_compile_constant
def is_kernels_available(MIN_VERSION: str = KERNELS_MIN_VERSION, MAX_VERSION: str = KERNELS_MAX_VERSION) -> bool:
    is_available, kernels_version = _is_package_available("kernels", return_version=True)
    viable_version = False
    if kernels_version != "N/A":
        viable_version = version.parse(kernels_version) >= version.parse(MIN_VERSION) and version.parse(
            kernels_version
        ) < version.parse(MAX_VERSION)
    return is_available and viable_version
