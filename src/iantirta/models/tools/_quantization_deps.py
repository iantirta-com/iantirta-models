
from functools import lru_cache
from packaging import version

from ._deps import _is_package_available, _make_compile_constant

HQQ_MIN_VERSION = "0.2.1"


@lru_cache
@_make_compile_constant
def is_hqq_available(min_version: str = HQQ_MIN_VERSION) -> bool:
    is_available, hqq_version = _is_package_available("hqq", return_version=True)
    return is_available and version.parse(hqq_version) >= version.parse(min_version)

@lru_cache
@_make_compile_constant
def is_optimum_available() -> bool:
    return _is_package_available("optimum")[0]


@lru_cache
@_make_compile_constant
def is_optimum_quanto_available():
    return is_optimum_available() and _is_package_available("optimum.quanto")[0]


@lru_cache
def is_quanto_greater(library_version: str, accept_dev: bool = False) -> bool:
    """
    Accepts a library version and returns True if the current version of the library is greater than or equal to the
    given version. If `accept_dev` is True, it will also accept development versions (e.g. 2.7.0.dev20250320 matches
    2.7.0).
    """
    if not is_optimum_quanto_available():
        return False

    _, quanto_version = _is_package_available("optimum.quanto", return_version=True)
    if accept_dev:
        return version.parse(version.parse(quanto_version).base_version) > version.parse(library_version)
    else:
        return version.parse(quanto_version) > version.parse(library_version)
        