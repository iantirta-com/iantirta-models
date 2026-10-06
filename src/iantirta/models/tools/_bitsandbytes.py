
from functools import lru_cache

from packaging import version

from ._deps import _is_package_available, _make_compile_constant

BITSANDBYTES_MIN_VERSION = "0.46.1"

@lru_cache
@_make_compile_constant
def is_bitsandbytes_available(min_version: str = BITSANDBYTES_MIN_VERSION) -> bool:
    is_available, bitsandbytes_version = _is_package_available("bitsandbytes", return_version=True)
    return is_available and version.parse(bitsandbytes_version) >= version.parse(min_version)
