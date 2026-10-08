from functools import lru_cache
from packaging import version

from ._deps import _is_package_available, _make_compile_constant
from ._torch import is_torch_available

ACCELERATE_MIN_VERSION = "1.1.0"


@lru_cache
@_make_compile_constant
def is_accelerate_available(min_version: str = ACCELERATE_MIN_VERSION) -> bool:
    if not is_torch_available():
        return False
    is_available, accelerate_version = _is_package_available("accelerate", return_version=True)
    return is_available and version.parse(accelerate_version) >= version.parse(min_version)
