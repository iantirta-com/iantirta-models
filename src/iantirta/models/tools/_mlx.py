
from functools import lru_cache

from ._deps import _is_package_available, _make_compile_constant


@lru_cache
@_make_compile_constant
def is_mlx_available() -> bool:
    return _is_package_available("mlx")[0]
