
from ._deps import _make_compile_constant, _is_package_available
from functools import lru_cache


@lru_cache
@_make_compile_constant
def is_mlx_available() -> bool:
    return _is_package_available("mlx")[0]


_is_mlx_available = False
if is_mlx_available():
    _is_mlx_available = True
