
from functools import lru_cache

from iantirta.models.tools._deps import _is_package_available, _make_compile_constant


@lru_cache
@_make_compile_constant
def is_nagisa_available() -> bool:
    return _is_package_available("nagisa")[0]


@lru_cache
@_make_compile_constant
def is_soynlp_available() -> bool:
    return _is_package_available("soynlp")[0]
