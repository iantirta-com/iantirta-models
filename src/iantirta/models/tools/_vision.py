
from functools import lru_cache

from iantirta.models.tools._deps import _is_package_available, _make_compile_constant


@lru_cache
@_make_compile_constant
def is_vision_available() -> bool:
    try:
        import PIL.Image  # noqa: F401

        return True
    except ImportError:
        return False
