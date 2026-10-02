
from ._deps import _make_compile_constant, _is_package_available
from functools import lru_cache

import packaging.version
from packaging import version

import logging

logger = logging.getLogger(__name__)


@lru_cache
@_make_compile_constant
def is_torch_available() -> bool:
    try:
        is_available, torch_version = _is_package_available(
            "torch", return_version=True
        )
        parsed_version = version.parse(torch_version)
        if is_available and parsed_version < version.parse("2.5.0"):
            logger.warning_once(
                "Disabling PyTorch because PyTorch >= 2.5 "
                f"is required but found {torch_version}"
            )
        return is_available and (
            version.parse(torch_version) >= version.parse("2.5.0")
        )
    except packaging.version.InvalidVersion:
        return False


_is_torch_available = False
if is_torch_available():
    _is_torch_available = True
