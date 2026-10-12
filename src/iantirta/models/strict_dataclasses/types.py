from collections.abc import Callable
from typing import (
    Annotated,
    Any,
    ForwardRef,
    Literal,
    TypeVar,
    Union,
    get_args,
    get_origin,
    overload,
)

try:
    # Python 3.11+
    from typing import NotRequired, Required  # type: ignore
except ImportError:
    try:
        # In case typing_extensions is installed
        from typing_extensions import NotRequired, Required  # type: ignore
    except ImportError:
        # Fallback: create dummy types that will never match
        Required = type("Required", (), {})  # type: ignore
        NotRequired = type("NotRequired", (), {})  # type: ignore


_TYPED_DICT_DEFAULT_VALUE = object()  # used as default value in TypedDict fields (to distinguish from None)


ValidatorT = Callable[[Any], None]
T = TypeVar("T")
TypedDictT = TypeVar("TypedDictT", bound=dict[str, Any])
