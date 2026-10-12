
from dataclasses import fields, dataclass, MISSING, make_dataclass, field
from functools import wraps, lru_cache
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

from .types import TypedDictT, Required, NotRequired, _TYPED_DICT_DEFAULT_VALUE


def wrap_init_to_accept_kwargs(cls: dataclass):
    # Get the original dataclass-generated __init__
    original_init = cls.__init__

    @wraps(original_init)
    def __init__(self, *args, **kwargs: Any) -> None:
        # Extract only the fields that are part of the dataclass
        dataclass_fields = {f.name for f in fields(cls)}
        standard_kwargs = {k: v for k, v in kwargs.items() if k in dataclass_fields}

        # We need to call bare `__init__` without `__post_init__` but the `original_init` of
        # any dataclas contains a call to post-init at the end (without kwargs)
        if len(args) > 0:
            raise ValueError(
                f"{cls.__name__} accepts only keyword arguments, but found `{len(args)}` positional args."
            )

        for f in fields(cls):  # type: ignore
            if f.name in standard_kwargs:
                setattr(self, f.name, standard_kwargs[f.name])
            elif f.default is not MISSING:
                setattr(self, f.name, f.default)
            elif f.default_factory is not MISSING:
                setattr(self, f.name, f.default_factory())
            else:
                raise TypeError(f"Missing required field - '{f.name}'")

        # Pass any additional kwargs to `__post_init__` and let the object
        # decide whether to set the attr or use for different purposes (e.g. BC checks)
        additional_kwargs = {}
        for name, value in kwargs.items():
            if name not in dataclass_fields:
                additional_kwargs[name] = value

        self.__post_init__(**additional_kwargs)

    cls.__init__ = __init__
    return cls


def _get_typed_dict_annotations(schema: type[TypedDictT]) -> dict[str, Any]:
    """Extract type annotations from a TypedDict class."""
    try:
        # Available in Python 3.14+
        import annotationlib

        return annotationlib.get_annotations(schema)
    except ImportError:
        return {
            # We do not use `get_type_hints` here to avoid evaluating ForwardRefs (which might fail).
            # ForwardRefs are not validated by @strict anyway.
            name: value if value is not None else type(None)
            for name, value in schema.__dict__.get("__annotations__", {}).items()
        }


def _is_required_or_notrequired(type_hint: Any) -> bool:
    """Helper to check if a type is Required/NotRequired."""
    return type_hint in (Required, NotRequired) or (get_origin(type_hint) in (Required, NotRequired))


@lru_cache
def _build_strict_cls_from_typed_dict(schema: type[TypedDictT]) -> type:
    from .strict_decorator import strict

    # Extract type hints from the TypedDict class
    type_hints = _get_typed_dict_annotations(schema)

    # If the TypedDict is not total, wrap fields as NotRequired (unless explicitly Required or NotRequired)
    if not getattr(schema, "__total__", True):
        for key, value in type_hints.items():
            origin = get_origin(value)

            if origin is Annotated:
                base, *meta = get_args(value)
                if not _is_required_or_notrequired(base):
                    base = NotRequired[base]
                type_hints[key] = Annotated[tuple([base] + list(meta))]  # type: ignore
            elif not _is_required_or_notrequired(value):
                type_hints[key] = NotRequired[value]

    # Convert type hints to dataclass fields
    _fields = []
    for key, value in type_hints.items():
        if get_origin(value) is Annotated:
            base, *meta = get_args(value)
            _fields.append((key, base, field(default=_TYPED_DICT_DEFAULT_VALUE, metadata={"validator": meta[0]})))
        else:
            _fields.append((key, value, field(default=_TYPED_DICT_DEFAULT_VALUE)))

    # Create a strict dataclass from the TypedDict fields
    return strict(make_dataclass(schema.__name__, fields))
