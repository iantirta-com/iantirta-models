import collections.abc
import inspect
from dataclasses import Field, field, MISSING
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
import types
from collections.abc import Callable


from .types import ValidatorT, TypedDictT, _TYPED_DICT_DEFAULT_VALUE, Required, NotRequired


def validated_field(
    validator: list[ValidatorT] | ValidatorT,
    default: Any = MISSING,
    default_factory: Any = MISSING,
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: dict | None = None,
    **kwargs: Any,
) -> Any:
    """
    Create a dataclass field with a custom validator.

    Useful to apply several checks to a field. If only applying one rule, check out the [`as_validated_field`] decorator.

    Args:
        validator (`Callable` or `list[Callable]`):
            A method that takes a value as input and raises ValueError/TypeError if the value is invalid.
            Can be a list of validators to apply multiple checks.
        **kwargs:
            Additional arguments to pass to `dataclasses.field()`.

    Returns:
        A field with the validator attached in metadata
    """
    if not isinstance(validator, list):
        validator = [validator]
    if metadata is None:
        metadata = {}
    metadata["validator"] = validator
    return field(  # type: ignore
        default=default,  # type: ignore
        default_factory=default_factory,  # type: ignore
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
        **kwargs,
    )


def validate_typed_dict(schema: type[TypedDictT], data: dict) -> None:
    """
    Validate that a dictionary conforms to the types defined in a TypedDict class.

    Under the hood, the typed dict is converted to a strict dataclass and validated using the `@strict` decorator.

    Args:
        schema (`type[TypedDictType]`):
            The TypedDict class defining the expected structure and types.
        data (`dict`):
            The dictionary to validate.

    Raises:
        `StrictDataclassFieldValidationError`:
            If any field in the dictionary does not conform to the expected type.

    Example:
    ```py
    >>> from typing import Annotated, TypedDict
    >>> from iantirta.models.vendor.huggingface_hub.dataclasses import validate_typed_dict

    >>> def positive_int(value: int):
    ...     if not value >= 0:
    ...         raise ValueError(f"Value must be positive, got {value}")

    >>> class User(TypedDict):
    ...     name: str
    ...     age: Annotated[int, positive_int]

    >>> # Valid data
    >>> validate_typed_dict(User, {"name": "John", "age": 30})

    >>> # Invalid type for age
    >>> validate_typed_dict(User, {"name": "John", "age": "30"})
    huggingface_hub.errors.StrictDataclassFieldValidationError: Validation error for field 'age':
        TypeError: Field 'age' expected int, got str (value: '30')

    >>> # Invalid value for age
    >>> validate_typed_dict(User, {"name": "John", "age": -1})
    huggingface_hub.errors.StrictDataclassFieldValidationError: Validation error for field 'age':
        ValueError: Value must be positive, got -1
    ```
    """
    from .utils import _build_strict_cls_from_typed_dict

    # Convert typed dict to dataclass
    strict_cls = _build_strict_cls_from_typed_dict(schema)

    # Validate the data by instantiating the strict dataclass
    strict_cls(**data)  # will raise if validation fails


# ---
# Validator
# ---

def _validate_none(name: str, value: Any) -> None:
    """Validate None type.

    'None' is not a type, it's a special value. type should be `NoneType` instead.
    But in type annotations 'None' is accepted so we must support it.
    """
    if value is not None:
        raise TypeError(f"Field '{name}' expected None, got {type(value).__name__}")


def _validate_union(name: str, value: Any, args: tuple[Any, ...]) -> None:
    """Validate that value matches one of the types in a Union."""
    errors = []
    for t in args:
        try:
            type_validator(name, value, t)
            return  # Valid if any type matches
        except TypeError as e:
            errors.append(str(e))

    raise TypeError(
        f"Field '{name}' with value {repr(value)} doesn't match any type in {args}. Errors: {'; '.join(errors)}"
    )


def _validate_literal(name: str, value: Any, args: tuple[Any, ...]) -> None:
    """Validate Literal type."""
    if isinstance(value, bool):
        if value not in [arg for arg in args if isinstance(arg, bool)]:
            raise TypeError(f"Field '{name}' expected one of {args}, got {value}")

    elif isinstance(value, int):
        if value not in [arg for arg in args if isinstance(arg, int) and not isinstance(arg, bool)]:
            raise TypeError(f"Field '{name}' expected one of {args}, got {value}")

    elif value not in args:
        raise TypeError(f"Field '{name}' expected one of {args}, got {value}")


def _validate_list(name: str, value: Any, args: tuple[Any, ...]) -> None:
    """Validate list or list[T] type."""
    if not isinstance(value, list):
        raise TypeError(f"Field '{name}' expected a list, got {type(value).__name__}")

    # If no type argument is provided (i.e., bare `typing.List`), skip item validation
    if not args:
        return

    # Validate each item in the list
    item_type = args[0]
    for i, item in enumerate(value):
        try:
            type_validator(f"{name}[{i}]", item, item_type)
        except TypeError as e:
            raise TypeError(f"Invalid item at index {i} in list '{name}'") from e


def _validate_dict(name: str, value: Any, args: tuple[Any, ...]) -> None:
    """Validate dict or dict[K, V] type."""
    if not isinstance(value, dict):
        raise TypeError(f"Field '{name}' expected a dict, got {type(value).__name__}")

    # If no type arguments are provided (i.e., bare `typing.Dict`), skip key/value validation
    if not args:
        return

    # Validate keys and values
    key_type, value_type = args
    for k, v in value.items():
        try:
            type_validator(f"{name}.key", k, key_type)
            type_validator(f"{name}[{k!r}]", v, value_type)
        except TypeError as e:
            raise TypeError(f"Invalid key or value in dict '{name}'") from e


def _validate_tuple(name: str, value: Any, args: tuple[Any, ...]) -> None:
    """Validate Tuple type."""
    if not isinstance(value, tuple):
        raise TypeError(f"Field '{name}' expected a tuple, got {type(value).__name__}")

    # Handle variable-length tuples: tuple[T, ...]
    if len(args) == 2 and args[1] is Ellipsis:
        for i, item in enumerate(value):
            try:
                type_validator(f"{name}[{i}]", item, args[0])
            except TypeError as e:
                raise TypeError(f"Invalid item at index {i} in tuple '{name}'") from e
    # Handle fixed-length tuples: tuple[T1, T2, ...]
    elif len(args) != len(value):
        raise TypeError(f"Field '{name}' expected a tuple of length {len(args)}, got {len(value)}")
    else:
        for i, (item, expected) in enumerate(zip(value, args)):
            try:
                type_validator(f"{name}[{i}]", item, expected)
            except TypeError as e:
                raise TypeError(f"Invalid item at index {i} in tuple '{name}'") from e


def _validate_set(name: str, value: Any, args: tuple[Any, ...]) -> None:
    """Validate set or set[T] type."""
    if not isinstance(value, set):
        raise TypeError(f"Field '{name}' expected a set, got {type(value).__name__}")

    # If no type argument is provided (i.e., bare `typing.Set`), skip item validation
    if not args:
        return

    # Validate each item in the set
    item_type = args[0]
    for i, item in enumerate(value):
        try:
            type_validator(f"{name} item", item, item_type)
        except TypeError as e:
            raise TypeError(f"Invalid item in set '{name}'") from e


def _validate_sequence(name: str, value: Any, args: tuple[Any, ...]) -> None:
    """Validate Sequence or Sequence[T] type."""
    if not isinstance(value, collections.abc.Sequence):
        raise TypeError(f"Field '{name}' expected a Sequence, got {type(value).__name__}")

    # If no type argument is provided (i.e., just `Sequence`), skip item validation
    if not args:
        return

    # Validate each item in the sequence
    item_type = args[0]
    for i, item in enumerate(value):
        try:
            type_validator(f"{name}[{i}]", item, item_type)
        except TypeError as e:
            raise TypeError(f"Invalid item at index {i} in sequence '{name}'") from e


def _validate_simple_type(name: str, value: Any, expected_type: type) -> None:
    """Validate simple type (int, str, etc.)."""
    if expected_type is int and isinstance(value, bool):
        raise TypeError(
            f"Field '{name}' expected {expected_type.__name__}, got {type(value).__name__} (value: {repr(value)})"
        )
    if not isinstance(value, expected_type):
        raise TypeError(
            f"Field '{name}' expected {expected_type.__name__}, got {type(value).__name__} (value: {repr(value)})"
        )


# Entry

def type_validator(name: str, value: Any, expected_type: Any) -> None:
    """Validate that 'value' matches 'expected_type'."""
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if expected_type is Any:
        return

    elif expected_type is None:
        _validate_none(name, value)

    elif validator := _BASIC_TYPE_VALIDATORS.get(origin):
        validator(name, value, args)

    elif isinstance(expected_type, type):  # simple types
        _validate_simple_type(name, value, expected_type)

    elif isinstance(expected_type, ForwardRef) or isinstance(expected_type, str):
        return

    elif origin is Required:
        if value is _TYPED_DICT_DEFAULT_VALUE:
            raise TypeError(f"Field '{name}' is required but missing.")
        type_validator(name, value, args[0])

    elif origin is NotRequired:
        if value is _TYPED_DICT_DEFAULT_VALUE:
            return
        type_validator(name, value, args[0])

    else:
        raise TypeError(f"Unsupported type for field '{name}': {expected_type}")


def _create_type_validator(field: Field) -> ValidatorT:
    """Create a type validator function for a field."""
    # Hacky: we cannot use a lambda here because of reference issues

    def validator(value: Any) -> None:
        type_validator(field.name, value, field.type)

    return validator


def _is_validator(validator: Any) -> bool:
    """Check if a function is a validator.

    A validator is a Callable that can be called with a single positional argument.
    The validator can have more arguments with default values.

    Basically, returns True if `validator(value)` is possible.
    """
    if not callable(validator):
        return False

    signature = inspect.signature(validator)
    parameters = list(signature.parameters.values())

    if len(parameters) == 0:
        return False

    if parameters[0].kind not in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.VAR_POSITIONAL,
    ):
        return False

    for parameter in parameters[1:]:
        if parameter.default == inspect.Parameter.empty:
            return False

    return True


_BASIC_TYPE_VALIDATORS: dict[Any, Callable[[str, Any, tuple[Any, ...]], None]] = {
    Union: _validate_union,
    Literal: _validate_literal,
    list: _validate_list,
    dict: _validate_dict,
    tuple: _validate_tuple,
    set: _validate_set,
    collections.abc.Sequence: _validate_sequence,
}

# TODO: make it first class citizen when bumping to Python 3.10+
_BASIC_TYPE_VALIDATORS[types.UnionType] = _validate_union  # x | y syntax, available only Python 3.10+
