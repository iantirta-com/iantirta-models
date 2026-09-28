
from dataclasses import fields, Field, MISSING
from functools import wraps
import collections.abc
import inspect
import types
from typing import Any
import typing as t


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


T = t.TypeVar("T")
Validator_T = t.Callable[[Any], None]

_TYPED_DICT_DEFAULT_VALUE = object()



@t.overload
def strict_dataclass(cls: t.Type[T]) -> t.Type[T]: ...

@t.overload
def strict_dataclass(*, post_init: bool = False) -> t.Callable[[t.Type[T]], t.Type[T]]: ...

def strict_dataclass(
    cls: t.Type[T] | None = None,
    *,
    post_init=False,
) -> Type[T] | t.Callable[[t.Type[T]], t.Type[T]]:
    def wrap(cls):
        if not hasattr(cls, "__dataclass_fields__"):
            raise TypeError(
                f"Class '{cls.__name__}' must be a dataclass before applying @strict_dataclass."
            )
        
        # List and store validators
        field_validators: dict[str, list[Validator_T]] = {}
        for f in fields(cls):  # type: ignore
            validators = []
            validators.append(_create_type_validator(f))
            custom_validator = f.metadata.get("validator")
            if custom_validator is not None:
                if not isinstance(custom_validator, list):
                    custom_validator = [custom_validator]
                for validator in custom_validator:
                    if not _is_validator(validator):
                        raise TypeError(
                            f"Invalid validator for field '{f.name}': {validator}. Must be a callable taking a single argument."
                        )
                validators.extend(custom_validator)
            field_validators[f.name] = validators
        cls.__validators__ = field_validators  # type: ignore

        original_setattr = cls.__setattr__

        def __strict_setattr__(self: Any, name: str, value: Any) -> None:
            """Custom __setattr__ method for strict dataclasses."""
            # Run all validators
            for validator in self.__validators__.get(name, []):
                try:
                    validator(value)
                except (ValueError, TypeError) as e:
                    raise TypeError(
                        f"field={name}, cause={e}"
                    ) from e

            # If validation passed, set the attribute
            original_setattr(self, name, value)

        cls.__setattr__ = __strict_setattr__

        if post_init:
            # (optional) Override __init__ to accept arbitrary keyword arguments
            original_init = cls.__init__

            @wraps(original_init)
            def __init__(self, *args, **kwargs: Any) -> None:
                # Extract only the fields that are part of the dataclass
                dataclass_fields = {f.name for f in fields(cls)}  # type: ignore
                standard_kwargs = {k: v for k, v in kwargs.items() if k in dataclass_fields}

                # User shouldn't define custom `__init__` when `accepts_kwargs`, and instead
                # are advised to move field manipulation to `__post_init__` (e.g., derive new field from existing ones)
                # We need to call bare `__init__` here without `__post_init__` but the``original_init`` would call
                # post-init right away with no kwargs.
                if len(args) > 0:
                    raise ValueError(
                        f"When `post_init=True`, {cls.__name__} accepts only keyword arguments, "
                        f"but found `{len(args)}` positional args."
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

            cls.__init__ = __init__  # type: ignore

            # Define a default __post_init__ if not defined
            if not hasattr(cls, "__post_init__"):

                def __post_init__(self, **kwargs: Any) -> None:
                    """Default __post_init__ to accept additional kwargs."""
                    for name, value in kwargs.items():
                        setattr(self, name, value)

                cls.__post_init__ = __post_init__  # type: ignore

            # (optional) Override __repr__ to include additional kwargs
            original_repr = cls.__repr__

            @wraps(original_repr)
            def __repr__(self) -> str:
                # Call the original __repr__ to get the standard fields
                standard_repr = original_repr(self)

                # Get additional kwargs
                additional_kwargs = [
                    # add a '*' in front of additional kwargs to let the user know they are not part of the dataclass
                    f"*{k}={v!r}"
                    for k, v in self.__dict__.items()
                    if k not in cls.__dataclass_fields__  # type: ignore [attr-defined]
                ]
                additional_repr = ", ".join(additional_kwargs)

                # Combine both representations
                return f"{standard_repr[:-1]}, {additional_repr})" if additional_kwargs else standard_repr

            if cls.__dataclass_params__.repr is True:  # type: ignore [attr-defined]
                cls.__repr__ = __repr__  # type: ignore

        # List all public methods starting with `validate_` => class validators.
        class_validators = []

        for name in dir(cls):
            if not name.startswith("validate_"):
                continue
            method = getattr(cls, name)
            if not callable(method):
                continue
            if len(inspect.signature(method).parameters) != 1:
                raise TypeError(
                    f"Class '{cls.__name__}' has a class validator '{name}' that takes more than one argument."
                    " Class validators must take only 'self' as an argument. Methods starting with 'validate_'"
                    " are considered to be class validators."
                )
            class_validators.append(method)


        cls.__class_validators__ = class_validators

        def validate(self: T) -> None:
            """Run class validators on the instance."""
            for validator in cls.__class_validators__:  # type: ignore [attr-defined]
                try:
                    validator(self)
                except (ValueError, TypeError) as e:
                    raise TypeError(
                        f"{validator.__name__} error: "
                        f"{e}"
                    ) from e

        # Hack to be able to raise if `.validate()` already exists except if it was created by this decorator on a parent class
        # (in which case we just override it)
        validate.__is_defined_by_strict_decorator__ = True  # type: ignore [attr-defined]

        if hasattr(cls, "validate"):
            if not getattr(cls.validate, "__is_defined_by_strict_decorator__", False):  # type: ignore [attr-defined]
                raise TypeError(
                    f"Class '{cls.__name__}' already implements a method called 'validate'."
                    " This method name is reserved when using the @strict decorator on a dataclass."
                    " If you want to keep your own method, please rename it."
                )
            
        cls.validate = validate
        
        initial_init = cls.__init__
        
        @wraps(initial_init)
        def init_with_validate(self, *args, **kwargs) -> None:
            """Run class validators after initialization."""
            initial_init(self, *args, **kwargs)  # type: ignore [call-arg]
            cls.validate(self)

        setattr(cls, "__init__", init_with_validate)
        
        return cls
        
    return wrap(cls) if cls is not None else wrap


def remap_legacy_layer_types(
    layer_types: list[str] | None = None,
    config: ModelConfig | None = None
) -> list[str] | None:
    """
    Remap legacy layer types to newer convention names. Any name that does not fit one of the `_LEGACY_LAYER_TYPE_REMAP`
    patterns is returned unchanged.
    This function can either take a list of `layer_types`, in which case a remapped list is returned, or a `config`,
    in which case the config's `layer_types` and `mtp_layer_types` will be modified in-place, and nothing will be returned.

    Args:
        layer_types (`list[str]`, optional):
            Layer type names that may include legacy values.
        config (`PreTrainedConfig`, optional):
            Config on which `layer_types` and `mtp_layer_types` will be remapped in-plce if they exist.


    Returns:
        `list[str]` if `layer_types` is passed, or `None` if `config` is passed.
    """
    if (layer_types is None) ^ (config is not None):
        raise ValueError("This function must take exactly one of `layer_types` or `config`")

    if layer_types is not None:
        raise NotImplementedError
        return [_LEGACY_LAYER_TYPE_REMAP.get(t, t) for t in layer_types]
    else:
        if getattr(config, "layer_types", None) is not None:
            raise NotImplementedError
            # This check should not be needed, but sometimes `layer_types` is a read-only @property (already following
            # correct conventions), so this avoids error when trying to `setattr` it
            if (remapped := remap_legacy_layer_types(config.layer_types)) != config.layer_types:
                config.layer_types = remapped
        if getattr(config, "mtp_layer_types", None) is not None:
            raise NotImplementedError
            # This check should not be needed, but sometimes `mtp_layer_types` is a read-only @property (already following
            # correct conventions), so this avoids error when trying to `setattr` it
            if (remapped := remap_legacy_layer_types(config.mtp_layer_types)) != config.mtp_layer_types:
                config.mtp_layer_types = remapped


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


def type_validator(name: str, value: Any, expected_type: Any) -> None:
    """Validate that 'value' matches 'expected_type'."""
    origin = t.get_origin(expected_type)
    args = t.get_args(expected_type)

    if expected_type is Any:
        return
    
    elif expected_type is None:
        if value is not None:
            raise TypeError(f"Field '{name}' expected None, got {type(value).__name__}")

    elif validator := _BASIC_TYPE_VALIDATORS.get(origin):
        validator(name, value, args)

    elif isinstance(expected_type, type):  # simple types
        _validate_simple_type(name, value, expected_type)
    elif isinstance(expected_type, t.ForwardRef) or isinstance(expected_type, str):
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


def _create_type_validator(field: Field) -> Validator_T:
    """Create a type validator function for a field."""
    # Hacky: we cannot use a lambda here because of reference issues

    def validator(value: Any) -> None:
        type_validator(field.name, value, field.type)

    return validator

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
    """Validate list[T] type."""
    if not isinstance(value, list):
        raise TypeError(f"Field '{name}' expected a list, got {type(value).__name__}")

    # Validate each item in the list
    item_type = args[0]
    for i, item in enumerate(value):
        try:
            type_validator(f"{name}[{i}]", item, item_type)
        except TypeError as e:
            raise TypeError(f"Invalid item at index {i} in list '{name}'") from e


def _validate_dict(name: str, value: Any, args: tuple[Any, ...]) -> None:
    """Validate dict[K, V] type."""
    if not isinstance(value, dict):
        raise TypeError(f"Field '{name}' expected a dict, got {type(value).__name__}")

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
    """Validate set[T] type."""
    if not isinstance(value, set):
        raise TypeError(f"Field '{name}' expected a set, got {type(value).__name__}")

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

_BASIC_TYPE_VALIDATORS: dict[Any, t.Callable[[str, Any, tuple[Any, ...]], None]] = {
    t.Union: _validate_union,
    t.Literal: _validate_literal,
    list: _validate_list,
    dict: _validate_dict,
    tuple: _validate_tuple,
    set: _validate_set,
    collections.abc.Sequence: _validate_sequence,
}

# TODO: make it first class citizen when bumping to Python 3.10+
_BASIC_TYPE_VALIDATORS[types.UnionType] = _validate_union  # x | y syntax, available only Python 3.10+
