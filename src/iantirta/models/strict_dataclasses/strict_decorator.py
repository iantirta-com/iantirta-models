import collections.abc
import inspect
import types
from collections.abc import Callable
from dataclasses import fields, MISSING
from functools import wraps
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

from .errors import (
    StrictDataclassClassValidationError,
    StrictDataclassDefinitionError,
    StrictDataclassFieldValidationError,
)
from .types import T, ValidatorT, TypedDictT
from .validator import _create_type_validator, _is_validator, validated_field
from .utils import wrap_init_to_accept_kwargs


# The overload decorator helps type checkers understand the different return types
@overload
def strict(cls: type[T]) -> type[T]: ...


@overload
def strict(*, accept_kwargs: bool = False) -> Callable[[type[T]], type[T]]: ...


def strict(cls: type[T] | None = None, *, accept_kwargs: bool = False) -> type[T] | Callable[[type[T]], type[T]]:
    """
    Decorator to add strict validation to a dataclass.

    This decorator must be used on top of `@dataclass` to ensure IDEs and static typing tools
    recognize the class as a dataclass.

    Can be used with or without arguments:
    - `@strict`
    - `@strict(accept_kwargs=True)`

    Args:
        cls:
            The class to convert to a strict dataclass.
        accept_kwargs (`bool`, *optional*):
            If True, allows arbitrary keyword arguments in `__init__`. Defaults to False.

    Returns:
        The enhanced dataclass with strict validation on field assignment.

    Example:
    ```py
    >>> from dataclasses import dataclass
    >>> from iantirta.models.vendor.huggingface_hub.dataclasses import as_validated_field, strict, validated_field

    >>> @as_validated_field
    >>> def positive_int(value: int):
    ...     if not value >= 0:
    ...         raise ValueError(f"Value must be positive, got {value}")

    >>> @strict(accept_kwargs=True)
    ... @dataclass
    ... class User:
    ...     name: str
    ...     age: int = positive_int(default=10)

    # Initialize
    >>> User(name="John")
    User(name='John', age=10)

    # Extra kwargs are accepted
    >>> User(name="John", age=30, lastname="Doe")
    User(name='John', age=30, *lastname='Doe')

    # Invalid type => raises
    >>> User(name="John", age="30")
    huggingface_hub.errors.StrictDataclassFieldValidationError: Validation error for field 'age':
        TypeError: Field 'age' expected int, got str (value: '30')

    # Invalid value => raises
    >>> User(name="John", age=-1)
    huggingface_hub.errors.StrictDataclassFieldValidationError: Validation error for field 'age':
        ValueError: Value must be positive, got -1
    ```
    """

    def wrap(cls: type[T]) -> type[T]:
        if not hasattr(cls, "__dataclass_fields__"):
            raise StrictDataclassDefinitionError(
                f"Class '{cls.__name__}' must be a dataclass before applying @strict."
            )

        # List and store validators
        field_validators: dict[str, list[ValidatorT]] = {}
        for f in fields(cls):  # type: ignore
            validators = []
            validators.append(_create_type_validator(f))
            custom_validator = f.metadata.get("validator")
            if custom_validator is not None:
                if not isinstance(custom_validator, list):
                    custom_validator = [custom_validator]
                for validator in custom_validator:
                    if not _is_validator(validator):
                        raise StrictDataclassDefinitionError(
                            f"Invalid validator for field '{f.name}': {validator}. Must be a callable taking a single argument."
                        )
                validators.extend(custom_validator)
            field_validators[f.name] = validators
        cls.__validators__ = field_validators  # type: ignore

        # Override __setattr__ to validate fields on assignment
        original_setattr = cls.__setattr__

        def __strict_setattr__(self: Any, name: str, value: Any) -> None:
            """Custom __setattr__ method for strict dataclasses."""
            # Run all validators
            for validator in self.__validators__.get(name, []):
                try:
                    validator(value)
                except (ValueError, TypeError) as e:
                    raise StrictDataclassFieldValidationError(field=name, cause=e) from e

            # If validation passed, set the attribute
            original_setattr(self, name, value)

        cls.__setattr__ = __strict_setattr__  # type: ignore

        if accept_kwargs:
            # (optional) Override __init__ to accept arbitrary keyword arguments
            cls = wrap_init_to_accept_kwargs(cls)

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
                raise StrictDataclassDefinitionError(
                    f"Class '{cls.__name__}' has a class validator '{name}' that takes more than one argument."
                    " Class validators must take only 'self' as an argument. Methods starting with 'validate_'"
                    " are considered to be class validators."
                )
            class_validators.append(method)

        cls.__class_validators__ = class_validators  # type: ignore

        # Add `validate` method to the class, but first check if it already exists
        def validate(self: T) -> None:
            """Run class validators on the instance."""
            for validator in cls.__class_validators__:  # type: ignore [attr-defined]
                try:
                    validator(self)
                except (ValueError, TypeError) as e:
                    raise StrictDataclassClassValidationError(validator=validator.__name__, cause=e) from e

        # Hack to be able to raise if `.validate()` already exists except if it was created by this decorator on a parent class
        # (in which case we just override it)
        validate.__is_defined_by_strict_decorator__ = True  # type: ignore [attr-defined]

        if hasattr(cls, "validate"):
            if not getattr(cls.validate, "__is_defined_by_strict_decorator__", False):  # type: ignore [attr-defined]
                raise StrictDataclassDefinitionError(
                    f"Class '{cls.__name__}' already implements a method called 'validate'."
                    " This method name is reserved when using the @strict decorator on a dataclass."
                    " If you want to keep your own method, please rename it."
                )

        cls.validate = validate  # type: ignore

        # Run class validators after initialization
        initial_init = cls.__init__

        @wraps(initial_init)
        def init_with_validate(self, *args, **kwargs) -> None:
            """Run class validators after initialization."""
            initial_init(self, *args, **kwargs)  # type: ignore [call-arg]
            cls.validate(self)  # type: ignore [attr-defined]

        setattr(cls, "__init__", init_with_validate)

        return cls

    # Return wrapped class or the decorator itself
    return wrap(cls) if cls is not None else wrap


def as_validated_field(validator: ValidatorT):
    """
    Decorates a validator function as a [`validated_field`] (i.e. a dataclass field with a custom validator).

    Args:
        validator (`Callable`):
            A method that takes a value as input and raises ValueError/TypeError if the value is invalid.
    """

    def _inner(
        default: Any = MISSING,
        default_factory: Any = MISSING,
        init: bool = True,
        repr: bool = True,
        hash: bool | None = None,
        compare: bool = True,
        metadata: dict | None = None,
        **kwargs: Any,
    ):
        return validated_field(
            validator,
            default=default,
            default_factory=default_factory,
            init=init,
            repr=repr,
            hash=hash,
            compare=compare,
            metadata=metadata,
            **kwargs,
        )

    return _inner
