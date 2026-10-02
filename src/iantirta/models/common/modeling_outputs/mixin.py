from __future__ import annotations

from collections import OrderedDict
from typing import Any
from typing_extensions import Self
from functools import partial, wraps
from collections.abc import Iterable
from dataclasses import fields, is_dataclass
from iantirta.models.tools.tensor import is_tensor

import torch

OUTPUT_TYPES: set[type[Any]] = set()


def _model_output_flatten(
    output: ModelOutputMixin
) -> tuple[list[Any], list[str]]:
    return list(output.values()), list(output.keys())


def _model_output_unflatten(
    values: Iterable[Any],
    context: list[str],
    output_type: type[ModelOutputMixin] | None = None,
) -> ModelOutputMixin:
    return output_type(**dict(zip(context, values)))


class ModelOutputMixin(OrderedDict):
    """
    Base class for all model outputs as dataclass.
    Has a `__getitem__` that allows indexing
    by integer or slice (like a tuple) or
    strings (like a dictionary) that will
    ignore the `None` attributes.
    Otherwise behaves like
    a regular python dictionary.

    <Tip warning={true}>

    You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    @staticmethod
    def register_output_type(output_type: type[Self]):
        # AMD CI runs PyTorch 2.8.0+rocm
        # which does not support tracing
        # `set.__contains__` through TorchDynamo.
        # Skip registration during compilation since the pytree node
        # is already registered from the preceding eager run.
        if torch.compiler.is_compiling():
            return
        if output_type in OUTPUT_TYPES:
            return

        import torch.utils._pytree as torch_pytree

        torch_pytree.register_pytree_node(
            output_type,
            _model_output_flatten,
            partial(
                _model_output_unflatten,
                output_type=output_type
            ),
            serialized_type_name=(
                f"{output_type.__module__}."
                f"{output_type.__name__}"
            ),
            flatten_with_keys_fn=torch_pytree._dict_flatten_with_keys,
        )
        OUTPUT_TYPES.add(output_type)

    def __init_subclass__(cls) -> None:
        """Register subclasses as pytree nodes.

        This is necessary to synchronize gradients when using
        `torch.nn.parallel.DistributedDataParallel` with
        `static_graph=True` with modules that output
        `ModelOutputMixin` subclasses.
        """
        cls.register_output_type(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_output_type(type(self))

        # Subclasses of ModelOutputMixin
        # must use the @dataclass decorator
        # This check is done in __init__
        # because the @dataclass decorator
        # operates after __init_subclass__
        # issubclass() would return True
        # for issubclass(ModelOutputMixin, ModelOutputMixin)
        # when False is needed
        # Just need to check that the current class
        # is not ModelOutput
        is_children = self.__class__ != ModelOutputMixin

        if is_children and not is_dataclass(self):
            raise TypeError(
                f"{self.__module__}.{self.__class__.__name__} "
                "is not a dataclass. "
                "This is a children of ModelOutputMixin "
                "and so must use the @dataclass decorator."
            )

    def __post_init__(self):
        """Check the ModelOutput dataclass.

        Only occurs if @dataclass decorator has been used.
        """
        self.register_output_type(type(self))
        _fields = fields(self)

        # Safety and consistency checks
        if not len(_fields):
            raise ValueError(
                f"{self.__class__.__name__} has no fields."
            )
        if not all(
            field.default is None
            for field in _fields[1:]
        ):
            raise ValueError(
                f"{self.__class__.__name__} "
                "should not have more than one required field."
            )

        first_field = getattr(self, _fields[0].name)
        other_fields_are_none = all(
            self.__dict__.get(field.name) is None
            for field in _fields[1:]
        )

        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                # reset first field to None and remove it from the internal dictionary
                setattr(self, _fields[0].name, None)
                super().__delitem__(_fields[0].name)
                for idx, element in enumerate(iterator):
                    if (
                        not isinstance(element, (list, tuple))
                        or len(element) != 2
                        or not isinstance(element[0], str)
                    ):
                        if idx == 0:
                            # If we do not have an iterator of key/values, set it as attribute
                            self[_fields[0].name] = first_field
                        else:
                            # If we have a mixed iterator, raise an error
                            raise ValueError(
                                f"Cannot set key/value for {element}. "
                                "It needs to be a tuple (key, value)."
                            )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[_fields[0].name] = first_field
        else:
            for field in _fields:
                v = self.__dict__.get(field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        field_names = {
            field.name
            for field in fields(self)
        }
        if name in field_names and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        callable, _args, *remaining = super().__reduce__()
        args = tuple(
            getattr(self, field.name)
            for field in fields(self)
        )
        return callable, args, *remaining

    def to_tuple(self) -> tuple:
        """
        Convert self to a tuple containing
        all the attributes/keys that are not `None`.
        """
        return tuple(
            self[k]
            for k in self.keys()
        )
