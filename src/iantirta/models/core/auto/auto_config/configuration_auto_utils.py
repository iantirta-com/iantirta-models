
import importlib
import logging
from collections import OrderedDict
from collections.abc import (
    Callable,
    Iterator,
    KeysView,
    ValuesView
)
from typing import TYPE_CHECKING

from .configuration_mapping import (
    SPECIAL_MODEL_TYPE_TO_MODULE_NAME,
    DEPRECATED_MODELS,
    CONFIG_MAPPING_NAMES
)

if TYPE_CHECKING:
    from iantirta.models import PreTrainedConfig


logger = logging.getLogger("iantirta.models.AutoConfig")


def model_type_to_module_name(key) -> str:
    """Converts a config key to the corresponding module."""
    # Special treatment
    if key in SPECIAL_MODEL_TYPE_TO_MODULE_NAME:
        key = SPECIAL_MODEL_TYPE_TO_MODULE_NAME[key]

        if key in DEPRECATED_MODELS:
            key = f"deprecated.{key}"
        return key

    key = key.replace("-", "_")
    if key in DEPRECATED_MODELS:
        key = f"deprecated.{key}"

    return key


class _LazyConfigMapping(OrderedDict[str, type["PreTrainedConfig"]]):
    """
    A dictionary that lazily load its values when they are requested.
    """

    def __init__(self, mapping) -> None:
        self._mapping = mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key: str) -> type["PreTrainedConfig"]:
        if key in self._extra_content:
            return self._extra_content[key]

        if key not in self._mapping:
            raise KeyError(key)

        value = self._mapping[key]
        module_name = model_type_to_module_name(key)

        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(
                f".{module_name}", "iantirta.models.models"
            )

        if hasattr(self._modules[module_name], value):
            return getattr(self._modules[module_name], value)

        # Some of the mappings have entries
        # model_type -> config of another model type.
        # In that case we try to grab the
        # object at the top level.
        iantirta_models_module = importlib.import_module(
            "iantirta.models"
        )
        return getattr(iantirta_models_module, value)

    def keys(self) -> list[str]:
        return list(self._mapping.keys()) + list(self._extra_content.keys())

    def values(self) -> list[type["PreTrainedConfig"]]:
        return [
            self[k] for k in self._mapping
        ] + list(self._extra_content.values())

    def items(self) -> list[tuple[str, type["PreTrainedConfig"]]]:
        return [
            (k, self[k]) for k in self._mapping
        ] + list(self._extra_content.items())

    def __iter__(self) -> Iterator[str]:
        return iter(
            list(self._mapping.keys())
            + list(self._extra_content.keys())
        )

    def __contains__(self, item: object) -> bool:
        return item in self._mapping or item in self._extra_content

    def register(
        self,
        key: str,
        value: type["PreTrainedConfig"],
        exist_ok=False
    ) -> None:
        """
        Register a new configuration in this mapping.
        """
        if key in self._mapping and not exist_ok:
            raise ValueError(
                f"'{key}' is already used by a "
                "Transformers config, pick another name."
            )
        self._extra_content[key] = value

    def __reduce__(self):
        return (
            self.__class__._from_pickle,
            (dict(self._mapping), dict(self._extra_content))
        )

    @classmethod
    def _from_pickle(cls, mapping, extra_content):
        obj = cls(mapping)
        obj._extra_content = extra_content
        return obj


CONFIG_MAPPING = _LazyConfigMapping(CONFIG_MAPPING_NAMES)
