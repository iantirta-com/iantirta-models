from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from functools import wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pretrained import PreTrainedConfig


# Mapping from old names to new names
_LEGACY_LAYER_TYPE_REMAP = {
    "mamba": "linear_attention",
    "attention": "full_attention",
    "deepseek_sparse_attention": "indexed_attention",  # for models with DSA indexer (GLM MoE DSA, DeepSeek V32, ...)
    "qwen_sparse_attention": "indexed_attention",  # QSA with block-compressed indexer keys (Qwen4-Exp)
}


def remap_legacy_layer_types(
    layer_types: list[str] | None = None, config: PreTrainedConfig | None = None
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
        return [_LEGACY_LAYER_TYPE_REMAP.get(t, t) for t in layer_types]
    else:
        if getattr(config, "layer_types", None) is not None:  # noqa: SIM102
            # This check should not be needed, but sometimes `layer_types` is a read-only @property (already following
            # correct conventions), so this avoids error when trying to `setattr` it
            if (remapped := remap_legacy_layer_types(config.layer_types)) != config.layer_types:
                config.layer_types = remapped
        if getattr(config, "mtp_layer_types", None) is not None:  # noqa: SIM102
            # This check should not be needed, but sometimes `mtp_layer_types` is a read-only @property (already following
            # correct conventions), so this avoids error when trying to `setattr` it
            if (remapped := remap_legacy_layer_types(config.mtp_layer_types)) != config.mtp_layer_types:
                config.mtp_layer_types = remapped


# copied from iantirta.models.vendor.huggingface_hub.dataclasses.strict when `accept_kwargs=True`
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


def recursive_diff_dict(dict_a, dict_b, config_obj=None):
    """
    Helper function to recursively take the diff between two nested dictionaries. The resulting diff only contains the
    values from `dict_a` that are different from values in `dict_b`.

    dict_b : the default config dictionary. We want to remove values that are in this one
    """
    from .pretrained import PreTrainedConfig

    diff = {}
    default = config_obj.__class__().to_dict() if config_obj is not None else {}
    for key, value in dict_a.items():
        # HeterogeneousConfigMixin: disable the heterogeneous attribute access validation
        obj_value = (
            config_obj._getattr_without_heterogeneous_validation(str(key), None) if config_obj is not None else None
        )
        if isinstance(obj_value, PreTrainedConfig) and key in dict_b and isinstance(dict_b[key], dict):
            diff_value = recursive_diff_dict(value, dict_b[key], config_obj=obj_value)
            diff[key] = diff_value
        elif key not in dict_b or (value != default[key]):
            diff[key] = value
    return diff


def is_timm_config_dict(config_dict: dict[str, Any]) -> bool:
    """Checks whether a config dict is a timm config dict."""
    return "pretrained_cfg" in config_dict
