from __future__ import annotations

import logging
from dataclasses import MISSING, dataclass, fields
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

from packaging import version

from iantirta.models import __version__

if TYPE_CHECKING:
    from .pretrained import PreTrainedConfig


logger = logging.getLogger(__name__)


CONFIG_NAME = "config.json"


# type hinting: specifying the type of config class that inherits from PreTrainedConfig
SpecificPreTrainedConfigType = TypeVar("SpecificPreTrainedConfigType", bound="PreTrainedConfig")

_FLOAT_TAG_KEY = "__float__"
_FLOAT_TAG_VALUES = {"Infinity": float("inf"), "-Infinity": float("-inf"), "NaN": float("nan")}


ALLOWED_ATTN_LAYER_TYPES = (
    "full_attention",
    "sliding_attention",
    "chunked_attention",
    "window_attention",  # non-overlapping windows usually in ViT
    "indexed_attention",  # For indexer-based attentions
    "compressed_sparse_attention",  # CSA, used in deepseek_v4
    "heavily_compressed_attention",  # HCA, used in deepseek_v4
    "minimax_m3_sparse",  # lightning-index sparse attention, used in minimax_m3_vl
    "conv",
    "moe",  # for nemotron_h, which uses either attention, mamba or moe
    "hybrid",  # layers that combine attention + mamba/linear-attention-shaped states (zamba2, falcon_h1, zaya1)
    "hybrid_sliding",  # layers that combine sliding attention + linear-attention-shaped states (zaya1)
    # Recurrent layers (mamba / mamba2 / GDN / minimax-lightning)
    "linear_attention",
)

ALLOWED_MLP_LAYER_TYPES = (
    "sparse",
    "dense",
)

# Keep a complete list of layer types as well for BC
ALLOWED_LAYER_TYPES = ALLOWED_ATTN_LAYER_TYPES + ALLOWED_MLP_LAYER_TYPES


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


# copied from huggingface_hub.dataclasses.strict when `accept_kwargs=True`
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


def get_configuration_file(configuration_files: list[str]) -> str:
    """
    Get the configuration file to use for this version of iantirta.models.

    Args:
        configuration_files (`list[str]`): The list of available configuration files.

    Returns:
        `str`: The configuration file to use.
    """
    configuration_files_map = {}
    for file_name in configuration_files:
        if file_name.startswith("config.") and file_name.endswith(".json") and file_name != "config.json":
            v = file_name.removeprefix("config.").removesuffix(".json")
            configuration_files_map[v] = file_name
    available_versions = sorted(configuration_files_map.keys())

    # Defaults to FULL_CONFIGURATION_FILE and then try to look at some newer versions.
    configuration_file = CONFIG_NAME
    transformers_version = version.parse(__version__)
    for v in available_versions:
        if version.parse(v) <= transformers_version:
            configuration_file = configuration_files_map[v]
        else:
            # No point going further since the versions are sorted.
            break

    return configuration_file


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


def get_head_shapes(config) -> tuple[int | list[int], int | list[int]]:
    """Returns a tuple `(num_kv_heads, head_dim)`, each of them either a single int for all layers, or a list of int
    with the value for each layer."""
    # Some models (e.g. Gemma4) have different head_dim and num_heads depending on layer type
    per_layer_attributes = config.per_layer_attributes or ()
    # Layers sharing kv states have no kv cache of their own, so they are excluded.
    layers = range(config.num_hidden_layers - getattr(config, "num_kv_shared_layers", 0))

    if "head_dim" in per_layer_attributes:
        head_dim = [config.per_layer_config[layer].head_dim for layer in layers]
    else:
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

    if "num_key_value_heads" in per_layer_attributes:
        num_kv_heads = [config.per_layer_config[layer].num_key_value_heads for layer in layers]
    else:
        num_kv_heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads

    return num_kv_heads, head_dim


def layer_type_validation(layer_types: list[str], num_hidden_layers: int | None = None, attention: bool = True):
    logger.warning(
        "`layer_type_validation` is deprecated and will be removed in v5.20. "
        "Use `PreTrainedConfig.validate_layer_type` instead"
    )

    if not all(layer_type in ALLOWED_LAYER_TYPES for layer_type in layer_types):
        raise ValueError(f"The `layer_types` entries must be in {ALLOWED_LAYER_TYPES}")
    if num_hidden_layers is not None and num_hidden_layers != len(layer_types):
        raise ValueError(
            f"`num_hidden_layers` ({num_hidden_layers}) must be equal to the number of layer types "
            f"({len(layer_types)})"
        )


def is_timm_config_dict(config_dict: dict[str, Any]) -> bool:
    """Checks whether a config dict is a timm config dict."""
    return "pretrained_cfg" in config_dict

