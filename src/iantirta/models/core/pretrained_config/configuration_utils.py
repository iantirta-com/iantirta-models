
import logging

logger = logging.getLogger("iantirta.models.PreTrainedConfig")


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


class SubConfigSpec:
    def __init__(
        self,
        config_class,
        model_type: str | None = None,
        init_kwargs: dict | None = None,
        optional: bool = False,
    ):
        """
        Specification for initializing sub-configs within a composite configuration class.

        Args:
            config_class (`PretrainedConfig` or `type`):
                The config class used to instantiate the subconfig (e.g. `Qwen2VLVisionConfig`)
                or `AutoConfig` if subconfig should be resolved dynamically from input kwargs.`
            model_type (`str`, *optional*):
                Must be provided only if `config_class=AutoConfig` - indicates a default model type associated
                with the subconfig.
            init_kwargs (`dict`, *optional*):
                Additional keyword arguments to pass when instantiating the subconfig.
            optional (`bool`, *optional*, defaults to `False`):
                Whether this subconfig is optional in the composite model hierarchy. If not optional,
                an error is raised when it's explicitly set to `None` in input kwargs.
        """
        self.init_kwargs = init_kwargs if init_kwargs is not None else {}
        self.config_class = config_class
        self.optional = optional

        # we can have `AutoConfig` with a default model-type (eg. LLaVA), or a subconfig
        # with `cls.model_type` attr set to non-empty value (e.g. Qwen2VLVisionConfig)
        if model_type is None and not hasattr(config_class, "model_type"):
            raise ValueError(
                "You have to provide either a valid `model_type` or an specific `config_class` "
                f"to init subconfigs correctly, but got model_type={model_type} and config_class={config_class}"
            )
        self.model_type = model_type if model_type is not None else config_class.model_type

    def get_config_class(self, model_type: str | None = None):
        # Avoid circular imports - we only need the static mapping here to map from `string` to config class
        from ..auto.auto_config import CONFIG_MAPPING

        if model_type not in CONFIG_MAPPING:
            raise ValueError(
                f"Unrecognized model type for subconfig: {model_type}. Should contain one of {', '.join(CONFIG_MAPPING.keys())}"
            )

        return CONFIG_MAPPING[model_type]

    def create_subconfig(self, key, subconfig=None, **kwargs):
        """
        Construct a subconfig class either from provided input or from spec defaults.
        In case the provided input is a `dict` without `model_type`, we fallback
        to spec's default model-type.
        Provided input can only be one of - [`PreTrainedConfig`, `dict`, `None`]
        """
        from .configuration_pretrained import PreTrainedConfig
        
        # early exit if sub-config is already a config instance
        if isinstance(subconfig, PreTrainedConfig):
            return subconfig, kwargs

        # Vision model backbones have their own utility for BC/Timm
        if key == "backbone_config":
            raise NotImplementedError("Backbone config")

        # For BC with released models where `config.json` might contain non-existant model types, we don't try to map
        # config by `model_type` if the spec points to a particular class (e.g. `MyModeltextConfig`). Model-type
        # resolution takes action only when the spec holds `config_class=AutoConfig`
        if not issubclass(self.config_class, PreTrainedConfig):
            model_type = subconfig.get("model_type", self.model_type) if subconfig is not None else self.model_type
            if not model_type:
                raise ValueError(f"Cannot resolve `{key}`: no `model_type` found inputs or in `sub_configs_defaults`.")
            subconfig_cls = self.get_config_class(model_type)

        else:
            subconfig_cls = self.config_class

        # Copy the dict to not mutate it in-place
        if isinstance(subconfig, dict):
            pass
        elif subconfig is None:
            subconfig = self.init_kwargs
            logger.info(f"`{key}` is None, initializing {subconfig_cls.__name__} with default values.")
        else:
            raise TypeError(f"`{key}` must be a `dict`, `PreTrainedConfig`, or `None`, got `{type(subconfig)}`")

        return subconfig_cls(**subconfig), kwargs

    @property
    def default_config(self):
        return self.get_config_class(self.model_type)(**self.init_kwargs)

    def default_config_fields(self):
        """
        Collect default subconfig kwargs resolved from subconfig's dataclass defaults
        and `self.init_kwargs`
        """
        from .configuration_pretrained import PreTrainedConfig

        default_subconfig_class = (
            self.get_config_class(self.model_type)
            if not issubclass(self.config_class, PreTrainedConfig)
            else self.config_class
        )
        subconfig_default_fields = default_subconfig_class.default_config_fields()
        subconfig_default_fields.update(self.init_kwargs)
        return subconfig_default_fields


# a small helper for BC - substitutes a class attribute
class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


def recursive_diff_dict(dict_a, dict_b, config_obj=None):
    """
    Helper function to recursively take the diff between two nested dictionaries. The resulting diff only contains the
    values from `dict_a` that are different from values in `dict_b`.

    dict_b : the default config dictionary. We want to remove values that are in this one
    """
    from .configuration_pretrained import PreTrainedConfig

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
        if getattr(config, "layer_types", None) is not None:
            # This check should not be needed, but sometimes `layer_types` is a read-only @property (already following
            # correct conventions), so this avoids error when trying to `setattr` it
            if (remapped := remap_legacy_layer_types(config.layer_types)) != config.layer_types:
                config.layer_types = remapped
        if getattr(config, "mtp_layer_types", None) is not None:
            # This check should not be needed, but sometimes `mtp_layer_types` is a read-only @property (already following
            # correct conventions), so this avoids error when trying to `setattr` it
            if (remapped := remap_legacy_layer_types(config.mtp_layer_types)) != config.mtp_layer_types:
                config.mtp_layer_types = remapped
