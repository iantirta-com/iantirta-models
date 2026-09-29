# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.
#
# Partial code of huggingface_hub, improved by iantirta.com
"""Configuration base class and utilities."""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import MISSING, dataclass, fields
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from typing_extensions import Self, dataclass_transform

from iantirta.models.error import CurrentlyNotImplementedError
from iantirta.models.files import ensure_file

from .vendor.huggingface_hub.dataclasses import strict
from .vendor.transformers.generation.configuration_utils import GenerationConfig

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


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


# Mapping from old names to new names
_LEGACY_LAYER_TYPE_REMAP = {
    "mamba": "linear_attention",
    "attention": "full_attention",
    "deepseek_sparse_attention": "indexed_attention",  # for models with DSA indexer (GLM MoE DSA, DeepSeek V32, ...)
    "qwen_sparse_attention": "indexed_attention",  # QSA with block-compressed indexer keys (Qwen4-Exp)
}


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
        raise CurrentlyNotImplementedError("Layer types")
        return [_LEGACY_LAYER_TYPE_REMAP.get(t, t) for t in layer_types]
    else:
        if getattr(config, "layer_types", None) is not None:
            raise CurrentlyNotImplementedError("config layer_types")
            # This check should not be needed, but sometimes `layer_types` is a read-only @property (already following
            # correct conventions), so this avoids error when trying to `setattr` it
            if (remapped := remap_legacy_layer_types(config.layer_types)) != config.layer_types:
                config.layer_types = remapped
        if getattr(config, "mtp_layer_types", None) is not None:
            raise CurrentlyNotImplementedError("config mpt_layer_types")
            # This check should not be needed, but sometimes `mtp_layer_types` is a read-only @property (already following
            # correct conventions), so this avoids error when trying to `setattr` it
            if (remapped := remap_legacy_layer_types(config.mtp_layer_types)) != config.mtp_layer_types:
                config.mtp_layer_types = remapped




@dataclass_transform(kw_only_default=True)
@strict(accept_kwargs=True)
@dataclass(repr=False)
class ModelConfig:

    # Class attributes that we don't want to save or have in `self.__dict__`
    # They are not supposed to be set/changed by users. Each field is set when
    # creating a model class
    base_config_key: ClassVar[str] = ""
    sub_configs: ClassVar[dict[str, type[ModelConfig]]] = {}

    attribute_map: ClassVar[dict[str, str]] = {}

    # Common attributes for all models
    dtype: str | torch.dtype | None = None

    # Fine-tuning task arguments
    id2label: dict[int, str] | dict[str, str] | None = None
    label2id: dict[str, int] | dict[str, str] | None = None
    problem_type: Literal["regression", "single_label_classification", "multi_label_classification"] | None = None

    #region Dunder 

    def __post_init__(self, **kwargs):
        # BC for the `torch_dtype` argument instead of the simpler `dtype`
        # Do not warn, as it would otherwise always be triggered since most configs on the hub have `torch_dtype`
        if (torch_dtype := kwargs.pop("torch_dtype", None)) is not None:
            # If both are provided, keep `dtype`
            self.dtype = self.dtype if self.dtype is not None else torch_dtype
        if self.dtype is not None and isinstance(self.dtype, str):
            # we will start using self.dtype in v5, but to be consistent with
            # from_pretrained's dtype arg convert it to an actual torch.dtype object
            import torch

            self.dtype = getattr(torch, self.dtype)

        # Keep the default value of `num_labels=2` in case users have saved a classifier with 2 labels
        # Our configs prev wouldn't save `id2label` for 2 labels because it is the default. In all other
        # cases we expect the config dict to have an `id2label` field if it's a clf model, or not otherwise
        if self.id2label is None:
            self.num_labels = kwargs.get("num_labels", self.num_labels if self.num_labels is not None else 2)
        else:
            if kwargs.get("num_labels") is not None and len(self.id2label) != kwargs.get("num_labels"):
                logger.warning(
                    f"You passed `num_labels={kwargs.get('num_labels')}` which is incompatible to "
                    f"the `id2label` map of length `{len(self.id2label)}`."
                )
            # Keys are always strings in JSON so convert ids to int
            self.id2label = {int(key): value for key, value in self.id2label.items()}

        if self.problem_type == "single_label_classification" and self.num_labels == 1:
            raise ValueError(
                '`problem_type="single_label_classification"` requires `num_labels > 1`. For binary '
                'classification use `num_labels=2`, or use `problem_type="regression"` for a '
                "single-output regression head."
            )

        # BC for rotary embeddings. We will pop out legacy keys from kwargs and rename to new format
        if hasattr(self, "rope_parameters"):
            raise CurrentlyNotImplementedError("rope_parameters")
            kwargs = self.convert_rope_params_to_dict(**kwargs)
        elif kwargs.get("rope_scaling") and kwargs.get("rope_theta"):
            raise CurrentlyNotImplementedError("rope_scaling & rope_theta")
            logger.warning(
                f"{self.__class__.__name__} got `key=rope_scaling` in kwargs but hasn't set it as attribute. "
                "For RoPE standardization you need to set `self.rope_parameters` in model's config. "
            )
            kwargs = self.convert_rope_params_to_dict(**kwargs)

        # Parameters for sequence generation saved in the config are popped instead of loading them.
        for parameter_name in GenerationConfig._get_default_generation_params():
            kwargs.pop(parameter_name, None)

        # Name or path to the pretrained checkpoint
        self._name_or_path = str(kwargs.pop("name_or_path", ""))
        # BC: configs saved by older versions may still carry this key, it is not used anymore. The revision of a
        # repository is now resolved once per load and passed around as `revision` (see `utils.hub.resolve_revision`).
        kwargs.pop("_commit_hash", None)

        # Attention/Experts implementation to use, if relevant (it sets it recursively on sub-configs)
        self._output_attentions: bool | None = kwargs.pop("output_attentions", False)
        self._attn_implementation: str | None = kwargs.pop("attn_implementation", None)
        self._experts_implementation: str | None = kwargs.pop("experts_implementation", None)

        # HeterogeneousConfigMixin: `per_layer_config` should be applied last, as heterogeneity needs to have all of the other kwargs set
        per_layer_config = kwargs.pop("per_layer_config", None)

        # Additional attributes without default values
        for key, value in kwargs.items():
            # Check this to avoid deserializing problematic fields from hub configs - they should use the public field
            if key not in ("_attn_implementation_internal", "_experts_implementation_internal"):
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # HeterogeneousConfigMixin
        if per_layer_config is not None:
            raise CurrentlyNotImplementedError(per_layer_config)
            self.per_layer_config = per_layer_config

        # TODO: to support models whose input embedding module is not named `embed_tokens` (e.g. GPT-NeoX's `embed_in`).
        if getattr(self, "tie_word_embeddings", False) and self.base_model_tp_plan is not None:
            raise CurrentlyNotImplementedError(self.tie_word_embeddings)
            self.base_model_tp_plan = {
                **self.base_model_tp_plan,
                "embed_tokens": "embedding_rowwise",
            }

        # Remap layer types if needed
        remap_legacy_layer_types(config=self)


    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls_has_custom_init = "__init__" in cls.__dict__
        # kw_only=True ensures fields without defaults in subclasses can follow
        # parent fields that have defaults (Python dataclass ordering rule).
        # Config fields are always passed as keyword arguments, so this is safe.
        cls = dataclass(cls, repr=False, kw_only=True)

        if not cls_has_custom_init:
            # Wrap all subclasses to accept arbitrary kwargs for BC
            # only if the subclass has no custom `__init__`. Most
            # remote code has an init defined, but some model are not
            # See https://huggingface.co/hmellor/Ilama-3.2-1B/blob/main/configuration_ilama.py
            cls = wrap_init_to_accept_kwargs(cls)

    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            raise CurrentlyNotImplementedError("attribute_map")
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            raise CurrentlyNotImplementedError("attribute_map")
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)
    
    def __eq__(self, other):
        raise CurrentlyNotImplementedError("equal")
        return isinstance(other, ModelConfig) and (self.__dict__ == other.__dict__)

    def __repr__(self):
        raise CurrentlyNotImplementedError("repr")
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def __iter__(self):
        raise CurrentlyNotImplementedError("iter")
        # HeterogeneousConfigMixin: keys of `self.__dict__` that are per-layer attributes
        # may require hiding when using a heterogeneous config.
        yield from self._iter_config_keys_with_heterogeneous_adjustment(self.__dict__)

    #endregion

    #region Property Attribute

    @property
    def name_or_path(self) -> str | None:
        return getattr(self, "_name_or_path", None)

    @name_or_path.setter
    def name_or_path(self, value):
        self._name_or_path = str(value)  # Make sure that name_or_path is a string (for JSON encoding)

    @property
    def num_labels(self) -> int | None:
        """
        `int` or `None`: The number of labels for classification models, or
        `None` when `id2label` is not set.
        """
        return len(self.id2label) if self.id2label is not None else None

    @num_labels.setter
    def num_labels(self, num_labels: int):
        # we do not store `num_labels` attribute in config, but instead
        # compute it based on the length of the `id2label` map
        if self.id2label is None or self.num_labels != num_labels:
            self.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
            self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))

    @property
    def output_attentions(self):
        """
        `bool`: Whether or not the model should returns all attentions.
        """
        raise NotImplementedError(self._attn_implementation)
        return self._output_attentions

    @output_attentions.setter
    def output_attentions(self, value: bool):
        raise NotImplementedError(self._attn_implementation)
        # If we set `output_attentions` explicitly before the attn implementation, dispatch eager
        if value and self._attn_implementation is None:
            self._attn_implementation = "eager"
        if value and self._attn_implementation != "eager":
            raise ValueError(
                "The `output_attentions` attribute is not supported when using the `attn_implementation` set to "
                f"{self._attn_implementation}. Please set it to 'eager' instead."
            )
        self._output_attentions = value

    # Optimization?
    @property
    def _attn_implementation(self):
        return self._attn_implementation_internal

    @_attn_implementation.setter
    def _attn_implementation(self, value: str | dict | None):
        """We set it recursively on the sub-configs as well"""
        # Set if for current config
        current_attn = getattr(self, "_attn_implementation", None)
        attn_implementation = value if not isinstance(value, dict) else value.get("", current_attn)
        self._attn_implementation_internal = attn_implementation

        if self._attn_implementation is not None or self._attn_implementation_internal is not None:
            raise CurrentlyNotImplementedError(self._attn_implementation_internal)
        
        # Set it recursively on the subconfigs
        for subconfig_key in self.sub_configs:
            raise CurrentlyNotImplementedError("Sub Configs")
            subconfig = getattr(self, subconfig_key, None)
            if subconfig is not None:
                current_subconfig_attn = getattr(subconfig, "_attn_implementation", None)
                sub_implementation = (
                    value if not isinstance(value, dict) else value.get(subconfig_key, current_subconfig_attn)
                )
                subconfig._attn_implementation = sub_implementation

    @property
    def _experts_implementation(self):
        return self._experts_implementation_internal

    @_experts_implementation.setter
    def _experts_implementation(self, value: str | dict | None):
        """We set it recursively on the sub-configs as well"""
        # Set if for current config
        current_moe = getattr(self, "_experts_implementation", None)
        experts_implementation = value if not isinstance(value, dict) else value.get("", current_moe)
        self._experts_implementation_internal = experts_implementation

        if self._experts_implementation is not None or self._experts_implementation_internal is not None:
            raise CurrentlyNotImplementedError(self._experts_implementation_internal)

        # Set it recursively on the subconfigs
        for subconfig_key in self.sub_configs:
            raise CurrentlyNotImplementedError("Sub Configs")
            subconfig = getattr(self, subconfig_key, None)
            if subconfig is not None:
                current_subconfig_moe = getattr(subconfig, "_experts_implementation", None)
                sub_implementation = (
                    value if not isinstance(value, dict) else value.get(subconfig_key, current_subconfig_moe)
                )
                subconfig._experts_implementation = sub_implementation

    # Common
    @property
    def torch_dtype(self):
        logger.warning_once("`torch_dtype` is deprecated! Use `dtype` instead!")
        return self.dtype

    @property
    def use_return_dict(self):
        logger.warning_once("`use_return_dict` is deprecated! Use `return_dict` instead!")
        return self.return_dict

    @torch_dtype.setter
    def torch_dtype(self, value):
        logger.warning_once("`torch_dtype` is deprecated! Use `dtype` instead!")
        self.dtype = value
    
    # Rope
    @property
    def rope_scaling(self):
        return self.rope_parameters

    @rope_scaling.setter
    def rope_scaling(self, value):
        if value is not None:
            raise CurrentlyNotImplementedError("rope_scaling")
        self.rope_parameters = value

    #endregion

    #region auto method
    
    @classmethod
    def from_dict(
        cls: type[Self],
        config_dict: dict[str, Any],
        **kwargs
    ) -> Self:
        """
        Instantiates a [`PreTrainedConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the [`~PreTrainedConfig.get_config_dict`] method.
            kwargs (`dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`PreTrainedConfig`]: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # To remove arg here are those passed along for our internal telemetry but we still need to remove them
        to_remove = ["_from_auto", "_from_pipeline"]
        valid_fields = [
            "num_labels",
            "attn_implementation",
            "experts_implementation",
            "output_attentions",
            "torch_dtype",
            "dtype",
            "name_or_path",
        ]
        for key, value in kwargs.items():
            if key in valid_fields:
                if key not in ["torch_dtype", "dtype"]:
                    config_dict[key] = value
                    to_remove.append(key)
                elif value != "auto":
                    config_dict[key] = value

        config = cls(**config_dict)
        for key, value in kwargs.items():
            if hasattr(config, key):
                current_attr = getattr(config, key)
                # To authorize passing a custom subconfig as kwarg in models that have nested configs.
                # We need to update only custom kwarg values instead and keep other attr in subconfig.
                if isinstance(current_attr, ModelConfig) and isinstance(value, dict):
                    current_attr_updated = current_attr.to_dict()
                    current_attr_updated.update(value)
                    value = current_attr.__class__(**current_attr_updated)
                setattr(config, key, value)
                to_remove.append(key)

        for key in to_remove:
            kwargs.pop(key, None)

        logger.info(f"Model config {config}")
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        options: dict | PretrainedOptions | None = None,
        **kwargs,
    ) -> Self:
        r"""
        Instantiate one of the configuration classes of the library from a pretrained model configuration.

        The configuration class to instantiate is selected based on the `model_type` property of the config object that
        is loaded, or when it's missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:
        """
        if not isinstance(options, PretrainedOptions):
            if options is not None:
                assert isinstance(options, dict)
                options = PretrainedOptions.from_dict(options)
            else:
                options = PretrainedOptions.from_dict(kwargs)
        
        config_dict = cls.get_config_dict(
            pretrained_model_name_or_path,
            options
        )
        if cls.base_config_key and cls.base_config_key in config_dict:
            raise CurrentlyNotImplementedError(cls.base_config_key)
        # Auto >>
        if "auto_map" in config_dict:
            # Remote Code
            raise CurrentlyNotImplementedError("Auto mapping")
        if "model_type" in config_dict:
            # Local Code
            if config_dict["model_type"] == "mistral":
                raise CurrentlyNotImplementedError("mistral")
            try:
                architectures = config_dict["architectures"]
                if len(architectures) > 1:
                    raise CurrentlyNotImplementedError("More than 1 architecrues")
                architecture = architectures[0]
                module_path = (
                    "iantirta.models.vendor.transformers.models."
                    + config_dict["model_type"]
                )
                module = importlib.import_module(
                    module_path
                )
                model_class = getattr(module, architecture, None)
                if model_class is None:
                    raise ModuleNotFoundError(model_class)
                else:
                    print(model_class.config)
                cls = model_class.config.__annotations__
            except ImportError as err:
                logger.warning(
                    "No vendor available for "
                    f"{module_path} + {architecture}\n"
                    f"  Error: {err}"
                )
            
            if (
                hasattr(cls, "model_type") and
                config_dict["model_type"] != cls.model_type
            ):
                if not cls.model_type:
                    raise CurrentlyNotImplementedError(
                        "Class model type is none "
                        f"{cls.model_type!r}"
                    )
                # sometimes the config has no `base_config_key` if the config is used in several composite models
                # e.g. LlamaConfig. In that case we try to see if there is match in `model_type` before raising a warning
                for v in config_dict.values():
                    if isinstance(v, dict) and v.get("model_type") == cls.model_type:
                        config_dict = v
                
                # raise warning only if we still can't see a match in `model_type`
                if config_dict["model_type"] != cls.model_type:
                    logger.warning(
                        f"You are using a model of type `{config_dict['model_type']}` to instantiate a model of type "
                        f"`{cls.model_type}`. This may be expected if you are loading a checkpoint that shares a subset "
                        f"of the architecture (e.g., loading a `sam2_video` checkpoint into `Sam2Model`), but is otherwise "
                        f"not supported and can yield errors. Please verify that the checkpoint is compatible with the "
                        f"model you are instantiating."
                    )
                    raise CurrentlyNotImplementedError("Different model_type")
            
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def _decode_special_floats(cls, obj: Any) -> Any:
        """
        Iterates over the passed object and decode specific floats that cannot be JSON-serialized. Python's JSON
        engine saves floats like `Infinity` (+/-) or `NaN` which are not compatible with other JSON engines.

        This method deserializes objects like `{'__float__': Infinity}` to their float values like `Infinity`.
        """
        # Currently only here...
        _FLOAT_TAG_KEY = "__float__"
        _FLOAT_TAG_VALUES = {"Infinity": float("inf"), "-Infinity": float("-inf"), "NaN": float("nan")}

        if isinstance(obj, dict):
            if set(obj.keys()) == {_FLOAT_TAG_KEY} and isinstance(obj[_FLOAT_TAG_KEY], str):
                tag = obj[_FLOAT_TAG_KEY]
                if tag in _FLOAT_TAG_VALUES:
                    return _FLOAT_TAG_VALUES[tag]
                return obj

            return {k: cls._decode_special_floats(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [cls._decode_special_floats(v) for v in obj]

        return obj

    @classmethod
    def _dict_from_json_file(
        cls,
        json_file: Path
    ):
        with json_file.open("r", encoding="utf-8") as reader:
            config_dict = json.load(reader)

        return cls._decode_special_floats(config_dict)
    
    @classmethod
    def _get_config_dict(
        cls,
        pretrained_model_name_or_path: str | Path,
        options: PretrainedOptions,
    ) -> dict[str, Any]:
        
        if options.gguf_file:
            raise CurrentlyNotImplementedError("gguf_file")

        resolved_config_file = ensure_file(
            pretrained_model_name_or_path,
            options._configuration_file,
            revision=options.revision,
        )

        if resolved_config_file is None:
            raise RuntimeError("configuration file not found.")

        try:
            config_dict = cls._dict_from_json_file(resolved_config_file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise OSError(f"It looks like the config file at '{resolved_config_file}' is not a valid JSON file.")

        logger.info(f"loading configuration file {resolved_config_file}")

        if "pretrained_cfg" in config_dict:
            # This is a timm wrapper
            raise CurrentlyNotImplementedError("timm")

        # Some checkpoints may contain the wrong model_type in the config file.
        # Allow the user to override it but warn them that it might not work.
        if options.model_type is not None and config_dict["model_type"] != options.model_type:
            logger.warning(
                f"{options._configuration_file} has 'model_type={config_dict['model_type']}' but you overrode "
                f"it with 'model_type={options.model_type}'. This may lead to unexpected behavior."
            )
            config_dict["model_type"] = options.model_type

        return config_dict
    
    @classmethod
    def get_config_dict(
        cls,
        pretrained_model_name_or_path: str | Path,
        options: PretrainedOptions,
    ) -> dict[str, Any]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        [`PreTrainedConfig`] using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            `tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the configuration object.

        """
        # Get config dict associated with the base config file
        config_dict = cls._get_config_dict(
            pretrained_model_name_or_path,
            options,
        )
        if config_dict is None:
            raise CurrentlyNotImplementedError("Empty config")
            return {}
    
        # That config file may point us toward another config file to use.
        if "configuration_files" in config_dict:
            raise CurrentlyNotImplementedError("Recursive configuration file")
        
        return config_dict

    #endregion

    #region validation

    def validate_output_attentions(self):
        if (
            self.output_attentions and
            self._attn_implementation not in [
                "eager", None
            ]
        ):
            raise ValueError(
                "The `output_attentions` attribute is not supported when using the `attn_implementation` set to "
                f"{self._attn_implementation}. Please set it to 'eager' instead."
            )

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.is_heterogeneous:
            for config in self.per_layer_config:
                config.validate_architecture()
            return
        if (
            hasattr(self, "head_dim")
            and hasattr(self, "num_heads")
            and hasattr(self, "embed_dim")
            and self.head_dim * self.num_heads != self.embed_dim
        ):
            raise ValueError(
                f"The embed_dim ({self.embed_dim}) is not a multiple of the number of attention "
                f"heads ({self.num_heads})."
            )

    def validate_token_ids(self):
        """Part of `@strict`-powered validation. Validates the contents of the special tokens."""
        text_config = self.get_text_config(decoder=True)
        vocab_size = getattr(text_config, "vocab_size", None)
        if vocab_size is not None:
            # Check for all special tokens, e..g. pad_token_id, image_token_id, audio_token_id
            for name in text_config:
                value = getattr(text_config, name)
                if name.endswith("_token_id") and isinstance(value, int) and not 0 <= value < vocab_size:
                    # Can't be an exception until we can load configs that fail validation: several configs on the Hub
                    # store invalid special tokens, e.g. `pad_token_id=-1`
                    logger.warning_once(
                        f"Model config: {name} must be `None` or an integer within the vocabulary (between 0 "
                        f"and {vocab_size - 1}), got {value}. This may result in unexpected behavior."
                    )

    def validate_layer_type(self):
        """Check that `mlp_layer_types` and `layer_types` is correctly defined."""
        for allowed_types, layer_types in zip(
            [ALLOWED_ATTN_LAYER_TYPES, ALLOWED_MLP_LAYER_TYPES], ["layer_types", "mlp_layer_types"]
        ):
            layers = getattr(self, layer_types, None)
            if not (layers is not None and hasattr(self, "num_hidden_layers")):
                return

            if not all(layer_type in allowed_types for layer_type in layers):
                raise ValueError(f"The `{layer_types}` entries must be in {allowed_types} but got {layers}")
            elif self.num_hidden_layers is not None and self.num_hidden_layers != len(layers):
                raise ValueError(
                    f"`num_hidden_layers` ({self.num_hidden_layers}) must be equal to the number of `{layer_types}` "
                    f"({len(layers)})"
                )


    #endregion


