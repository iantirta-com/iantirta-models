
import copy
import json
import logging
import os
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

from iantirta.models import __version__
from iantirta.models.remote.files import cached_file
from iantirta.models.tools.types import ExplicitEnum

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import torch

    from ...common.modeling_utils import PreTrainedModel
    from .pretrained import PreTrainedConfig


# -----
# Heteroginity
# -----

_SENTINEL = object()

class AmbiguousGlobalPerLayerAttributeError(RuntimeError):
    """Raised when a per-layer attribute is
    read from a heterogeneous global config.
    """


def _get_layer_config(
    config: "PreTrainedConfig",
    layer_overrides: dict[str, Any],
) -> "PreTrainedConfig":
    output_config = copy.copy(config)
    output_config.__dict__.pop("_heterogeneity_spec", None)

    output_config.skip = layer_overrides.get("skip", [])

    for attr, value in layer_overrides.items():
        if attr == "skip":
            continue
        setattr(output_config, attr, value)

    return output_config


class _PerLayerConfigView(Sequence["PreTrainedConfig"]):
    def __init__(self, config: "PreTrainedConfig") -> None:
        self._config = config

    def __len__(self) -> int:
        return self._config.num_hidden_layers

    def __getitem__(
        self,
        layer_idx: int | slice | str
    ) -> "PreTrainedConfig | list[PreTrainedConfig]":
        # Return the config for a specific layer type,
        # if the model is homogeneous for that layer type
        if isinstance(layer_idx, str):
            if (layer_types := getattr(
                self._config, "layer_types", None
            )) is None:
                raise ValueError(
                    f"Layer type '{layer_idx}' requested, "
                    "but config.layer_types is not defined. "
                )

            if layer_idx not in layer_types:
                raise ValueError(
                    f"Layer type '{layer_idx}' not found in "
                    "config.layer_types: {layer_types}. "
                    f"Available layer types: {set(layer_types)}"
                )

            # Config is actually homogeneous so just return the global config
            if not self._config.is_heterogeneous:
                return self._config

            # Ensure that all layers of the requested
            # type have the same overrides
            layer_overrides = self._config._heterogeneity_spec.per_layer_overrides
            reference_overrides = layer_overrides.get(
                layer_types.index(layer_idx), {}
            )
            for idx, layer_type in enumerate(layer_types):
                if (
                    layer_type == layer_idx
                    and (
                        layer_overrides.get(idx, {})
                        != reference_overrides
                    )
                ):
                    raise ValueError(
                        f"Layer type '{layer_idx}' is not "
                        f"homogeneous across layers (layer {idx} differs). "
                        "Use an integer index to "
                        "access a specific layer's config."
                    )

            return _get_layer_config(self._config, reference_overrides)

        # Return a list of configs for a slice of layers
        if isinstance(layer_idx, slice):
            return [self[i] for i in range(*layer_idx.indices(len(self)))]

        if layer_idx < 0:
            layer_idx += len(self)
        if layer_idx < 0 or layer_idx >= len(self):
            raise IndexError("list index out of range")

        # Config is actually homogeneous so just return the global config
        if not self._config.is_heterogeneous:
            return self._config

        heterogeneity_spec = self._config._heterogeneity_spec
        return _get_layer_config(
            self._config,
            heterogeneity_spec.per_layer_overrides.get(layer_idx, {}),
        )


def _get_explicit_per_layer_overrides(
    config: "PreTrainedConfig"
) -> dict[int, dict[str, Any]]:
    heterogeneity_spec = config._heterogeneity_spec
    explicit_per_layer_overrides = {}

    for layer_idx in range(config.num_hidden_layers):
        layer_overrides = copy.deepcopy(
            heterogeneity_spec.per_layer_overrides.get(
                layer_idx, {}
            )
        )

        for attr in heterogeneity_spec.explicit_per_layer_attributes:
            if attr not in layer_overrides:
                layer_overrides[attr] = (
                    config._getattr_without_heterogeneous_validation(attr)
                )

        if layer_overrides:
            explicit_per_layer_overrides[layer_idx] = layer_overrides

    return explicit_per_layer_overrides


class HeterogeneousConfigMixin:
    """Mixin for heterogeneous per-layer config behavior.

    This mixin owns heterogeneity-specific state and rules. ``PreTrainedConfig`` assigns the ``per_layer_config``
    property in the post-init phase and calls hook methods where heterogeneity needs to participate in the config lifecycle: attribute
    access, key iteration, and serialization.
    """

    def __getattribute__(self, key: str) -> Any:
        # In heterogeneous configs, per-layer attributes are ambiguous on the global config.
        # Callers must read them from a concrete layer unless they explicitly opt into the global value.
        heterogeneity_spec = super().__getattribute__("__dict__").get("_heterogeneity_spec")
        if heterogeneity_spec is not None:  # noqa: SIM102
            if key in heterogeneity_spec.per_layer_attributes:
                if not super().__getattribute__("allow_global_per_layer_attribute_access"):
                    raise AmbiguousGlobalPerLayerAttributeError(
                        f"'{key}' is a per-layer attribute and may vary across layers. Access it via the individual layer "
                        f"configs instead (e.g. config.per_layer_config[i].{key}). To read the global config value from "
                        f"config.{key} anyway, set `allow_global_per_layer_attribute_access` to `True` on the config. "
                        f"Warning: only do this if the caller can safely handle heterogeneous configs; code that assumes "
                        f"a homogeneous model may use the global value incorrectly."
                    )

                logger.warning_once(
                    f"Reading global config value for per-layer attribute `{key}` on a heterogeneous config. "
                    "Only do this if the caller can safely handle heterogeneous configs; code that assumes a homogeneous "
                    "model may use the global value incorrectly."
                )

        return super().__getattribute__(key)

    @property
    def is_heterogeneous(self) -> bool:
        return hasattr(self, "_heterogeneity_spec")

    @property
    def per_layer_config(self) -> Sequence["PreTrainedConfig"]:
        return _PerLayerConfigView(self)

    @per_layer_config.setter
    def per_layer_config(self, per_layer_config: dict[int | str, dict[str, Any]] | None) -> None:
        if per_layer_config is None:
            self.__dict__.pop("_heterogeneity_spec", None)
            return

        _apply_heterogeneous_config(self, per_layer_config)

    @property
    def per_layer_attributes(self) -> set[str] | None:
        if not self.is_heterogeneous:
            return None
        return self._heterogeneity_spec.per_layer_attributes

    @property
    def allow_global_per_layer_attribute_access(self) -> bool:
        return self.__dict__.get("allow_global_per_layer_attribute_access", False)

    @allow_global_per_layer_attribute_access.setter
    def allow_global_per_layer_attribute_access(self, value: bool) -> None:
        self.__dict__["allow_global_per_layer_attribute_access"] = value

    @property
    def serialize_explicit_per_layer_config(self) -> bool:
        return self.__dict__.get("serialize_explicit_per_layer_config", False)

    @serialize_explicit_per_layer_config.setter
    def serialize_explicit_per_layer_config(self, value: bool) -> None:
        self.__dict__["serialize_explicit_per_layer_config"] = value

    def _iter_config_keys_with_heterogeneous_adjustment(self, keys: Iterable[str]) -> Iterable[str]:
        # Per-layer attributes intentionally raise on direct access and should not be exposed by iteration,
        # unless `allow_global_per_layer_attribute_access` is True.
        if self.is_heterogeneous and not self.allow_global_per_layer_attribute_access:
            for key in keys:
                if key not in self.per_layer_attributes:
                    yield key
        else:
            yield from keys

    def _update_heterogeneous_to_dict_output(self, d: dict[str, Any]) -> None:
        if not self.is_heterogeneous:
            return

        if self.serialize_explicit_per_layer_config:
            per_layer_overrides = _get_explicit_per_layer_overrides(self)
        else:
            per_layer_overrides = self._heterogeneity_spec.per_layer_overrides

        if per_layer_overrides:
            # Zero-pad so keys sort numerically in JSON (0,1,...,10 not 0,1,10,2,...)
            max_digits = len(str(max(per_layer_overrides.keys())))
            d["per_layer_config"] = {
                str(layer_idx).zfill(max_digits): copy.deepcopy(layer_overrides)
                for layer_idx, layer_overrides in per_layer_overrides.items()
            }
        else:
            d["per_layer_config"] = {}

        d.pop("_heterogeneity_spec", None)

    def _getattr_without_heterogeneous_validation(self, key: str, default: Any = _SENTINEL) -> Any:
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]

        try:
            return super().__getattribute__(key)
        except AttributeError:
            if default is _SENTINEL:
                raise
            return default

    def _hasattr_without_heterogeneous_validation(self, key: str) -> bool:
        try:
            self._getattr_without_heterogeneous_validation(key)
        except AttributeError:
            return False
        return True


# -----
# Generation
# -----

METADATA_FIELDS = ("_from_model_config", "_commit_hash", "_original_object_hash", "transformers_version")
STATIC_CACHE_IMPLEMENTATIONS = ("static", "offloaded_static")
DYNAMIC_CACHE_IMPLEMENTATIONS = ("dynamic", "offloaded", "quantized")
# All the following are redundant and deprecated, but kept for BC
DEPRECATED_STATIC_CACHE_IMPLEMENTATIONS = (
    "sliding_window",
    "hybrid",
    "hybrid_chunked",
    "offloaded_hybrid",
    "offloaded_hybrid_chunked",
)
ALL_STATIC_CACHE_IMPLEMENTATIONS = STATIC_CACHE_IMPLEMENTATIONS + DEPRECATED_STATIC_CACHE_IMPLEMENTATIONS
ALL_CACHE_IMPLEMENTATIONS = ALL_STATIC_CACHE_IMPLEMENTATIONS + DYNAMIC_CACHE_IMPLEMENTATIONS


GENERATION_CONFIG_NAME = "generation_config.json"


def _should_warn(outer_attr: str, inner_attr: str, user_set_attributes: set | None) -> bool:
    """Determine if we should raise a warning for the combination `outer_attr` and `inner_attr`, based on whether
    they were provided explicitly, i.e. if they were in `user_set_attributes`.
    For example, if `outer_attr="do_sample"`, the warnings should be suppressed for `inner_attr` flags (e.g. "top_p") that weren't
    explicitly set by the caller. When `do_sample=False` is explicitly required by the user, values such as `top_p` inherited
    from a model's `generation_config.json` are harmless when the user opts for greedy decoding.
    """
    outer_sample_set = user_set_attributes is not None and outer_attr in user_set_attributes
    inner_attr_set = user_set_attributes is not None and inner_attr in user_set_attributes
    # We should warn only if both are explicitly set, none are set, or only the inner_attr is set while outer_attr is not
    return (
        (outer_sample_set and inner_attr_set)
        or (not outer_sample_set and not inner_attr_set)
        or (inner_attr_set and not outer_sample_set)
    )


class GenerationMode(ExplicitEnum):
    """
    Possible generation modes, downstream of the [`~generation.GenerationMixin.generate`] method.
    """

    # Non-beam methods
    CONTRASTIVE_SEARCH = "contrastive_search"
    GREEDY_SEARCH = "greedy_search"
    SAMPLE = "sample"
    ASSISTED_GENERATION = "assisted_generation"
    DOLA_GENERATION = "dola_generation"
    # Beam methods
    BEAM_SEARCH = "beam_search"
    BEAM_SAMPLE = "beam_sample"
    CONSTRAINED_BEAM_SEARCH = "constrained_beam_search"
    GROUP_BEAM_SEARCH = "group_beam_search"


# -----
# RotaryEmbeddingConfig (RoPE)
# -----

class RotaryEmbeddingConfigMixin:
    """
    A Mixin containing the functionality to standardize and validate RoPE parameters.
    """

    default_theta = 10_000.0
    default_rope_type = "default"  # override only for axial models
    ignore_keys_at_rope_validation = set()  # noqa: RUF012

    def nested_rope_parameter_keys(self, rope_parameters: dict) -> list[str]:
        """
        Return the keys `rope_parameters` is nested under, or an empty list if it is a flat dict. Only the layer
        types the config declares count, so a config that declares none is never treated as nested.
        """
        # Deepseekv4 has `layer_types` which are different from `_rope_type_labels`
        labels = getattr(self, "_rope_type_labels", None) or getattr(self, "layer_types", None) or ()
        return [key for key in rope_parameters if key in labels]

    def convert_rope_params_to_dict(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or self.rope_parameters
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}

        # Standardize and validate the correctness of rotary position embeddings parameters. Priority for these parameters is:
        # 1. Values in `rope_parameters` dict (where they should be after standardization)
        # 2. Values in `kwargs` (i.e. it's in config.json but not MyConfig.__init__'s args)
        # 3. Values in the config's attributes (i.e. it's in MyConfig.__init__'s args)
        # 4. Default values (i.e. not present at all but other RoPE parameters are present)
        rope_theta = kwargs.pop("rope_theta", getattr(self, "rope_theta", self.default_theta))
        partial_rotary_factor = kwargs.get("partial_rotary_factor", getattr(self, "partial_rotary_factor", None))

        # When `rope_parameters` is nested, the defaults belong in each nested dict rather than next to them
        nested_keys = self.nested_rope_parameter_keys(self.rope_parameters)
        nested_parameters = [self.rope_parameters[key] for key in nested_keys] or [self.rope_parameters]
        for rope_parameters in nested_parameters:
            if rope_parameters is None:
                continue
            rope_parameters.setdefault("rope_theta", rope_theta)
            if partial_rotary_factor is not None:
                rope_parameters.setdefault("partial_rotary_factor", partial_rotary_factor)

        if partial_rotary_factor is not None:
            self.ignore_keys_at_rope_validation = set(self.ignore_keys_at_rope_validation or []) | {
                "partial_rotary_factor"
            }

        self.standardize_rope_params()
        return kwargs

    def standardize_rope_params(self):
        """
        Helper to standardize the config's rope params field by ensuring the params are defined for each
        later type. For old model the fn will duplicate a single rope param in each layer type (backward compatibility)
        """
        # Move `rope_theta` and `partial_rotary_factor` to the `rope_parameters`, if not there yet
        rope_theta = getattr(self, "rope_theta", None)
        partial_rotary_factor = getattr(self, "partial_rotary_factor", None)
        rope_parameters = getattr(self, "rope_parameters", None) or {}

        nested_keys = self.nested_rope_parameter_keys(rope_parameters)

        # Case 0: no RoPE params defined
        if not (rope_parameters or rope_theta):
            # partial_rotary_factor without rope_theta is invalid, so we don't check for it here
            logger.warning("`standardize_rope_params` was called but no RoPE parameters were found.")
            return
        # Case 1: RoPE params are not nested by layer type -> one global dict
        elif not nested_keys:
            rope_parameters.setdefault("rope_type", rope_parameters.get("type", "default"))
            rope_parameters.setdefault("rope_theta", rope_theta)
            if partial_rotary_factor is not None:
                rope_parameters.setdefault("partial_rotary_factor", partial_rotary_factor)

            # Force set the default type to model's expected `default_rope`. For most models it's a no-op
            # used only to keep BC with old ckpt that require axial rope type
            if self.default_rope_type != "default" and rope_parameters["rope_type"] == "default":
                rope_parameters["rope_type"] = self.default_rope_type

            # Move pretraining-time maximum length to rope parameter dict for RoPE types with scaling
            if rope_parameters["rope_type"] in ["llama3", "yarn", "longrope"]:
                if hasattr(self, "original_max_position_embeddings"):
                    # NOTE: Phi3 (and potentially other models) save `original_max_position_embeddings` field
                    # containing the pretrained value outside rope parameters. This is an exception case where we
                    # give priority to `self.original_max_position_embeddings
                    self.rope_parameters["original_max_position_embeddings"] = self.original_max_position_embeddings
                else:
                    self.rope_parameters.setdefault("original_max_position_embeddings", self.max_position_embeddings)

        # Case 2: different RoPE for each layer -> several params as nested dict
        else:
            for layer_type in nested_keys:
                # skip if saved with `None` value
                if rope_parameters[layer_type] is None:
                    continue
                rope_parameters[layer_type].setdefault("rope_type", rope_parameters[layer_type].get("type", "default"))
                rope_parameters[layer_type].setdefault("rope_theta", rope_theta)
                if partial_rotary_factor is not None:
                    rope_parameters[layer_type].setdefault("partial_rotary_factor", partial_rotary_factor)

                if rope_parameters[layer_type]["rope_type"] in ["llama3", "yarn", "longrope"]:
                    self.rope_parameters[layer_type].setdefault(
                        "original_max_position_embeddings", self.max_position_embeddings
                    )

                # Force set the default type to model's expected `default_rope`. For most models it's a no-op
                # used only to keep BC with old ckpt that require axial rope type
                if self.default_rope_type != "default" and rope_parameters[layer_type]["rope_type"] == "default":
                    rope_parameters[layer_type]["rope_type"] = self.default_rope_type

        self.rope_parameters = rope_parameters

    def validate_rope(self: "PreTrainedConfig"):
        """
        Validate the RoPE config arguments, given a `"PreTrainedConfig"` object
        """
        # Don't validate if no rope_parameters found (`None`) or if it's an empty dict
        # Note that validation runs every time a new config is created, even if config is non-RoPE
        rope_parameters_dict = getattr(self, "rope_parameters", None)
        if not rope_parameters_dict:
            return

        nested_keys = self.nested_rope_parameter_keys(rope_parameters_dict)
        if nested_keys:
            rope_parameters_dict = {key: rope_parameters_dict[key] for key in nested_keys}
        else:
            rope_parameters_dict = {"full_attention": rope_parameters_dict}

        # Heterogeneous configs can't read `head_dim` globally, so the even-dim check is skipped for them
        head_dim = None if self.is_heterogeneous else getattr(self, "head_dim", None)

        for layer_type, rope_parameters in rope_parameters_dict.items():
            # skip when set to `None`, possibly a NoPE layer
            if rope_parameters is None:
                continue
            rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
            validation_fn = getattr(self, f"_validate_{rope_type}_rope_parameters", None)
            rope_parameters["rope_type"] = rope_type

            if validation_fn is not None:
                validation_fn(rope_parameters, ignore_keys=self.ignore_keys_at_rope_validation)
            else:
                logger.warning(
                    f"Missing validation function in 'RotaryEmbeddingConfigMixin' for 'rope_type'='{rope_type}'"
                )

            # An odd partial rotary dim is rounded up and still fits in the head, but a fully-rotated odd head doesn't
            # Synthetic test fixtures and Hub test checkpoints (e.g. tiny-llama) use head_dim <= 4 and are allowed
            partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
            if (
                head_dim is not None
                and head_dim > 4
                and head_dim % 2
                and int(head_dim * partial_rotary_factor) == head_dim
            ):
                raise ValueError(
                    f"RoPE requires an even rotary dimension, but got `head_dim`={head_dim} with "
                    f"`partial_rotary_factor`={partial_rotary_factor} for `{layer_type}`."
                )

    def _validate_axial_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        self._validate_default_rope_parameters(rope_parameters, ignore_keys=ignore_keys)

    def _validate_default_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        required_keys = {"rope_type"}
        optional_keys = {"rope_theta"}
        received_keys = set(rope_parameters.keys())
        rope_type = rope_parameters["rope_type"]
        self._check_received_keys(
            rope_type, received_keys, required_keys, optional_keys=optional_keys, ignore_keys=ignore_keys
        )

    def _validate_linear_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        required_keys = {"rope_type", "factor"}
        optional_keys = {"rope_theta"}
        received_keys = set(rope_parameters.keys())
        rope_type = rope_parameters["rope_type"]
        self._check_received_keys(
            rope_type, received_keys, required_keys, optional_keys=optional_keys, ignore_keys=ignore_keys
        )

        factor = rope_parameters["factor"]
        if factor is None or not isinstance(factor, (float, int)) or factor < 1.0:
            logger.warning(f"`rope_parameters`'s factor field must be a float or int >= 1, got {factor}")

    def _validate_dynamic_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        required_keys = {"rope_type", "factor"}
        optional_keys = {"rope_theta"}
        received_keys = set(rope_parameters.keys())
        rope_type = rope_parameters["rope_type"]
        self._check_received_keys(
            rope_type, received_keys, required_keys, optional_keys=optional_keys, ignore_keys=ignore_keys
        )

        factor = rope_parameters["factor"]
        if factor is None or not isinstance(factor, (float, int)) or factor < 1.0:
            logger.warning(f"`rope_parameters`'s factor field must be a float or int >= 1, got {factor}")

    def _validate_yarn_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        required_keys = {"rope_type", "factor", "original_max_position_embeddings"}
        optional_keys = {
            "rope_theta",
            "attention_factor",
            "beta_fast",
            "beta_slow",
            "mscale",
            "mscale_all_dim",
            "truncate",
        }
        received_keys = set(rope_parameters.keys())
        rope_type = rope_parameters["rope_type"]
        self._check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

        factor = rope_parameters["factor"]
        if factor is None or not isinstance(factor, (float, int)) or factor < 1.0:
            logger.warning(f"`rope_parameters`'s factor field must be a float or int >= 1, got {factor}")

        attention_factor = rope_parameters.get("attention_factor")
        if attention_factor is not None and (not isinstance(attention_factor, float) or attention_factor < 0):
            logger.warning(
                f"`rope_parameters`'s attention_factor field must be a float greater than 0, got {attention_factor}"
            )
        beta_fast = rope_parameters.get("beta_fast")
        if beta_fast is not None and not isinstance(beta_fast, (float, int)):
            logger.warning(f"`rope_parameters`'s beta_fast field must be a float or int, got {beta_fast}")
        beta_slow = rope_parameters.get("beta_slow")
        if beta_slow is not None and not isinstance(beta_slow, (float, int)):
            logger.warning(f"`rope_parameters`'s beta_slow field must be a float or int, got {beta_slow}")

        if (beta_fast or 32) < (beta_slow or 1):
            logger.warning(
                f"`rope_parameters`'s beta_fast field must be greater than beta_slow, got beta_fast={beta_fast} "
                f"(defaults to 32 if None) and beta_slow={beta_slow} (defaults to 1 if None)"
            )

        # Double-check: `factor` should be the ratio between the pre-yarn and post-yarn context lengths.
        # NOTE: we might get `implicit_factor == 1` if config's `original_max_position_embeddings` was
        # inferred from `max_position_embeddings` during standardization
        original_max_position_embeddings = rope_parameters["original_max_position_embeddings"]
        implicit_factor = self.max_position_embeddings / original_max_position_embeddings
        if implicit_factor != factor and implicit_factor != 1:
            logger.warning_once(
                f"The explicitly set RoPE scaling factor (config.rope_parameters['factor'] = {factor}) does not match "
                "the ratio implicitly set by other parameters (implicit factor = "
                "post-yarn context length / pre-yarn context length = "
                "config.max_position_embeddings / config.rope_parameters['original_max_position_embeddings'] = "
                f"{implicit_factor}). Using the explicit factor ({factor}) in YaRN. This may cause unexpected "
                "behaviour in model usage, please correct the 'original_max_position_embeddings' fields in the model config."
            )

    def _validate_longrope_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        required_keys = {"rope_type", "short_factor", "long_factor", "original_max_position_embeddings"}
        optional_keys = {"rope_theta", "attention_factor", "factor"}
        received_keys = set(rope_parameters.keys())
        rope_type = rope_parameters["rope_type"]
        self._check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(self, "head_dim", self.hidden_size // self.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)

        short_factor = rope_parameters.get("short_factor")
        if not (isinstance(short_factor, list) and all(isinstance(x, (int, float)) for x in short_factor)):
            logger.warning(f"`rope_parameters`'s short_factor field must be a list of numbers, got {short_factor}")
        if len(short_factor) != dim // 2:
            logger.warning(
                f"`rope_parameters`'s short_factor field must have length {dim // 2}, got {len(short_factor)}"
            )

        long_factor = rope_parameters.get("long_factor")
        if not (isinstance(long_factor, list) and all(isinstance(x, (int, float)) for x in long_factor)):
            logger.warning(f"`rope_parameters`'s long_factor field must be a list of numbers, got {long_factor}")
        if len(long_factor) != dim // 2:
            logger.warning(
                f"`rope_parameters`'s long_factor field must have length {dim // 2}, got {len(long_factor)}"
            )

        factor = rope_parameters.get("factor")
        original_max_position_embeddings = rope_parameters["original_max_position_embeddings"]

        # Handle Phi3 divergence: we prefer the use of `attention_factor` and/or `factor` over
        # `original_max_position_embeddings` to compute internal variables. The latter is undesirable
        if factor is None and original_max_position_embeddings is not None:
            logger.warning_once(
                "This model config has set a `rope_parameters['original_max_position_embeddings']` field, to be used together with "
                "`max_position_embeddings` to determine a scaling factor. Please set the `factor` field of `rope_parameters`"
                "with this ratio instead -- we recommend the use of this field over `original_max_position_embeddings`, "
                "as it is compatible with most model architectures."
            )
        elif factor is None and original_max_position_embeddings is None:
            logger.warning("Missing required keys in `rope_parameters`: 'factor'")
        elif not isinstance(factor, (float, int)) or factor < 1.0:
            logger.warning(f"`rope_parameters`'s factor field must be a float or int >= 1, got {factor}")

        attention_factor = rope_parameters.get("attention_factor")
        if attention_factor is not None and (not isinstance(attention_factor, (float, int)) or attention_factor < 0.0):
            logger.warning(
                f"`rope_parameters`'s attention_factor field must be a float or int greater than 0, got {attention_factor}"
            )

    def _validate_llama3_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        required_keys = {
            "rope_type",
            "factor",
            "original_max_position_embeddings",
            "low_freq_factor",
            "high_freq_factor",
            "rope_theta",
        }
        rope_type = rope_parameters["rope_type"]
        received_keys = set(rope_parameters.keys())
        self._check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

        factor = rope_parameters["factor"]
        if factor is None or not isinstance(factor, (float, int)) or factor < 1.0:
            logger.warning(f"`rope_parameters`'s factor field must be a float or int >= 1, got {factor}")

        low_freq_factor = rope_parameters["low_freq_factor"]
        high_freq_factor = rope_parameters["high_freq_factor"]
        if low_freq_factor is None or not isinstance(low_freq_factor, (float, int)):
            logger.warning(f"`rope_parameters`'s low_freq_factor field must be a float, or int got {low_freq_factor}")
        if high_freq_factor is None or not isinstance(high_freq_factor, (float, int)):
            logger.warning(
                f"`rope_parameters`'s high_freq_factor field must be a float or int, got {high_freq_factor}"
            )
        if high_freq_factor <= low_freq_factor:
            logger.warning(
                "`rope_parameters`'s high_freq_factor field must be greater than low_freq_factor, got high_freq_factor="
                f"{high_freq_factor} and low_freq_factor={low_freq_factor}"
            )

        original_max_position_embeddings = rope_parameters["original_max_position_embeddings"]
        if original_max_position_embeddings is None or not isinstance(original_max_position_embeddings, int):
            logger.warning(
                "`rope_parameters`'s original_max_position_embeddings field must be an integer, got "
                f"{original_max_position_embeddings}"
            )
        if original_max_position_embeddings >= self.max_position_embeddings:
            logger.warning(
                "`rope_parameters`'s original_max_position_embeddings field must be less than max_position_embeddings, got "
                f"{original_max_position_embeddings} and max_position_embeddings={self.max_position_embeddings}"
            )

    def _validate_proportional_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        required_keys = {"rope_type", "rope_theta"}
        rope_type = rope_parameters["rope_type"]
        received_keys = set(rope_parameters.keys())
        self._check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

        partial_rotary_factor = rope_parameters.get("partial_rotary_factor")
        if partial_rotary_factor is None:
            logger.warning(
                "`rope_parameters`'s partial_rotary_factor is None. This will default to 1.0 in the computation, "
                "making this equivalent to the linear_scaling RoPE type. Provide a value in the range [0.0, 1.0) to "
                "make use of the proportional RoPE functionality."
            )

    @staticmethod
    def _check_received_keys(
        rope_type: str,
        received_keys: set,
        required_keys: set,
        optional_keys: set | None = None,
        ignore_keys: set | None = None,
    ):
        """Compare the received keys in `config.rope_parameters` against the expected and optional keys"""
        # BC: "rope_type" was originally "type" -- let's check for "rope_type" when "type" is present
        if "type" in received_keys:
            received_keys -= {"type"}
            required_keys.add("rope_type")

        optional_keys = optional_keys or set()
        if "partial_rotary_factor" not in optional_keys:
            optional_keys.add("partial_rotary_factor")

        # Some models need to store model-specific keys, and we don't want to throw warning at them
        if ignore_keys is not None:
            received_keys -= set(ignore_keys)

        missing_keys = required_keys - received_keys
        if missing_keys:
            raise KeyError(f"Missing required keys in `rope_parameters` for 'rope_type'='{rope_type}': {missing_keys}")

        unused_keys = received_keys - required_keys - optional_keys
        if unused_keys:
            logger.warning(f"Unrecognized keys in `rope_parameters` for 'rope_type'='{rope_type}': {unused_keys}")
