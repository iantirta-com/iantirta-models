from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .configuration_mixin import ConfigMixin

_SENTINEL = object()

class AmbiguousGlobalPerLayerAttributeError(RuntimeError):
    """Raised when a per-layer attribute is read from a heterogeneous global config."""


@dataclass
class _HeterogeneitySpec:
    per_layer_overrides: dict[int, dict[str, Any]]
    per_layer_attributes: set[str]
    explicit_per_layer_attributes: set[str]


class _PerLayerConfigView(Sequence["PreTrainedConfig"]):
    def __init__(self, config: PreTrainedConfig) -> None:
        self._config = config

    def __len__(self) -> int:
        return self._config.num_hidden_layers

    def __getitem__(self, layer_idx: int | slice | str) -> PreTrainedConfig | list[PreTrainedConfig]:
        # Return the config for a specific layer type, if the model is homogeneous for that layer type
        if isinstance(layer_idx, str):
            if (layer_types := getattr(self._config, "layer_types", None)) is None:
                raise ValueError(f"Layer type '{layer_idx}' requested, but config.layer_types is not defined. ")

            if layer_idx not in layer_types:
                raise ValueError(
                    f"Layer type '{layer_idx}' not found in config.layer_types: {layer_types}. "
                    f"Available layer types: {set(layer_types)}"
                )

            # Config is actually homogeneous so just return the global config
            if not self._config.is_heterogeneous:
                return self._config

            # Ensure that all layers of the requested type have the same overrides
            layer_overrides = self._config._heterogeneity_spec.per_layer_overrides
            reference_overrides = layer_overrides.get(layer_types.index(layer_idx), {})
            for idx, layer_type in enumerate(layer_types):
                if layer_type == layer_idx and layer_overrides.get(idx, {}) != reference_overrides:
                    raise ValueError(
                        f"Layer type '{layer_idx}' is not homogeneous across layers (layer {idx} differs). "
                        f"Use an integer index to access a specific layer's config."
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


class HeterogeneousConfigMixin(ConfigMixin):
    """Mixin for heterogeneous per-layer config behavior.

    This mixin owns heterogeneity-specific state and rules. ``PreTrainedConfig`` assigns the ``per_layer_config``
    property in the post-init phase and calls hook methods where heterogeneity needs to participate in the config lifecycle: attribute
    access, key iteration, and serialization.
    """

    @staticmethod
    def _normalize_layer_overrides(layer_overrides: dict[str, Any]) -> dict[str, Any]:
        normalized = copy.deepcopy(layer_overrides)
    
        if "skip" in normalized:
            skip = normalized.pop("skip")
            if isinstance(skip, str) or not isinstance(skip, Iterable):
                raise TypeError("`skip` must be an iterable of strings.")
    
            skip = set(skip)
            if not all(isinstance(item, str) for item in skip):
                raise TypeError("`skip` must contain only strings.")
    
            if skip:
                normalized["skip"] = sorted(skip)
    
        return normalized

    @staticmethod
    def _validate_layer_indices(config: PreTrainedConfig, per_layer_overrides: dict[int, dict[str, Any]]) -> None:
        if not per_layer_overrides:
            return
    
        num_hidden_layers = config.num_hidden_layers
        invalid_layer_indices = [
            layer_idx for layer_idx in per_layer_overrides if layer_idx < 0 or layer_idx >= num_hidden_layers
        ]
        if invalid_layer_indices:
            raise ValueError(
                f"`per_layer_config` keys must be integer layer indices in the range [0, {num_hidden_layers}); "
                f"got {invalid_layer_indices}."
            )

    @staticmethod
    def _validate_sliding_window_and_attention_chunk_size(
        config: PreTrainedConfig, per_layer_overrides: dict[int, dict[str, Any]]
    ) -> None:
        problematic_indices = []
        for layer_idx in range(config.num_hidden_layers):
            layer_overrides = per_layer_overrides.get(layer_idx, {})
    
            sliding_window = layer_overrides.get(
                "sliding_window", config._getattr_without_heterogeneous_validation("sliding_window", None)
            )
            attention_chunk_size = layer_overrides.get(
                "attention_chunk_size",
                config._getattr_without_heterogeneous_validation("attention_chunk_size", None),
            )
    
            if sliding_window is not None and attention_chunk_size is not None:
                problematic_indices.append(layer_idx)
    
        if problematic_indices:
            raise ValueError(
                f"The following layers have the mutually exclusive `sliding_window` and `attention_chunk_size` both defined: "
                f"{problematic_indices}. To fix this, either remove a conflicting attribute from the global config,"
                f"or set it to `None` in `per_layer_config` for the problematic layers."
            )

    def _get_per_layer_attributes(per_layer_overrides: dict[int, dict[str, Any]]) -> set[str]:
        per_layer_attributes: set[str] = set()
        for layer_overrides in per_layer_overrides.values():
            per_layer_attributes.update(layer_overrides)
    
        per_layer_attributes.discard("skip")
        return per_layer_attributes

    @staticmethod
    def _modify_config_and_create_heterogeneity_spec(
        config: PreTrainedConfig, per_layer_overrides: dict[int, dict[str, Any]]
    ) -> _HeterogeneitySpec:
        explicit_per_layer_attributes = HeterogeneousConfigMixin._get_per_layer_attributes(per_layer_overrides)
    
        # Ensure all required global attributes are defined
        missing_required_global_attributes = set()
        for attr in explicit_per_layer_attributes:
            if len(per_layer_overrides) != config.num_hidden_layers:
                if not config._hasattr_without_heterogeneous_validation(attr):
                    missing_required_global_attributes.add(attr)
            else:
                for layer_overrides in per_layer_overrides.values():
                    if attr not in layer_overrides:
                        if not config._hasattr_without_heterogeneous_validation(attr):
                            missing_required_global_attributes.add(attr)
                        break
    
        if missing_required_global_attributes:
            raise ValueError(
                f"The following attributes are missing: {sorted(missing_required_global_attributes)}\nPlease define them globally, or provide them for every layer in `per_layer_config`"
            )
    
        # Remove per-layer overrides that match the global value
        for attr in explicit_per_layer_attributes:
            if not config._hasattr_without_heterogeneous_validation(attr):
                continue
    
            global_value = config._getattr_without_heterogeneous_validation(attr)
            for layer_overrides in per_layer_overrides.values():
                if attr in layer_overrides and layer_overrides[attr] == global_value:
                    del layer_overrides[attr]
    
        # Delete all empty layer configs
        for layer_idx, layer_overrides in list(per_layer_overrides.items()):
            if not layer_overrides:
                del per_layer_overrides[layer_idx]
    
        per_layer_attributes = HeterogeneousConfigMixin._get_per_layer_attributes(per_layer_overrides)
        heterogeneity_spec = _HeterogeneitySpec(
            per_layer_overrides=per_layer_overrides,
            per_layer_attributes=per_layer_attributes,
            explicit_per_layer_attributes=explicit_per_layer_attributes,
        )
    
        return heterogeneity_spec

    @staticmethod
    def _apply_heterogeneous_config(
        config: PreTrainedConfig,
        per_layer_config: dict[int | str, dict[str, Any]],
    ) -> None:
        """Register per-layer configuration overrides on a model config.
    
        In a heterogeneous model, individual layers can differ from the global config
        (e.g., different ``intermediate_size``, ``num_key_value_heads``, or entire
        sub-layers skipped via the ``skip`` attribute).
    
        This function validates the overrides and stores a ``_HeterogeneitySpec`` on ``config._heterogeneity_spec``.
        At model-init time, ``apply_heterogeneous_modeling`` reads this spec to patch
        each layer with its resolved config.
    
        Args:
            config: The global model config to modify in-place.
            per_layer_config: Mapping from layer index to a dictionary
                of attribute overrides. Only layers that differ from the global
                config need to be included.
        """
    
        normalized_per_layer_overrides = {
            int(layer_idx): HeterogeneousConfigMixin._normalize_layer_overrides(layer_overrides)
            for layer_idx, layer_overrides in per_layer_config.items()
        }
    
        HeterogeneousConfigMixin._validate_layer_indices(config, normalized_per_layer_overrides)
        HeterogeneousConfigMixin._validate_sliding_window_and_attention_chunk_size(config, normalized_per_layer_overrides)
    
        config._heterogeneity_spec = HeterogeneousConfigMixin._modify_config_and_create_heterogeneity_spec(config, normalized_per_layer_overrides)

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

        # HeterogeneousConfigMixin: `per_layer_config` should be applied last, as heterogeneity needs to have all of the other kwargs set
        per_layer_config = kwargs.pop("per_layer_config", None)
        if per_layer_config is not None:
            self.per_layer_config = per_layer_config


    def __getattribute__(self, key: str) -> Any:
        # In heterogeneous configs, per-layer attributes are ambiguous on the global config.
        # Callers must read them from a concrete layer unless they explicitly opt into the global value.
        heterogeneity_spec = super().__getattribute__("__dict__").get("_heterogeneity_spec")
        if heterogeneity_spec is not None:
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
    def per_layer_config(self) -> Sequence[PreTrainedConfig]:
        return _PerLayerConfigView(self)

    @per_layer_config.setter
    def per_layer_config(self, per_layer_config: dict[int | str, dict[str, Any]] | None) -> None:
        if per_layer_config is None:
            self.__dict__.pop("_heterogeneity_spec", None)
            return

        self._apply_heterogeneous_config(self, per_layer_config)

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

    @staticmethod
    def _get_explicit_per_layer_overrides(config: PreTrainedConfig) -> dict[int, dict[str, Any]]:
        heterogeneity_spec = config._heterogeneity_spec
        explicit_per_layer_overrides = {}
    
        for layer_idx in range(config.num_hidden_layers):
            layer_overrides = copy.deepcopy(heterogeneity_spec.per_layer_overrides.get(layer_idx, {}))
    
            for attr in heterogeneity_spec.explicit_per_layer_attributes:
                if attr not in layer_overrides:
                    layer_overrides[attr] = config._getattr_without_heterogeneous_validation(attr)
    
            if layer_overrides:
                explicit_per_layer_overrides[layer_idx] = layer_overrides
    
        return explicit_per_layer_overrides

    def _update_heterogeneous_to_dict_output(self, d: dict[str, Any]) -> None:
        if not self.is_heterogeneous:
            return

        if self.serialize_explicit_per_layer_config:
            per_layer_overrides = self._get_explicit_per_layer_overrides(self)
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

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.is_heterogeneous:
            for config in self.per_layer_config:
                config.validate_architecture()
        else:
            super().validate_architecture()
        
    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = super().to_dict()

        # HeterogeneousConfigMixin: update the serialized output.
        self._update_heterogeneous_to_dict_output(output)

        return output

    def __iter__(self):
        # HeterogeneousConfigMixin: keys of `self.__dict__` that are per-layer attributes
        # may require hiding when using a heterogeneous config.
        yield from self._iter_config_keys_with_heterogeneous_adjustment(self.__dict__)


if __name__ == "__main__":
    from rich import inspect as i
    import time
    start_time = time.time()
    config = HeterogeneousConfigMixin()
    end_time = time.time()
    i(config)
    print(config)
    print("Initialized Within", end_time - start_time)
