
from dataclasses import dataclass
import os
from typing import Any, TYPE_CHECKING
import copy
from pathlib import Path
import json
from typing_extensions import Self
import logging
from collections.abc import Sequence

from iantirta.models.remote.files import cached_file

if TYPE_CHECKING:
    from .pretrained import PreTrainedConfig

logger = logging.getLogger(__name__)

_FLOAT_TAG_KEY = "__float__"
_FLOAT_TAG_VALUES = {
    "Infinity": float("inf"),
    "-Infinity": float("-inf"),
    "NaN": float("nan")
}
_SENTINEL = object()


@dataclass
class ConfigMixin:
    """ Base class only for remote download resolving.
    and simple instancing.
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        **kwargs
    ):
        raise NotImplementedError()

    @classmethod
    def get_config_dict(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:

        original_kwargs = copy.deepcopy(kwargs)
        # Get config dict associated with the base config file
        config_dict, kwargs = cls._get_config_dict(
            pretrained_model_name_or_path,
            **kwargs
        )
        if config_dict is None:
            return {}, kwargs

        # That config file may point us toward another config file to use.
        if "configuration_files" in config_dict:
            raise NotImplementedError()
            # configuration_file = get_configuration_file(config_dict["configuration_files"])
            # config_dict, kwargs = cls._get_config_dict(
            #     pretrained_model_name_or_path, _configuration_file=configuration_file, **original_kwargs
            # )

        return config_dict, kwargs

    @classmethod
    def _get_config_dict(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Main Download Entry."""
        cache_dir = kwargs.pop("cache_dir", None)
        _ = kwargs.pop("force_download", False)
        _ = kwargs.pop("proxies", None)
        _ = kwargs.pop("token", None)
        _ = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        _ = kwargs.pop("trust_remote_code", None)
        subfolder = kwargs.pop("subfolder", "").strip("/")

        gguf_file = kwargs.get("gguf_file")

        if (
            config_file := (Path(subfolder) / pretrained_model_name_or_path)
        ).is_file():
            pass
        else:
            configuration_file = (
                kwargs.pop(
                    "_configuration_file", "config.json"
                ) if gguf_file is None else gguf_file
            )
            try:
                config_file = cached_file(
                    pretrained_model_name_or_path,
                    configuration_file,
                    cache_dir=cache_dir,
                    revision=revision,
                    subfolder=subfolder,
                )
                if config_file is None:
                    return None, kwargs
            except Exception:  # noqa: BLE001
                # For any other exception, we throw a generic error.
                raise OSError(
                    "Can't load the configuration of "
                    f"'{pretrained_model_name_or_path}'. "
                    "If you were trying to load it "
                    "from 'https://huggingface.co/models', "
                    "make sure you don't have a local directory with the same"
                    " name. Otherwise, make sure "
                    f"'{pretrained_model_name_or_path}' "
                    "is the correct path to a directory"
                    f" containing a {configuration_file} file"
                )

        try:
            if gguf_file:
                raise NotImplementedError()
            else:
                config_dict = cls._dict_from_json_file(
                    config_file
                )
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise OSError(
                "It looks like the config file at "
                f"'{config_file}' is not a "
                "valid JSON file."
            )

        assert "model_type" in config_dict
        return config_dict, kwargs

    @classmethod
    def _dict_from_json_file(
        cls,
        json_file: str | os.PathLike | Path
    ):
        with Path(json_file).open(encoding="utf-8") as reader:
            config_dict = json.loads(reader)

        return cls._decode_special_floats(config_dict)

    @classmethod
    def _decode_special_floats(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            if (
                set(obj.keys()) == {_FLOAT_TAG_KEY}
                and isinstance(obj[_FLOAT_TAG_KEY], str)
            ):
                tag = obj[_FLOAT_TAG_KEY]
                if tag in _FLOAT_TAG_VALUES:
                    return _FLOAT_TAG_VALUES[tag]
                return obj

            return {k: cls._decode_special_floats(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [cls._decode_special_floats(v) for v in obj]

        return obj

    @classmethod
    def from_dict(
        cls,
        config_dict: dict[str, Any],
        **kwargs
    ) -> Self:
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # To remove arg here are those passed
        # along for our internal telemetry but
        # we still need to remove them
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
                # To authorize passing a custom subconfig
                # as kwarg in models that have nested configs.
                # We need to update only custom kwarg values
                # instead and keep other attr in subconfig.
                if (
                    isinstance(current_attr, ConfigMixin)
                    and isinstance(value, dict)
                ):
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

    def to_dict(self) -> dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type

        # Pop "kwargs" since they are unpacked and set in the post init
        output.pop("kwargs", None)

        def to_list(value):
            if isinstance(value, tuple):
                value = [to_list(item) for item in value]
            return value

        for key, value in output.items():
            # Deal with nested configs like CLIP
            if isinstance(value, ConfigMixin):
                value = value.to_dict()

            # Some models have defaults as tuples because dataclass
            # doesn't allow mutables. Let's convert back to `list``
            elif isinstance(value, tuple):
                value = to_list(value)

            output[key] = value

        self._remove_keys_not_serialized(output)

        if hasattr(self, "quantization_config"):
            output["quantization_config"] = (
                self.quantization_config.to_dict()
                if not isinstance(
                    self.quantization_config, dict
                )
                and self.quantization_config is not None
                else self.quantization_config
            )
        self.dict_dtype_to_str(output)

        # HeterogeneousConfigMixin: update the serialized output.
        self._update_heterogeneous_to_dict_output(output)

        return output

    def dict_dtype_to_str(self, d: dict[str, Any]) -> None:
        if d.get("dtype") is not None:
            if isinstance(d["dtype"], dict):
                d["dtype"] = {
                    k: str(v).split(".")[-1]
                    for k, v in d["dtype"].items()
                }
            # models like Emu3 can have "dtype"
            # as token in config's vocabulary map,
            # so we also exclude int type here to
            # avoid error in this special case.
            elif not isinstance(d["dtype"], (str, int)):
                d["dtype"] = str(d["dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self.dict_dtype_to_str(value)

    def _remove_keys_not_serialized(self, d: dict[str, Any]) -> None:
        for key_to_remove in [
            "_is_quantized",
            "_auto_class",
            "_commit_hash",
            "_attn_implementation_internal",
            "_experts_implementation_internal",
            "ignore_keys_at_rope_validation",
            "base_model_tp_plan",
            "base_model_pp_plan",
            "base_model_fsdp_plan",
            "distributed_config",
        ]:
            d.pop(key_to_remove, None)

        if "_output_attentions" in d:
            d["output_attentions"] = d.pop("_output_attentions")

        for value in d.values():
            if isinstance(value, dict):
                self._remove_keys_not_serialized(value)


# -----
# Heteroginity
# -----


class AmbiguousGlobalPerLayerAttributeError(RuntimeError):
    """Raised when a per-layer attribute is
    read from a heterogeneous global config.
    """


def _get_layer_config(
    config: PreTrainedConfig,
    layer_overrides: dict[str, Any],
) -> PreTrainedConfig:
    output_config = copy.copy(config)
    output_config.__dict__.pop("_heterogeneity_spec", None)

    output_config.skip = layer_overrides.get("skip", [])

    for attr, value in layer_overrides.items():
        if attr == "skip":
            continue
        setattr(output_config, attr, value)

    return output_config


class _PerLayerConfigView(Sequence["PreTrainedConfig"]):
    def __init__(self, config: PreTrainedConfig) -> None:
        self._config = config

    def __len__(self) -> int:
        return self._config.num_hidden_layers

    def __getitem__(
        self,
        layer_idx: int | slice | str
    ) -> PreTrainedConfig | list[PreTrainedConfig]:
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
    config: PreTrainedConfig
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

    def __getattribute__(self, key: str) -> Any:
        # In heterogeneous configs, per-layer attributes are ambiguous on the global config.
        # Callers must read them from a concrete layer unless they explicitly opt into the global value.
        heterogeneity_spec = super().__getattribute__("__dict__").get("_heterogeneity_spec")
        if heterogeneity_spec is not None and key in heterogeneity_spec.per_layer_attributes:
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
    def per_layer_config(
        self,
        per_layer_config: dict[int | str, dict[str, Any]] | None
    ) -> None:
        if per_layer_config is None:
            self.__dict__.pop("_heterogeneity_spec", None)
            return

    @property
    def serialize_explicit_per_layer_config(self) -> bool:
        return self.__dict__.get("serialize_explicit_per_layer_config", False)

    @serialize_explicit_per_layer_config.setter
    def serialize_explicit_per_layer_config(self, value: bool) -> None:
        self.__dict__["serialize_explicit_per_layer_config"] = value

    @property
    def per_layer_attributes(self) -> set[str] | None:
        if not self.is_heterogeneous:
            return None
        return self._heterogeneity_spec.per_layer_attributes

    @property
    def allow_global_per_layer_attribute_access(self) -> bool:
        return self.__dict__.get(
            "allow_global_per_layer_attribute_access",
            False
        )

    @allow_global_per_layer_attribute_access.setter
    def allow_global_per_layer_attribute_access(self, value: bool) -> None:
        self.__dict__["allow_global_per_layer_attribute_access"] = value

    def _update_heterogeneous_to_dict_output(self, d: dict[str, Any]) -> None:
        if not self.is_heterogeneous:
            return

        if self.serialize_explicit_per_layer_config:
            per_layer_overrides = _get_explicit_per_layer_overrides(self)
        else:
            per_layer_overrides = self._heterogeneity_spec.per_layer_overrides

        if per_layer_overrides:
            # Zero-pad so keys sort numerically
            # in JSON (0,1,...,10 not 0,1,10,2,...)
            max_digits = len(str(max(per_layer_overrides.keys())))
            d["per_layer_config"] = {
                str(layer_idx).zfill(max_digits): copy.deepcopy(layer_overrides)
                for layer_idx, layer_overrides in per_layer_overrides.items()
            }
        else:
            d["per_layer_config"] = {}

        d.pop("_heterogeneity_spec", None)

    def _getattr_without_heterogeneous_validation(
        self,
        key: str,
        default: Any = _SENTINEL
    ) -> Any:
        if (
            key != "attribute_map"
            and key in super().__getattribute__("attribute_map")
        ):
            key = super().__getattribute__("attribute_map")[key]

        try:
            return super().__getattribute__(key)
        except AttributeError:
            if default is _SENTINEL:
                raise
            return default
