import json
import math
import copy
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar
from .configuration_utils import (
    logger,
    _FLOAT_TAG_KEY,
    _FLOAT_TAG_VALUES,
)
from typing_extensions import dataclass_transform
from dataclasses import fields, MISSING, dataclass
from ... import __version__
from ...strict_dataclasses import strict, wrap_init_to_accept_kwargs


@dataclass_transform(kw_only_default=True)
@strict(accept_kwargs=True)
@dataclass(repr=False)
class ConfigMixin:
    # Class attributes that we don't want to save or have in `self.__dict__`
    # They are not supposed to be set/changed by users. Each field is set when
    # creating a model class
    _auto_class: ClassVar[str | None] = None

    # Attributes set internally when saving and used to infer model
    # class for `Auto` mapping
    model_type: ClassVar[str] = ""
    transformers_version: str | None = None
    architectures: list[str] | None = None

    # Common attributes for all models
    output_hidden_states: bool | None = False
    return_dict: bool | None = True
    dtype: str | torch.dtype | None = None
    chunk_size_feed_forward: int = 0
    is_encoder_decoder: bool = False

    # Fine-tuning task arguments
    id2label: dict[int, str] | dict[str, str] | None = None
    label2id: dict[str, int] | dict[str, str] | None = None
    problem_type: Literal["regression", "single_label_classification", "multi_label_classification"] | None = None

    def __post_init__(self, **kwargs) -> None:
        if hasattr(super(), '__post_init__'):
            raise RuntimeError()

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

        # Name or path to the pretrained checkpoint
        self._name_or_path = str(kwargs.pop("name_or_path", ""))
        # BC: configs saved by older versions may still carry this key, it is not used anymore. The revision of a
        # repository is now resolved once per load and passed around as `revision` (see `utils.hub.resolve_revision`).
        kwargs.pop("_commit_hash", None)

        # Additional attributes without default values
        for key, value in kwargs.items():
            # Check this to avoid deserializing problematic fields from hub configs - they should use the public field
            if key not in ("_attn_implementation_internal", "_experts_implementation_internal"):
                try:
                    logger.warning(
                        "Additional attribute without default value:\n"
                        f"    ({key} - {value})"
                    )
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

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
    def name_or_path(self) -> str | None:
        return getattr(self, "_name_or_path", None)

    @name_or_path.setter
    def name_or_path(self, value):
        self._name_or_path = str(value)  # Make sure that name_or_path is a string (for JSON encoding)

    @property
    def torch_dtype(self):
        logger.warning("`torch_dtype` is deprecated! Use `dtype` instead!")
        return self.dtype

    @torch_dtype.setter
    def torch_dtype(self, value):
        logger.warning("`torch_dtype` is deprecated! Use `dtype` instead!")
        self.dtype = value

    @property
    def use_return_dict(self):
        logger.warning("`use_return_dict` is deprecated! Use `return_dict` instead!")
        return self.return_dict

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if hasattr(super(), "validate_architecture"):
            raise RuntimeError()

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

    def _remove_keys_not_serialized(self, d: dict[str, Any]) -> None:
        """
        Checks and removes if there are any keys in the dict that should not be serialized when saving the config.
        Runs recursive check on the dict, to remove from all sub configs.
        """

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

    def dict_dtype_to_str(self, d: dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a *dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        if d.get("dtype") is not None:
            if isinstance(d["dtype"], dict):
                d["dtype"] = {k: str(v).split(".")[-1] for k, v in d["dtype"].items()}
            # models like Emu3 can have "dtype" as token in config's vocabulary map,
            # so we also exclude int type here to avoid error in this special case.
            elif not isinstance(d["dtype"], (str, int)):
                d["dtype"] = str(d["dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self.dict_dtype_to_str(value)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        if hasattr(super(), 'to_dict'):
            raise RuntimeError()

        output = copy.deepcopy(self.__dict__)

        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type

        # Transformers version when serializing the model
        output["transformers_version"] = __version__

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
                del value["transformers_version"]

            # Some models have defaults as tuples because dataclass
            # doesn't allow mutables. Let's convert back to `list``
            elif isinstance(value, tuple):
                value = to_list(value)

            output[key] = value

        self._remove_keys_not_serialized(output)

        if hasattr(self, "quantization_config"):
            output["quantization_config"] = (
                self.quantization_config.to_dict()
                if not isinstance(self.quantization_config, dict) and self.quantization_config is not None
                else self.quantization_config
            )
        self.dict_dtype_to_str(output)

        return output

    @classmethod
    def _encode_special_floats(cls, obj: Any) -> Any:
        """
        Iterates over the passed object and encode specific floats that cannot be JSON-serialized. Python's JSON
        engine saves floats like `Infinity` (+/-) or `NaN` which are not compatible with other JSON engines.

        It serializes floats like `Infinity` as an object: `{'__float__': Infinity}`.
        """
        if isinstance(obj, float):
            if math.isnan(obj):
                return {_FLOAT_TAG_KEY: "NaN"}
            if obj == float("inf"):
                return {_FLOAT_TAG_KEY: "Infinity"}
            if obj == float("-inf"):
                return {_FLOAT_TAG_KEY: "-Infinity"}
            return obj

        if isinstance(obj, dict):
            return {k: cls._encode_special_floats(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [cls._encode_special_floats(v) for v in obj]

        return obj

    @classmethod
    def _decode_special_floats(cls, obj: Any) -> Any:
        """
        Iterates over the passed object and decode specific floats that cannot be JSON-serialized. Python's JSON
        engine saves floats like `Infinity` (+/-) or `NaN` which are not compatible with other JSON engines.

        This method deserializes objects like `{'__float__': Infinity}` to their float values like `Infinity`.
        """
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

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        config_dict = self.to_dict()

        # Handle +/-Infinity and NaNs
        config_dict = self._encode_special_floats(config_dict)

        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def __eq__(self, other):
        return isinstance(other, PreTrainedConfig) and (self.__dict__ == other.__dict__)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoConfig"):
        """
        Register this class with a given auto class. This should only be used for custom configurations as the ones in
        the library are already mapped with `AutoConfig`.



        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoConfig"`):
                The auto class to register this new configuration with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        from .. import auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class
    
    @classmethod
    def is_remote_code(cls) -> bool:
        """Return whether the current config is custom code, i.e. code loaded from the hub, or class that we just
        registered via `register_for_auto_class`."""
        return cls._auto_class is not None

    @classmethod
    def is_custom_code(cls) -> bool:
        """Return whether the current config is custom code, i.e. either code loaded from the hub, or defined in any
        user-specific module/session."""
        return cls.is_remote_code() or not cls.__module__.startswith("iantirta.models.")



if __name__ == "__main__":
    from rich import inspect as i
    import time
    start_time = time.time()
    config = ConfigMixin()
    end_time = time.time()
    i(config)
    print(config)
    print("Initialized Within", end_time - start_time)

    config.register_for_auto_class()
    print(config._auto_class)