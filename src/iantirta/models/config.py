from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
import typing as t

import importlib

from .files import ensure_file


logger = logging.getLogger(__name__)

_FLOAT_TAG_KEY = "__float__"
_FLOAT_TAG_VALUES = {
    "Infinity": float("inf"),
    "-Infinity": float("-inf"),
    "NaN": float("nan"),
}


class ModelGenerationConfig:
    @staticmethod
    def _get_default_generation_params() -> dict[str, Any]:
        """
        Defaults to be applied when unset by the model OR by the user, such that `model.generate()` works with minimal
        parameterization.

        Pretrained checkpoints should set these as appropriate in their `generation_config.json`, to establish
        a better default baseline. Be mindful that tests will often use these values.
        """
        return {
            "max_length": 20,
            "min_length": 0,
            "do_sample": False,
            "use_cache": True,
            "early_stopping": False,
            "num_beams": 1,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "typical_p": 1.0,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "encoder_no_repeat_ngram_size": 0,
            "bad_words_ids": None,
            "num_return_sequences": 1,
            "output_scores": False,
            "return_dict_in_generate": False,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "remove_invalid_values": False,
            "exponential_decay_length_penalty": None,
            "suppress_tokens": None,
            "begin_suppress_tokens": None,
            "epsilon_cutoff": 0.0,
            "eta_cutoff": 0.0,
            "encoder_repetition_penalty": 1.0,
            "num_assistant_tokens": 20,
            "num_assistant_tokens_schedule": "constant",
            "assistant_confidence_threshold": 0.4,
            "assistant_lookbehind": 10,
            "target_lookbehind": 10,
            # Deprecated arguments (moved to the Hub). TODO joao, manuel: remove in v4.62.0
            "num_beam_groups": 1,
            "diversity_penalty": 0.0,
        }


@dataclass
class ModelConfig:
    """
    """
    # Class attributes that we don't want to save or have in `self.__dict__`
    # They are not supposed to be set/changed by users. Each field is set when
    # creating a model class
    base_config_key: t.ClassVar[str] = ""
    sub_configs: t.ClassVar[dict[str, type[ModelConfig]]] = {}
    
    # Attributes set internally when saving and used to infer model
    # class for `Auto` mapping
    model_type: str | None = None
    architectures: list[str] | None = None

    # Common attributes for all models
    dtype: str | torch.dtype | None = None

    _extra: dict[str, Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    
    def post_init_hook(self): 
        if (torch_dtype := self._extra.pop("torch_dtype", None)) is not None:
            # legacy, still being used
            self.dtype = self.dtype if self.dtype is not None else torch_dtype
           
        if self.dtype is not None and isinstance(self.dtype, str):
            import torch
            self.dtype = getattr(torch, self.dtype)

        # Parameters for sequence generation saved in the config are popped instead of loading them.
        for name in ModelGenerationConfig._get_default_generation_params().keys():
            self._extra.pop(name, None)
        
        # Name or path to the pretrained checkpoint
        self._name_or_path = str(self._extra.pop("name_or_path", ""))
        # BC: configs saved by older versions may still carry this key, it is not used anymore. The revision of a
        # repository is now resolved once per load and passed around as `revision` (see `utils.hub.resolve_revision`).
        self._extra.pop("_commit_hash", None)

        self.validate()

    @property
    def name_or_path(self) -> str | None:
        return getattr(self, "_name_or_path", None)

    @name_or_path.setter
    def name_or_path(self, value):
        self._name_or_path = str(value)  # Make sure that name_or_path is a string (for JSON encoding)


    def validate(self) -> None:
        pass
        
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> ModelConfig:
        known = {f.name for f in fields(cls)}

        kwargs = {}
        extra = {}
    
        for key, value in config_dict.items():
            if key in known:
                kwargs[key] = value
            else:
                extra[key] = value
    
        config = cls(**kwargs)
        config._extra = extra
        config.post_init_hook()
        
        logger.info(f"Model config {config}")
        return config
        
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
    ) -> ModelConfig:
        config_dict = cls.get_config_dict(
            pretrained_model_name_or_path
        )
        # AutoMapping
        if "model_type" in config_dict:
            try:
                module = importlib.import_module("iantirta.models.vendor." + config_dict["model_type"])
                config_class = module._config_class
            except ImportError:
                logger.warning(f"No vendor available for {config_dict['model_type']}")
                config_class = cls
        else:
            config_class = cls
        return config_class.from_dict(config_dict)

    @classmethod
    def get_config_dict(
        cls,
        pretrained_model_name_or_path: str | Path,
    ) -> dict[str, Any]:
        config_file = ensure_file(
            pretrained_model_name_or_path,
            "config.json",
        )

        config_dict = cls._dict_from_json(config_file)

        # TODO: support external configuration_files later.
        if "configuration_files" in config_dict:
            raise NotImplementedError("TODO.")
        return config_dict

    @classmethod
    def _dict_from_json(cls, json_file: Path) -> dict[str, Any]:
        with json_file.open("r", encoding="utf-8") as reader:
            config_dict = json.load(reader)

        return cls._decode_special_floats(config_dict)

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
