# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.
#
# Partial code of huggingface_hub, improved by iantirta.com
from __future__ import annotations

import importlib
import logging
from pathlib import Path

from typing_extensions import Self

from ..exceptions import YetToImplement
from .config import ModelConfig
from .mixin import PretrainedOptions

logger = logging.getLogger(__name__)


__all__ = [
    "AutoConfig",
]

def get_auto_config(
        *,
        config: ModelConfig | None = None,
        config_dict: dict | None = None
) -> type[ModelConfig]:
    if (config_dict is None) ^ (config is not None):
        raise ValueError("This function must take exactly one of `config_dict` or `config`")
    from ..model.auto import get_auto_model
    if config is not None:
        model_class = get_auto_model(config=config)
    elif config_dict is not None:
        model_class = get_auto_model(config_dict=config_dict)
    try:
        return model_class.config_class
    except Exception:  # noqa: TRY203
        raise


class AutoConfig:
    def __init__(self):
        raise RuntimeError("Wasnt Supposed to be initialize")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        options: PretrainedOptions | dict | None = None,
        **kwargs
    ) -> type[Self]:
        if not isinstance(options, PretrainedOptions):
            if options and isinstance(options, dict):
                options = PretrainedOptions.from_dict(options)
            elif kwargs and isinstance(kwargs, dict):
                options = PretrainedOptions.from_dict(kwargs)
            else:
                options = PretrainedOptions()

        config_dict = ModelConfig.get_config_dict(
            pretrained_model_name_or_path,
            options,
        )
        if "auto_map" in config_dict:
            raise YetToImplement("remote code model is not supported")

        if "model_type" in config_dict:
            if config_dict["model_type"] == "mistral":
                raise YetToImplement("mistral model is not supported")
            return get_auto_config(config_dict=config_dict).from_dict(config_dict)
        raise ValueError(
            "Unrecognized model in "
            f"{pretrained_model_name_or_path}. "
            "Should have a `model_type` key in "
            f"its {options._configuration_file}."
        )
