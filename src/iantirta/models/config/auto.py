# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.
#
# Partial code of huggingface_hub, improved by iantirta.com
from __future__ import annotations

import importlib
from pathlib import Path
import logging
from .config import ModelConfig
from .mixin import PretrainedOptions
from ..exceptions import YetToImplement
from typing_extensions import Self

logger = logging.getLogger(__name__)


__all__ = [
    "AutoConfig",
]


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

            try:
                module_path = (
                    "iantirta.models.vendor.transformers."
                    + config_dict["model_type"]
                )
                module = importlib.import_module(module_path)
            except ImportError as err:
                logger.warning(
                    "No vendor available for "
                    f"{module_path}\n"
                    f"  Error: {err}"
                )
                raise err
            if len(config_dict["architectures"]) > 1:
                raise YetToImplement(
                    "Multiple architectures config not supported"
                )
            try:
                return (
                    getattr(
                        module,
                        config_dict["architectures"][0]
                    )
                    .config_class
                    .from_dict(config_dict)
                )
            except Exception:
                raise

        raise ValueError(
            "Unrecognized model in "
            f"{pretrained_model_name_or_path}. "
            "Should have a `model_type` key in "
            f"its {options._configuration_file}."
        )
