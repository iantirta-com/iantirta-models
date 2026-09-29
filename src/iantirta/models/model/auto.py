# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.
#
# Partial code of huggingface_hub, improved by iantirta.com
from __future__ import annotations

import importlib

from pathlib import Path
from typing_extensions import Self
from dataclasses import asdict
import logging

from .mixin import ModelPretrainedOptions
from ..config import PretrainedConfig, AutoConfig
from ..exceptions import YetToImplement

logger = logging.getLogger(__name__)


class AutoModel:
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *model_args,
        options: ModelPretrainedOptions | dict | None = None,
        **kwargs
    ) -> type[Self]:
        if not isinstance(options, ModelPretrainedOptions):
            if options and isinstance(options, dict):
                options = ModelPretrainedOptions.from_dict(options)
            elif kwargs and isinstance(kwargs, dict):
                options = ModelPretrainedOptions.from_dict(kwargs)

        # peft

        if not isinstance(
            options.config,
            PretrainedConfig
        ):
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                **asdict(options)
            )

        try:
            module_path = (
                "iantirta.models.vendor.transformers."
                + config.model_type
            )
            module = importlib.import_module(module_path)
        except ImportError as err:
            logger.warning(
                "No vendor available for "
                f"{module_path}\n"
                f"  Error: {err}"
            )
            raise err
        try:
            model_class = getattr(
                module,
                config.architectures[0]
            )
            text_config_class = config.sub_configs.get("text_config", None)
            if (
                text_config_class is not None
                and (
                    getattr(model_class, "config_class", None)
                    == text_config_class
                )
            ):
                raise YetToImplement(
                    "text config class same as currenr config class"
                )
            return model_class.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                options=options,
            )
        except Exception:
            raise
        raise ValueError(
            "Unrecognized configuration class "
            f"{config.__class__} for this kind of "
            f"AutoModel: {cls.__name__}.\n"
        )
