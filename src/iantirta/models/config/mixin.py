# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.
#
# Partial code of huggingface_hub, improved by iantirta.com
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from functools import wraps
from dataclasses import dataclass, fields, MISSING, asdict
from typing_extensions import Self

from ..exceptions import YetToImplement
from ..files import cached_file

logger = logging.getLogger(__name__)


__all__ = [
    "ConfigMixin",
    "PretrainedOptions",
]


_FLOAT_TAG_KEY = "__float__"
_FLOAT_TAG_VALUES = {
    "Infinity": float("inf"),
    "-Infinity": float("-inf"),
    "NaN": float("nan")
}


@dataclass(slots=True)
class PretrainedOptions:
    cache_dir: str | Path | None = None
    force_download: bool = False
    local_files_only: bool = False
    token: str | bool | None = None
    revision: str = "main"

    proxies: str | None = None
    trust_remote_code: bool | None = None
    subfolder: str = ""

    gguf_file: str | None = None
    _configuration_file: str = "config.json"

    model_type: str | None = None

    @classmethod
    def from_dict(cls, options: dict) -> PretrainedOptions:
        return cls(**options)


class ConfigMixin:

    def __post_init__(self, **kwargs):
        if kwargs:
            raise RuntimeError(
                "Unused Kwargs:\n"
                f"{kwargs}"
            )

    @classmethod
    def from_pretrained(
        cls: type[Self],
        pretrained_model_name_or_path: str | Path,
        options: PretrainedOptions | dict | None = None,
        **kwargs
    ) -> type[Self]:
        raise NotImplementedError("use AutoConfig.from_pretrained instead")

    @classmethod
    def from_dict(
        cls: type[Self],
        config_dict: dict[str, Any],
        **kwargs,
    ) -> type[Self]:

        config = cls(**config_dict)
        logger.info(f"Model config {config}")
        return config

    @classmethod
    def get_config_dict(
        cls,
        pretrained_model_name_or_path: str | Path,
        options: PretrainedOptions,
    ) -> dict[str, Any]:
        config_dict = cls._get_config_dict(
            pretrained_model_name_or_path,
            options
        )

        if "configuration_files" in config_dict:
            raise YetToImplement(
                "Multiple configuration file is not supported"
            )

        return config_dict

    @classmethod
    def _get_config_dict(
        cls,
        pretrained_model_name_or_path: str | Path,
        options: PretrainedOptions,
    ) -> dict[str, Any]:
        if (Path(options.subfolder) / pretrained_model_name_or_path).is_file():
            resolved_config_file = Path(pretrained_model_name_or_path)
            is_local = True
        else:
            try:
                configuration_file = options._configuration_file
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    configuration_file,
                    **asdict(options)
                )
            except OSError:
                raise
            except Exception:
                raise OSError(
                    f"Can't load the configuration of '{pretrained_model_name_or_path}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the same"
                    f" name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory"
                    f" containing a {configuration_file} file\n"
                )

        try:
            if options.gguf_file:
                raise YetToImplement("gguf file is not supported")
            else:
                config_dict = cls._dict_from_json_file(
                    resolved_config_file
                )
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise OSError(
                "It looks like the config file at "
                f"'{resolved_config_file}' is not a "
                "valid JSON file."
            )

        if is_local:
            logger.info(f"loading configuration file {resolved_config_file}")
        else:
            logger.info(f"loading configuration file {configuration_file} from cache at {resolved_config_file}")

        if "pretrained_cfg" in config_dict:
            # This is a timm wrapper
            raise YetToImplement("Timm model not yet supported")

        if (
            options.model_type is not None and
            (
                config_dict["model_type"]
                != options.model_type
            )
        ):
            logger.warning(
                f"{configuration_file} has "
                f"'model_type={config_dict['model_type']}' "
                "but you overrode it with "
                f"'model_type={options.model_type}'. "
                "This may lead to unexpected behavior."
            )
            config_dict["model_type"] = options.model_type

        return config_dict

    @classmethod
    def _dict_from_json_file(cls, json_file: Path):
        with json_file.open("r", encoding="utf-8") as reader:
            config_dict = json.load(reader)

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


# copied from huggingface_hub.dataclasses.strict when `accept_kwargs=True`
def wrap_init_to_accept_kwargs(cls: dataclass):
    # Get the original dataclass-generated __init__
    original_init = cls.__init__

    @wraps(original_init)
    def __init__(self, *args, **kwargs: Any) -> None:
        # Extract only the fields that are part of the dataclass
        dataclass_fields = {f.name for f in fields(cls)}
        standard_kwargs = {k: v for k, v in kwargs.items() if k in dataclass_fields}

        # We need to call bare `__init__` without `__post_init__` but the `original_init` of
        # any dataclas contains a call to post-init at the end (without kwargs)
        if len(args) > 0:
            raise ValueError(
                f"{cls.__name__} accepts only keyword arguments, but found `{len(args)}` positional args."
            )

        for f in fields(cls):  # type: ignore
            if f.name in standard_kwargs:
                setattr(self, f.name, standard_kwargs[f.name])
            elif f.default is not MISSING:
                setattr(self, f.name, f.default)
            elif f.default_factory is not MISSING:
                setattr(self, f.name, f.default_factory())
            else:
                raise TypeError(f"Missing required field - '{f.name}'")

        # Pass any additional kwargs to `__post_init__` and let the object
        # decide whether to set the attr or use for different purposes (e.g. BC checks)
        additional_kwargs = {}
        for name, value in kwargs.items():
            if name not in dataclass_fields:
                additional_kwargs[name] = value

        self.__post_init__(**additional_kwargs)

    cls.__init__ = __init__
    return cls
