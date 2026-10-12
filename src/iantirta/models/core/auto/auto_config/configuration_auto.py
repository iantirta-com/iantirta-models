
import os

from typing import TYPE_CHECKING

from .configuration_auto_utils import CONFIG_MAPPING, logger

if TYPE_CHECKING:
    from iantirta.models import PreTrainedConfig


__all__ = [
    "AutoConfig",
    "CONFIG_MAPPING"
]


class AutoConfig:
    r"""
    This is a generic configuration class that will be instantiated as
    one of the configuration classes of the library
    when created with the [`~AutoConfig.from_pretrained`] class method.

    This class cannot be instantiated directly
    using `__init__()` (throws an error).
    """

    def __init__(self) -> None:
        raise OSError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.from_pretrained"
            "(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def for_model(cls, model_type: str, *args, **kwargs) -> "PreTrainedConfig":
        if model_type in CONFIG_MAPPING:
            config_class = CONFIG_MAPPING[model_type]
            return config_class(*args, **kwargs)
        raise ValueError(
            f"Unrecognized model identifier: {model_type}. "
            f"Should contain one of {', '.join(CONFIG_MAPPING.keys())}"
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        **kwargs
    ):
        r"""
        Instantiate one of the configuration classes of the library from a pretrained model configuration.

        The configuration class to instantiate is selected based on the `model_type` property of the config object that
        is loaded, or when it's missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

        List options

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                    - A string, the *model id* of a pretrained model configuration hosted inside a model repo on
                      huggingface.co.
                    - A path to a *directory* containing a configuration file saved using the
                      [`~PreTrainedConfig.save_pretrained`] method, or the [`~PreTrainedModel.save_pretrained`] method,
                      e.g., `./my_model_directory/`.
                    - a path to a saved configuration JSON *file*, e.g.,
                      `./my_model_directory/configuration.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final configuration object.

                If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused_kwargs* is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
                part of `kwargs` which has not been used to update `config` and is otherwise ignored.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            kwargs(additional keyword arguments, *optional*):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.

        Examples:

        ```python
        >>> from transformers import AutoConfig

        >>> # Download configuration from huggingface.co and cache.
        >>> config = AutoConfig.from_pretrained("google-bert/bert-base-uncased")

        >>> # Download configuration from huggingface.co (user-uploaded) and cache.
        >>> config = AutoConfig.from_pretrained("dbmdz/bert-base-german-cased")

        >>> # If configuration file is in a directory (e.g., was saved using *save_pretrained('./test/saved_model/')*).
        >>> config = AutoConfig.from_pretrained("./test/bert_saved_model/")

        >>> # Load a specific configuration file.
        >>> config = AutoConfig.from_pretrained("./test/bert_saved_model/my_configuration.json")

        >>> # Change some config attributes when loading a pretrained config.
        >>> config = AutoConfig.from_pretrained("google-bert/bert-base-uncased", output_attentions=True, foo=False)
        >>> config.output_attentions
        True

        >>> config, unused_kwargs = AutoConfig.from_pretrained(
        ...     "google-bert/bert-base-uncased", output_attentions=True, foo=False, return_unused_kwargs=True
        ... )
        >>> config.output_attentions
        True

        >>> unused_kwargs
        {'foo': False}
        ```
        """
        kwargs["_from_auto"] = True
        kwargs["name_or_path"] = pretrained_model_name_or_path
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        code_revision = kwargs.pop("code_revision", None)

        # Resolve the revision once, so the config and the
        # remote code below come from the same repository state.
        from iantirta.models import PreTrainedConfig
        from iantirta.models.remote.hf_api import resolve_revision

        kwargs["revision"] = resolve_revision(
            pretrained_model_name_or_path,
            kwargs.get("revision"),
            token=kwargs.get("token"),
            local_files_only=kwargs.get("local_files_only", False),
            cache_dir=kwargs.get("cache_dir"),
        )

        # Actual Download here
        config_dict, unused_kwargs = PreTrainedConfig.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )
        has_remote_code = (
            "auto_map" in config_dict
            and "AutoConfig" in config_dict["auto_map"]
        )
        has_local_code = (
            "model_type" in config_dict
            and config_dict["model_type"] in CONFIG_MAPPING
        )
        explicit_local_code = (
            has_local_code
            and not CONFIG_MAPPING[
                config_dict["model_type"]
                ].__module__.startswith("transformers.")
        )
        if has_remote_code:
            raise NotImplementedError("Using remote code is not supported")

        if has_remote_code and trust_remote_code and not explicit_local_code:
            raise NotImplementedError('Using remote code is not supported')

        elif "model_type" in config_dict:
            # Apply heuristic: if model_type is mistral but
            # layer_types is present, treat as ministral
            if (
                config_dict["model_type"] == "mistral"
                and "layer_types" in config_dict
            ):
                logger.info(
                    "Detected mistral model with layer_types, treating "
                    "as ministral for alternating attention compatibility. "
                )
                config_dict["model_type"] = "ministral"

            try:
                config_class = CONFIG_MAPPING[config_dict["model_type"]]
            except KeyError:
                raise ValueError(
                    f"The checkpoint you are trying to load has model type `{config_dict['model_type']}` "
                    "but Transformers does not recognize this architecture. This could be because of an "
                    "issue with the checkpoint, or because your version of Transformers is out of date.\n\n"
                    "You can update Transformers with the command `pip install --upgrade transformers`. If this "
                    "does not work, and the checkpoint is very new, then there may not be a release version "
                    "that supports this model yet. In this case, you can get the most up-to-date code by installing "
                    "Transformers from source with the command "
                    "`pip install git+https://github.com/huggingface/transformers.git`"
                )
            return config_class.from_dict(config_dict, **unused_kwargs)

        from iantirta.models import t_constant
        raise ValueError(
            f"Unrecognized model in {pretrained_model_name_or_path}. "
            f"Should have a `model_type` key in its {t_constant.CONFIG_NAME}."
        )

    @staticmethod
    def register(model_type, config, exist_ok=False) -> None:
        """
        Register a new configuration for this class.

        Args:
            model_type (`str`): The model type like "bert" or "gpt".
            config ([`PreTrainedConfig`]): The config to register.
        """
        from iantirta.models import PreTrainedConfig

        if issubclass(config, PreTrainedConfig) and config.model_type != model_type:
            raise ValueError(
                "The config you are passing has a `model_type` attribute that is not consistent with the model type "
                f"you passed (config has {config.model_type} and you passed {model_type}. Fix one of those so they "
                "match!"
            )
        CONFIG_MAPPING.register(model_type, config, exist_ok=exist_ok)


if __name__ == "__main__":
    from rich import inspect
    from pprint import pprint

    repo = "Qwen/Qwen3-ASR-1.7B-hf"
    
    config = AutoConfig.from_pretrained(repo)
    inspect(config)
    pprint(config)

    from transformers import AutoConfig as TAutoConfig
    config2 = TAutoConfig.from_pretrained(repo)
    inspect(config2)
