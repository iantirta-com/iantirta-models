# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Auto Model class."""

import copy
import importlib
import json
import logging
import os
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any, TypeVar

from iantirta.models.core.config.pretrained import (
    CONFIG_NAME,
    PreTrainedConfig,
)
from iantirta.models.common.modeling_utils.integrations.peft import (
    find_adapter_config_file,
    is_peft_available,
)
from iantirta.models.tools._torch import is_torch_available

from .auto_config import CONFIG_MAPPING_NAMES, AutoConfig, model_type_to_module_name

if is_torch_available():
    from iantirta.models.common.generation_utils.mixin import GenerationMixin

logger = logging.getLogger(__name__)


_T = TypeVar("_T")
# Tokenizers will depend on packages installed, too much variance and there are no common base or Protocol
_LazyAutoMappingValue = tuple[type[Any] | None, type[Any] | None]


MODEL_MAPPING_NAMES = OrderedDict(
    [
        # Base model mapping
        ("qwen3", "Qwen3Model"),
        ("qwen3_asr", "Qwen3ASRModel"),
        ("qwen3_asr_encoder", "Qwen3ASREncoder"),
        ("wav2vec2", "Wav2Vec2Model"),
    ]
)

# Models that accept text and optionally multimodal data in inputs
# and can generate text and optionally multimodal data.
MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES = OrderedDict(
    [
        ("qwen3_asr", "Qwen3ASRForConditionalGeneration"),
    ]
)

MODEL_FOR_CTC_MAPPING_NAMES = OrderedDict(
    [
        # Model for Connectionist temporal classification (CTC) mapping
        ("wav2vec2", "Wav2Vec2ForCTC"),
    ]
)


def _get_model_class(config, model_mapping):
    supported_models = model_mapping[type(config)]
    if not isinstance(supported_models, (list, tuple)):
        return supported_models

    name_to_model = {model.__name__: model for model in supported_models}
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in name_to_model:
            return name_to_model[arch]

    # If not architecture is set in the config or match the supported models, the first element of the tuple is the
    # defaults.
    return supported_models[0]


def add_generation_mixin_to_remote_model(model_class):
    """
    Adds `GenerationMixin` to the inheritance of `model_class`, if `model_class` is a PyTorch model.

    This function is used for backwards compatibility purposes: in v4.45, we've started a deprecation cycle to make
    `PreTrainedModel` stop inheriting from `GenerationMixin`. Without this function, older models dynamically loaded
    from the Hub may not have the `generate` method after we remove the inheritance.
    """
    # 1. If it is not a PT model (i.e. doesn't inherit Module), do nothing
    if "torch.nn.modules.module.Module" not in str(model_class.__mro__):
        return model_class

    # 2. If it already **directly** inherits from GenerationMixin, do nothing
    if "GenerationMixin" in str(model_class.__bases__):
        return model_class

    # 3. Prior to v4.45, we could detect whether a model was `generate`-compatible if it had its own `generate` and/or
    # `prepare_inputs_for_generation` method.
    has_custom_generate_in_class = hasattr(model_class, "generate") and "GenerationMixin" not in str(
        model_class.generate
    )
    has_custom_prepare_inputs = hasattr(model_class, "prepare_inputs_for_generation") and "GenerationMixin" not in str(
        model_class.prepare_inputs_for_generation
    )
    if has_custom_generate_in_class or has_custom_prepare_inputs:
        model_class_with_generation_mixin = type(
            model_class.__name__, (model_class, GenerationMixin), {**model_class.__dict__}
        )
        return model_class_with_generation_mixin
    return model_class


class _BaseAutoModelClass:
    # Base class for auto models.
    _model_mapping = None

    def __init__(self, *args, **kwargs) -> None:
        raise OSError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
        has_local_code = type(config) in cls._model_mapping
        explicit_local_code = has_local_code and not _get_model_class(
            config, cls._model_mapping
        ).__module__.startswith("transformers.")
        if has_remote_code:
            class_ref = config.auto_map[cls.__name__]
            if "--" in class_ref:
                upstream_repo = class_ref.split("--")[0]
            else:
                upstream_repo = None  # noqa: F841
            # trust_remote_code = resolve_trust_remote_code(
            #     trust_remote_code, config._name_or_path, has_local_code, has_remote_code, upstream_repo=upstream_repo
            # )

        if has_remote_code and trust_remote_code and not explicit_local_code:
            from ...dynamic_module_utils import (  # type: ignore
                get_class_from_dynamic_module,
            )
            if "--" in class_ref:
                repo_id, class_ref = class_ref.split("--")
            else:
                repo_id = config.name_or_path
            model_class = get_class_from_dynamic_module(class_ref, repo_id, **kwargs)
            cls.register(config.__class__, model_class, exist_ok=True)
            model_class.register_for_auto_class(auto_class=cls)
            _ = kwargs.pop("code_revision", None)
            model_class = add_generation_mixin_to_remote_model(model_class)
            return model_class._from_config(config, **kwargs)
        elif has_local_code:
            model_class = _get_model_class(config, cls._model_mapping)
            text_config_class = config.sub_configs.get("text_config", None)
            # getattr avoids AttributeError, as registered remote-code model classes may lack config_class
            if text_config_class is not None and getattr(model_class, "config_class", None) == text_config_class:
                # TODO: Validate that copying the parent quantization config to the text sub-config preserves
                # modules_to_not_convert and skip-module matching when composite-model module prefixes differ.
                parent_config = config
                config = config.get_text_config()
                # Check both `quantization_config` being present and also not null,
                # as a `config.json` can have `"quantization_config": null` in it
                parent_quant = getattr(parent_config, "quantization_config", None)
                if parent_quant is not None:
                    config.quantization_config = parent_quant
            return model_class._from_config(config, **kwargs)

        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping)}."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], *model_args, **kwargs):
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.get("trust_remote_code")
        kwargs["_from_auto"] = True
        hub_kwargs_names = [
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "revision",
            "subfolder",
            "token",
        ]
        hub_kwargs = {name: kwargs.pop(name) for name in hub_kwargs_names if name in kwargs}
        code_revision = kwargs.pop("code_revision", None)
        kwargs.pop("_commit_hash", None)  # BC: not used anymore, `revision` is resolved to a commit hash instead
        adapter_kwargs = kwargs.pop("adapter_kwargs", None)

        token = hub_kwargs.pop("token", None)

        if token is not None:
            hub_kwargs["token"] = token

        # Resolve the revision once: the adapter config, the model config, the remote code and the weights are then all
        # read from the exact same repository state, without any further call to the Hub to revalidate it.
        requested_revision = hub_kwargs.get("revision")
        # hub_kwargs["revision"] = resolve_revision(
        #     pretrained_model_name_or_path,
        #     requested_revision,
        #     token=token,
        #     local_files_only=hub_kwargs.get("local_files_only", False),
        #     cache_dir=hub_kwargs.get("cache_dir"),
        # )

        if is_peft_available():
            if adapter_kwargs is None:
                adapter_kwargs = {}
            adapter_kwargs = adapter_kwargs.copy()  # avoid mutating original
            if token is not None:
                adapter_kwargs["token"] = token

            adapter_kwargs.setdefault("revision", hub_kwargs["revision"])
            maybe_adapter_path = find_adapter_config_file(pretrained_model_name_or_path, **adapter_kwargs)

            if maybe_adapter_path is not None:
                with open(maybe_adapter_path, "r", encoding="utf-8") as f:
                    adapter_config = json.load(f)

                    adapter_kwargs["_adapter_model_path"] = pretrained_model_name_or_path
                    # Only override the model name/path if the current value doesn't point to a
                    # complete model with an embedded adapter so that local models with embedded
                    # adapters will load from the local base model rather than pull the base
                    # model named in the adapter's config from the hub.
                    if not os.path.exists(pretrained_model_name_or_path) or not os.path.exists(
                        os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
                    ):
                        pretrained_model_name_or_path = adapter_config["base_model_name_or_path"]
                        # The commit hash we resolved above belongs to the adapter repository, so resolve again
                        # against the base model repository we have just been pointed at.
                        # hub_kwargs["revision"] = resolve_revision(
                        #     pretrained_model_name_or_path,
                        #     requested_revision,
                        #     token=token,
                        #     local_files_only=hub_kwargs.get("local_files_only", False),
                        #     cache_dir=hub_kwargs.get("cache_dir"),
                        # )
                        hub_kwargs["revision"] = requested_revision

        if not isinstance(config, PreTrainedConfig):
            kwargs_orig = copy.deepcopy(kwargs)
            # ensure not to pollute the config object with dtype="auto" - since it's
            # meaningless in the context of the config object - torch.dtype values are acceptable
            if kwargs.get("torch_dtype") == "auto":
                _ = kwargs.pop("torch_dtype")
            if kwargs.get("dtype") == "auto":
                _ = kwargs.pop("dtype")
            # to not overwrite the quantization_config if config has a quantization_config
            if kwargs.get("quantization_config") is not None:
                _ = kwargs.pop("quantization_config")

            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                code_revision=code_revision,
                **hub_kwargs,
                **kwargs,
            )

            # A concrete dtype is absorbed into the config above and then dropped at the composite
            # `get_text_config()` swap, so re-inject the user's value as an explicit kwarg to force the model's
            # `from_pretrained` to honor it over the config's saved dtype (#46459).
            if kwargs_orig.get("torch_dtype", None) is not None:
                kwargs["torch_dtype"] = kwargs_orig["torch_dtype"]
            if kwargs_orig.get("dtype", None) is not None:
                kwargs["dtype"] = kwargs_orig["dtype"]
            if kwargs_orig.get("quantization_config", None) is not None:
                kwargs["quantization_config"] = kwargs_orig["quantization_config"]

        has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
        has_local_code = type(config) in cls._model_mapping
        explicit_local_code = has_local_code and not _get_model_class(
            config, cls._model_mapping
        ).__module__.startswith("iantirta.models.")
        upstream_repo = None
        if has_remote_code:
            class_ref = config.auto_map[cls.__name__]
            if "--" in class_ref:
                upstream_repo = class_ref.split("--")[0]  # noqa: F841
        # trust_remote_code = resolve_trust_remote_code(
        #     trust_remote_code,
        #     pretrained_model_name_or_path,
        #     has_local_code,
        #     has_remote_code,
        #     upstream_repo=upstream_repo,
        # )
        # kwargs["trust_remote_code"] = trust_remote_code

        # Set the adapter kwargs
        kwargs["adapter_kwargs"] = adapter_kwargs

        if has_remote_code and trust_remote_code and not explicit_local_code:
            from ...dynamic_module_utils import (  # type: ignore
                get_class_from_dynamic_module,
            )
            model_class = get_class_from_dynamic_module(
                class_ref, pretrained_model_name_or_path, code_revision=code_revision, **hub_kwargs, **kwargs
            )
            _ = hub_kwargs.pop("code_revision", None)
            cls.register(config.__class__, model_class, exist_ok=True)
            model_class.register_for_auto_class(auto_class=cls)
            model_class = add_generation_mixin_to_remote_model(model_class)
            return model_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
            )
        elif has_local_code:
            model_class = _get_model_class(config, cls._model_mapping)
            text_config_class = config.sub_configs.get("text_config", None)
            # getattr avoids AttributeError, as registered remote-code model classes may lack config_class
            if text_config_class is not None and getattr(model_class, "config_class", None) == text_config_class:
                # TODO: Validate that copying the parent quantization config to the text sub-config preserves
                # modules_to_not_convert and skip-module matching when composite-model module prefixes differ.
                parent_config = config
                config = config.get_text_config()
                # Check both `quantization_config` being present and also not null,
                # as a `config.json` can have `"quantization_config": null` in it
                parent_quant = getattr(parent_config, "quantization_config", None)
                if parent_quant is not None:
                    config.quantization_config = parent_quant
            return model_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
            )
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping)}."
        )


def getattribute_from_module(module, attr):
    if attr is None:
        return None
    if isinstance(attr, tuple):
        return tuple(getattribute_from_module(module, a) for a in attr)
    if isinstance(attr, dict):
        return {k: getattribute_from_module(module, v) for k, v in attr.items()}
    if hasattr(module, attr):
        return getattr(module, attr)
    # Some of the mappings have entries model_type -> object of another model type. In that case we try to grab the
    # object at the top level.
    transformers_module = importlib.import_module("iantirta.models")

    if module != transformers_module:
        try:
            return getattribute_from_module(transformers_module, attr)
        except ValueError:
            raise ValueError(f"Could not find {attr} neither in {module} nor in {transformers_module}!")
    else:
        raise ValueError(f"Could not find {attr} in {transformers_module}!")


class _LazyAutoMapping(OrderedDict[type[PreTrainedConfig], _LazyAutoMappingValue]):
    """
    A mapping config to object (model or tokenizer for instance) that will load keys and values when it is accessed.

    Args:
        - config_mapping: The map model type to config class
        - model_mapping: The map model type to model (or tokenizer) class
    """

    def __init__(self, config_mapping, model_mapping) -> None:
        self._config_mapping = config_mapping
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        self._model_mapping = model_mapping
        self._model_mapping._model_mapping = self
        self._extra_content = {}
        self._modules = {}

    def __len__(self) -> int:
        common_keys = set(self._config_mapping.keys()).intersection(self._model_mapping.keys())
        return len(common_keys) + len(self._extra_content)

    def __getitem__(self, key: type[PreTrainedConfig]) -> _LazyAutoMappingValue:
        if key in self._extra_content:
            return self._extra_content[key]
        model_type = self._reverse_config_mapping[key.__name__]
        if model_type in self._model_mapping:
            model_name = self._model_mapping[model_type]
            return self._load_attr_from_module(model_type, model_name)

        # Maybe there was several model types associated with this config.
        model_types = [k for k, v in self._config_mapping.items() if v == key.__name__]
        for mtype in model_types:
            if mtype in self._model_mapping:
                model_name = self._model_mapping[mtype]
                return self._load_attr_from_module(mtype, model_name)
        raise KeyError(key)

    def _load_attr_from_module(self, model_type, attr):
        module_name = model_type_to_module_name(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "iantirta.models.models")
        return getattribute_from_module(self._modules[module_name], attr)

    def keys(self) -> list[type[PreTrainedConfig]]:
        mapping_keys = [
            self._load_attr_from_module(key, name)
            for key, name in self._config_mapping.items()
            if key in self._model_mapping
        ]
        return mapping_keys + list(self._extra_content.keys())

    def get(self, key: type[PreTrainedConfig], default: _T) -> _LazyAutoMappingValue | _T:
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __bool__(self) -> bool:
        return bool(self.keys())

    def values(self) -> list[_LazyAutoMappingValue]:
        mapping_values = [
            self._load_attr_from_module(key, name)
            for key, name in self._model_mapping.items()
            if key in self._config_mapping
        ]
        return mapping_values + list(self._extra_content.values())

    def items(self) -> list[tuple[type[PreTrainedConfig], _LazyAutoMappingValue]]:
        mapping_items = [
            (
                self._load_attr_from_module(key, self._config_mapping[key]),
                self._load_attr_from_module(key, self._model_mapping[key]),
            )
            for key in self._model_mapping
            if key in self._config_mapping
        ]
        return mapping_items + list(self._extra_content.items())

    def __iter__(self) -> Iterator[type[PreTrainedConfig]]:
        return iter(self.keys())

    def __contains__(self, item: type) -> bool:
        if item in self._extra_content:
            return True
        if not hasattr(item, "__name__") or item.__name__ not in self._reverse_config_mapping:
            return False
        model_type = self._reverse_config_mapping[item.__name__]
        return model_type in self._model_mapping

    def register(self, key: type[PreTrainedConfig] | str, value: _LazyAutoMappingValue, exist_ok=False) -> None:
        """
        Register a new model in this mapping.
        """
        if hasattr(key, "__name__") and key.__name__ in self._reverse_config_mapping:
            model_type = self._reverse_config_mapping[key.__name__]
            if model_type in self._model_mapping and not exist_ok:
                raise ValueError(f"'{key}' is already used by a Transformers model.")

        # Some remote code may simply register a new custom model/processor/..., while using a native Transformers config. In such
        # cases, we should skip registering, as we will otherwise always remap the native config to the custom model/processor/... in
        # the same session, even if `trust_remote_code=False` is specified by the user (in which case we should use the native
        # Transformers model/processor/... corresponding to the config)
        # This is because remote/native is indistinguisable from the config class only in such cases, as they both use the same class - then
        # `from_pretrained`/`from_config` are responsible to grab the correct class depending on whether `trust_remote_code` is True/False
        if getattr(key, "__module__", "").startswith("transformers."):
            return

        # Register the new mapping (this will always take precedence in __getattr__ and __contains__ compared to base mapping)
        self._extra_content[key] = value

    def __reduce__(self):
        return (
            self.__class__._from_pickle,
            (self._config_mapping, self._model_mapping, dict(self._extra_content)),
        )

    @classmethod
    def _from_pickle(cls, config_mapping, model_mapping, extra_content):
        obj = cls(config_mapping, model_mapping)
        obj._extra_content = extra_content
        return obj


MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES)
MODEL_FOR_MULTIMODAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES)
MODEL_FOR_CTC_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CTC_MAPPING_NAMES)


class AutoModel(_BaseAutoModelClass):
    _model_mapping = MODEL_MAPPING


class AutoModelForMultimodalLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MULTIMODAL_LM_MAPPING


class AutoModelForCTC(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CTC_MAPPING


__all__ = [
    "MODEL_FOR_CTC_MAPPING",
    "MODEL_FOR_MULTIMODAL_LM_MAPPING",
    "MODEL_MAPPING",
    "AutoModel",
    "AutoModelForCTC",
    "AutoModelForMultimodalLM",
]
