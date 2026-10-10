
import importlib
import os
from collections import OrderedDict

from ..config import PreTrainedConfig



CONFIG_MAPPING_NAMES = OrderedDict(
    [
        ("qwen3", "Qwen3Config"),
        ("qwen3_asr", "Qwen3ASRConfig"),
        ("qwen3_asr_encoder", "Qwen3ASREncoderConfig"),
        
        ("wav2vec2", "Wav2Vec2Config"),
    ]
)

SPECIAL_MODEL_TYPE_TO_MODULE_NAME = OrderedDict(
    [
        ("qwen3_asr_encoder", "qwen3_asr"),
    ]
)

FEATURE_EXTRACTOR_MAPPING_NAMES = OrderedDict(
    [
        ("qwen3_asr", "Qwen3ASRFeatureExtractor"),
        ("wav2vec2", "Wav2Vec2FeatureExtractor"),
    ]
)

PROCESSOR_MAPPING_NAMES = OrderedDict(
    [
        ("qwen3_asr", "Qwen3ASRProcessor"),
        ("wav2vec2", "Wav2Vec2Processor"),
    ]
)


def model_type_to_module_name(key) -> str:
    """Converts a config key to the corresponding module."""
    # Special treatment
    if key in SPECIAL_MODEL_TYPE_TO_MODULE_NAME:
        key = SPECIAL_MODEL_TYPE_TO_MODULE_NAME[key]
        return key

    key = key.replace("-", "_")
    return key


class _LazyConfigMap(OrderedDict[str, type]):
    def __init__(self, mapping) -> None:
        self._mapping = mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key: str) -> type[PreTrainedConfig]:
        if key in self._extra_content:
            return self._extra_content[key]

        if key not in self._mapping:
            raise KeyError(key)

        value = self._mapping[key]
        module_name = model_type_to_module_name(key)

        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(
                f".{module_name}", "iantirta.models.models"
            )

        if hasattr(self._modules[module_name], value):
            return getattr(self._modules[module_name], value)

        # Some of the mappings have entries
        # model_type -> config of another model type.
        # In that case we try to grab the
        # object at the top level.
        iantirta_models_module = importlib.import_module(
            "iantirta.models"
        )
        return getattr(iantirta_models_module, value)


CONFIG_MAPPING = _LazyConfigMap(CONFIG_MAPPING_NAMES)


class AutoConfig:
    """ Auto Config Class. """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        **kwargs
    ):
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        code_revision = kwargs.pop("code_revision", None)
        # resolved revision can be used to cached?
        config_dict, unused_kwargs = (
            PreTrainedConfig.get_config_dict(
                pretrained_model_name_or_path,
                **kwargs
            )
        )

        if model_type := config_dict.get("model_type", False):
            try:
                config_class = CONFIG_MAPPING[model_type]
            except KeyError:
                raise ValueError(
                    f"Unknown model_type: '{model_type}'"
                )

            return config_class.from_dict(config_dict, **unused_kwargs)

        raise ValueError(
            f"Unrecognized model in {pretrained_model_name_or_path}. "
            f"Should have a `model_type` key in its 'config.json'."
        )


__all__ = [
    "CONFIG_MAPPING",
    "AutoConfig",
]
