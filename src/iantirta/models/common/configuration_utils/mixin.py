
from dataclasses import dataclass
import os

@dataclass
class ConfigMixin:

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        **kwargs
    ):
        raise NotImplementedError

    @classmethod
    def get_config_dict()