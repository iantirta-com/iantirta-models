# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.
#
# Partial code of huggingface_hub, improved by iantirta.com
from __future__ import annotations

import inspect
import sys
from typing import get_type_hints

from iantirta.models.config import PreTrainedConfig

from .mixin import ModelMixin


class Model(ModelMixin):
    # General model properties
    config_class: type[PreTrainedConfig] | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        child_annotation = inspect.get_annotations(cls).get("config", None)
        child_attribute = cls.__dict__.get("config_class", None)

        full_annotation = get_type_hints(cls).get("config", None)
        full_attribute = cls.config_class

        if child_attribute is not None:
            cls.config_class = child_attribute
        elif child_annotation is not None:
            cls.config_class = child_annotation
        elif full_attribute is not None:
            cls.config_class = full_attribute
        elif full_annotation is not None:
            cls.config_class = full_annotation

        if isinstance(cls.config_class, str):
            module = sys.modules[cls.__module__]
            cls.config_class = getattr(module, cls.config_class)
    