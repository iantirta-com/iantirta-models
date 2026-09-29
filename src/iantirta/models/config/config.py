# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.
#
# Partial code of huggingface_hub, improved by iantirta.com
from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import dataclass_transform

from .mixin import ConfigMixin, wrap_init_to_accept_kwargs
from ..vendor.huggingface_hub.dataclasses import strict


__all__ = [
    "ModelConfig",
]


@dataclass_transform(kw_only_default=True)
@strict(accept_kwargs=True)
@dataclass(repr=False)
class ModelConfig(ConfigMixin):
    # Attributes set internally when saving and used to infer model
    # class for `Auto` mapping
    model_type: ClassVar[str] = ""
    architectures: list[str] | None = None

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls = dataclass(cls, repr=False, kw_only=True)

        if "__init__" not in cls.__dict__:
            cls = wrap_init_to_accept_kwargs(cls)
