# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.
#
# Partial code of huggingface_hub, improved by iantirta.com
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar
from typing_extensions import dataclass_transform

from .mixin import ConfigMixin, wrap_init_to_accept_kwargs
from ..vendor.huggingface_hub.dataclasses import strict


if TYPE_CHECKING:
    import torch

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
    transformers_version: str | None = None

    # Common attributes for all models
    dtype: str | torch.dtype | None = None

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        # Check first, order matter
        cls_has_custom_init = "__init__" in cls.__dict__

        cls = dataclass(cls, repr=False, kw_only=True)

        if not cls_has_custom_init:
            cls = wrap_init_to_accept_kwargs(cls)
