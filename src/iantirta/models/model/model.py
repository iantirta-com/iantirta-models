# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.
#
# Partial code of huggingface_hub, improved by iantirta.com
from __future__ import annotations

from .mixin import ModelMixin


class Model(ModelMixin):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
