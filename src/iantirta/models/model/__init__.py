# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.
#
# Partial code of huggingface_hub, improved by iantirta.com

from __future__ import annotations

from .model import Model

# For transformers vendor
PretrainedModel = Model
PreTrainedModel = PretrainedModel

__all__ = [
    "Model",
    "PreTrainedModel",
    "PretrainedModel",
]