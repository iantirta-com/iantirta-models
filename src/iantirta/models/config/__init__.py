# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.
#
# Partial code of huggingface_hub, improved by iantirta.com
from __future__ import annotations

from .auto import AutoConfig
from .config import ModelConfig

# For transformers vendor
PretrainedConfig = ModelConfig
PreTrainedConfig = PretrainedConfig


__all__ = [
    "AutoConfig",
    "PreTrainedConfig",
    "PretrainedConfig",
]
