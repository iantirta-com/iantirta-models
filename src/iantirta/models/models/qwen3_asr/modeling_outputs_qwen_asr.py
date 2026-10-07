from dataclasses import dataclass

import torch

from iantirta.models.cache.mixin import Cache
from iantirta.models.core.outputs.mixin import ModelOutput


@dataclass
class Qwen3ASRCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    audio_hidden_states: torch.FloatTensor | None = None
