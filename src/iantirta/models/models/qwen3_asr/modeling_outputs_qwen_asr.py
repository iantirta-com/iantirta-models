from dataclasses import dataclass

import torch

from iantirta.models.common.cache_utils import Cache
from iantirta.models.common.modeling_outputs import ModelOutputMixin


@dataclass
class Qwen3ASRCausalLMOutputWithPast(ModelOutputMixin):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    audio_hidden_states: torch.FloatTensor | None = None
