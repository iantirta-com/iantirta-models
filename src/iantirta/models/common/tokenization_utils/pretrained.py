
from abc import abstractmethod

import torch

from ..modeling_utils import PreTrainedModel


class PreTrainedAudioTokenizerBase(PreTrainedModel):
    """
    Class that additionally defines the behavior of any `audio_tokenizer` to be added.
    Characteristic for any of them:
        1. Encode raw audio into discrete audio codebooks (with x channels)
        2. Decode from discrete audio codebooks back to raw audio
    It is possible that they can decode in different ways given a different representation
    but they are forced to support 2. nonetheless, e.g. see `DAC`.
    """

    @abstractmethod
    def encode(self, input_values: torch.Tensor, *args, **kwargs):
        """
        Encode raw audio retrieved from a respective `FeatureExtractor` into discrete audio codebooks (with x channels)
        """

    @abstractmethod
    def decode(self, audio_codes: torch.Tensor, *args, **kwargs):
        """Decode from discrete audio codebooks back to raw audio"""
