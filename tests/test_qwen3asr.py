# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from __future__ import annotations

import pytest
import torch


def test_qwen3_asr_import():
    from iantirta.models import (
        Qwen3ASRConfig,
        Qwen3ASREncoderConfig,
    )

    assert Qwen3ASRConfig.model_type == "qwen3_asr"
    assert Qwen3ASREncoderConfig.model_type == "qwen3_asr_encoder"


def test_qwen3_asr_config():
    from iantirta.models.vendor.transformers.models.qwen3_asr import (
        Qwen3ASRConfig,
    )

    config = Qwen3ASRConfig(
        audio_config={
            "model_type": "qwen3_asr_encoder",
            "encoder_layers": 1,
            "encoder_attention_heads": 4,
            "encoder_ffn_dim": 64,
            "d_model": 32,
        },
        text_config={
            "model_type": "qwen3",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 128,
        },
    )

    assert config.audio_config.model_type == "qwen3_asr_encoder"
    assert config.text_config.model_type == "qwen3"


def test_qwen3_asr_model_import():
    from iantirta.models.vendor.transformers.models.qwen3_asr import (
        Qwen3ASRForConditionalGeneration,
    )

    assert Qwen3ASRForConditionalGeneration is not None


def test_qwen3_asr_small_model():
    from iantirta.models.vendor.transformers.models.qwen3_asr import (
        Qwen3ASRConfig,
        Qwen3ASRForConditionalGeneration,
    )

    config = Qwen3ASRConfig(
        audio_config={
            "model_type": "qwen3_asr_encoder",
            "encoder_layers": 1,
            "encoder_attention_heads": 4,
            "encoder_ffn_dim": 64,
            "d_model": 32,
        },
        text_config={
            "model_type": "qwen3",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 128,
        },
    )

    model = Qwen3ASRForConditionalGeneration(config)

    assert model is not None


@pytest.mark.manual
def test_qwen3_asr_real_config():
    from iantirta.models.vendor.transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B-hf",
    )

    assert config.model_type == "qwen3_asr"


@pytest.mark.manual
def test_qwen3_asr_real_model_load():
    from iantirta.models.vendor.transformers import (
        Qwen3ASRForConditionalGeneration,
    )

    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B-hf",
    )

    assert model is not None
