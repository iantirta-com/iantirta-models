# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from iantirta.models.vendor.transformers import AutoConfig, PreTrainedConfig


# @dataclass
class TestConfig(PreTrainedConfig):
    model_type: str = "test_type"


@pytest.mark.parametrize(
    "config_dict",
    [
        {
            "architectures": ["test_arch"],
            "torch_dtype": torch.float32
        },
        {
            "architectures": ["test_arch"],
            "dtype": "float32"
        }
    ]
)
def test_config_from_dict(config_dict):
    config = PreTrainedConfig.from_dict(config_dict)
    assert isinstance(config, PreTrainedConfig)
    for key in config_dict:
        assert getattr(config, key, None) is not None


@pytest.mark.manual
@pytest.mark.parametrize(
    "pretrained_name",
    [
        "facebook/mms-1b-all",
        "Qwen/Qwen3-ASR-1.7B-hf",
    ]
)
def test_config_from_pretrained(pretrained_name):
    config = AutoConfig.from_pretrained(pretrained_name)
    assert config
