from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from torch import nn

from iantirta.models.config import ModelConfig
from iantirta.models.model import Model


@dataclass
class TestConfig(ModelConfig):
    model_type: str = "test"
    input_size: int = 4
    hidden_size: int = 8
    output_size: int = 2


class TestModel(Model):
    config_class = TestConfig

    def __init__(self, config: TestConfig):
        super().__init__(config)

        self.network = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.output_size),
        )

    def forward(self, x):
        return self.network(x)


def test_config_from_dict():
    config = TestConfig.from_dict({
        "model_type": "test",
        "input_size": 4,
        "hidden_size": 8,
        "output_size": 2,
    })

    assert config.model_type == "test"
    assert config.input_size == 4
    assert config.hidden_size == 8
    assert config.output_size == 2


def test_model_from_pretrained(tmp_path):
    config = TestConfig()

    original_model = TestModel(config)

    # Save config.
    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_type": "test",
            "input_size": 4,
            "hidden_size": 8,
            "output_size": 2,
        }),
        encoding="utf-8",
    )

    # Save weights.
    torch.save(
        original_model.state_dict(),
        tmp_path / "pytorch_model.bin",
    )

    # Load through our API.
    loaded_model = TestModel.from_pretrained(tmp_path)

    assert loaded_model.config == config

    for name, tensor in original_model.state_dict().items():
        assert torch.equal(
            tensor,
            loaded_model.state_dict()[name],
        )


def test_model_produces_same_output(tmp_path):
    config = TestConfig()

    original_model = TestModel(config)
    original_model.eval()

    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_type": "test",
            "input_size": 4,
            "hidden_size": 8,
            "output_size": 2,
        }),
        encoding="utf-8",
    )

    torch.save(
        original_model.state_dict(),
        tmp_path / "pytorch_model.bin",
    )

    loaded_model = TestModel.from_pretrained(tmp_path)
    loaded_model.eval()

    x = torch.randn(3, 4)

    with torch.no_grad():
        expected = original_model(x)
        actual = loaded_model(x)

    assert torch.equal(expected, actual)

def test_model_from_pretrained_safetensors(tmp_path):
    from safetensors.torch import save_file

    config = TestConfig()

    original_model = TestModel(config)

    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_type": "test",
            "input_size": 4,
            "hidden_size": 8,
            "output_size": 2,
        }),
        encoding="utf-8",
    )

    save_file(
        original_model.state_dict(),
        tmp_path / "model.safetensors",
    )

    loaded_model = TestModel.from_pretrained(tmp_path)

    for name, tensor in original_model.state_dict().items():
        assert torch.equal(
            tensor,
            loaded_model.state_dict()[name],
        )

def test_config_from_pretrained_real_mms():
    mms_repo = "facebook/mms-1b-all"
    config = ModelConfig.from_pretrained(mms_repo)

    assert config
    from pprint import pprint
    pprint("="*20)
    pprint(config)
    pprint("="*20)

def test_config_from_pretrained_real_qwen():
    mms_repo = "Qwen/Qwen3-ASR-1.7B-hf"
    config = ModelConfig.from_pretrained(mms_repo)

    assert config
    from pprint import pprint
    pprint("="*20)
    pprint(config)
    pprint("="*20)

def test_model_from_pretrained_real_mms():
    mms_repo = "facebook/mms-1b-all"
    model = Model.from_pretrained(mms_repo)

    assert model
    from pprint import pprint
    pprint("="*20)
    pprint(model)
    pprint(model.config)
    pprint("="*20)