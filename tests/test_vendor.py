from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
VENDOR = SRC / "iantirta" / "models" / "vendor"


def test_vendor_aliases():
    import iantirta.models

    import huggingface_hub
    assert huggingface_hub is iantirta.models.vendor.huggingface_hub
    
    import transformers
    assert transformers is iantirta.models.vendor.transformers
    
    assert huggingface_hub is iantirta.models.vendor.huggingface_hub
    assert transformers is iantirta.models.vendor.transformers


def test_vendor_paths():
    import huggingface_hub
    import transformers

    assert "iantirta/models/vendor/huggingface_hub" in huggingface_hub.__file__
    assert "iantirta/models/vendor/transformers" in transformers.__file__


def test_vendor_hub_imports():
    from iantirta.models.vendor.huggingface_hub import (
        HfApi,
        hf_hub_download,
        snapshot_download,
    )


def test_vendor_hub_lazy_imports():
    from iantirta.models.vendor.huggingface_hub import (
        ModelInfo,
        RepoCard,
        HfFileSystem,
    )


def test_vendor_without_external_hf_packages():
    env = os.environ.copy()

    code = """
import iantirta.models

import huggingface_hub
import transformers

assert "iantirta.models.vendor.huggingface_hub" in huggingface_hub.__name__
assert "iantirta.models.vendor.transformers" in transformers.__name__

from transformers import Wav2Vec2Config

config = Wav2Vec2Config()
assert config.model_type == "wav2vec2"
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr


def test_vendor_transformers_wav2vec2_import():
    from iantirta.models.vendor.transformers.models.wav2vec2 import (
        Wav2Vec2Config,
        Wav2Vec2Model,
    )


def test_vendor_wav2vec2_config():
    from iantirta.models.vendor.transformers import Wav2Vec2Config

    config = Wav2Vec2Config()

    assert config is not None
    assert config.model_type == "wav2vec2"


def test_vendor_wav2vec2_model():
    import torch

    from iantirta.models.vendor.transformers import (
        Wav2Vec2Config,
        Wav2Vec2Model,
    )

    config = Wav2Vec2Config(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
    )

    model = Wav2Vec2Model(config)

    x = torch.randn(1, 1600)

    output = model(x)

    assert output.last_hidden_state.ndim == 3
