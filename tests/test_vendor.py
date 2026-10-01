# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

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


def test_vendor_module_paths():
    import huggingface_hub
    import transformers

    assert "iantirta/models/vendor/huggingface_hub" in (
        huggingface_hub.__file__.replace("\\", "/")
    )

    assert "iantirta/models/vendor/transformers" in (
        transformers.__file__.replace("\\", "/")
    )


def test_vendor_module_identity():
    import huggingface_hub
    import transformers

    assert huggingface_hub.__name__ == (
        "iantirta.models.vendor.huggingface_hub"
    )
    assert transformers.__name__ == (
        "iantirta.models.vendor.transformers"
    )


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

from iantirta.models.vendor.transformers import Wav2Vec2Config

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
