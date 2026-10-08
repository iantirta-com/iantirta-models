# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from __future__ import annotations

import pytest
import torch



def test_wav2vec2_import():
    from iantirta.models import (
        Wav2Vec2Config,
    )

    assert Wav2Vec2Config.model_type == "wav2vec2"


def test_wav2vec2_config():
    from iantirta.models.models.wav2vec2 import (
        Wav2Vec2Config,
    )

    config = Wav2Vec2Config()

    assert config.model_type == "wav2vec2"


def test_wav2vec2_model_import():
    from iantirta.models.models.wav2vec2 import (
        Wav2Vec2ForCTC,
    )

    assert Wav2Vec2ForCTC is not None


def test_wav2vec2_small_model():
    from iantirta.models.models.wav2vec2 import (
        Wav2Vec2ForCTC,
        Wav2Vec2Config,
    )

    config = Wav2Vec2Config()

    model = Wav2Vec2ForCTC(config)

    assert model is not None


@pytest.mark.manual
def test_wav2vec2_real_model_load():
    from iantirta.models import (
        Wav2Vec2Config,
        Wav2Vec2Model,
    )

    config = Wav2Vec2Config.from_pretrained(
        "facebook/wav2vec2-base"
    )

    model = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base",
    )

    assert model.config.model_type == "wav2vec2"
    assert model.config.hidden_size > 0


@pytest.mark.manual
def test_wav2vec2_real_forward():
    from iantirta.models import Wav2Vec2Model

    model = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base",
    )

    model.eval()

    audio = torch.randn(1, 16000)

    with torch.inference_mode():
        output = model(audio)

    assert output.last_hidden_state.ndim == 3
    assert output.last_hidden_state.shape[0] == 1


@pytest.mark.manual
def test_wav2vec2_checkpoint_keys():
    from iantirta.models import Wav2Vec2Model

    model = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base",
    )

    state = model.state_dict()

    assert any(
        key.startswith("feature_extractor.")
        for key in state
    )

    assert any(
        key.startswith("feature_projection.")
        for key in state
    )

    assert any(
        key.startswith("encoder.")
        for key in state
    )

