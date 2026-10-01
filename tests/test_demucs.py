# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from fractions import Fraction

import torch

from iantirta.models.demucs.modeling_utils import (
    DemucsBagOfModel,
    DemucsConfig,
    _decode_json,
    _unflatten_state,
    hf_repo_name,
)


def test_hf_repo_name():
    assert hf_repo_name("htdemucs") == "HTDemucs"
    assert hf_repo_name("htdemucs_ft") == "HTDemucs-ft"
    assert hf_repo_name("mdx_extra_q") == "Demucs-mdx_extra_q"
    assert hf_repo_name("mdx_q") == "Demucs-mdx_q"


def test_decode_json_fraction():
    value = {
        "segment": {
            "_type": "fraction",
            "numerator": 1,
            "denominator": 2,
        }
    }

    assert _decode_json(value) == {
        "segment": Fraction(1, 2),
    }


def test_decode_json_nested():
    value = {
        "a": [
            {
                "_type": "fraction",
                "numerator": 3,
                "denominator": 4,
            }
        ]
    }

    assert _decode_json(value) == {
        "a": [Fraction(3, 4)],
    }




def test_unflatten_state():
    tensors = {
        "a": torch.tensor([1.0]),
        "b": torch.tensor([2.0]),
    }

    structure = {
        "_dict": [
            ("first", {"_tensor": "a"}),
            (
                "nested",
                {
                    "_list": [
                        {"_tensor": "b"},
                    ]
                },
            ),
        ]
    }

    result = _unflatten_state(tensors, structure)

    assert torch.equal(result["first"], tensors["a"])
    assert torch.equal(result["nested"][0], tensors["b"])


def test_demucs_config_yaml(tmp_path):
    path = tmp_path / "config.yaml"

    path.write_text(
        """
name_or_path: test
segment: 7
models:
  - model_a
  - model_b
weights:
  - [1.0, 2.0]
  - [0.5, 0.5]
"""
    )

    config = DemucsConfig.from_pretrained(path)

    assert config.segment == 7
    assert config.bag_models == ["model_a", "model_b"]
    assert config.weights == [
        [1.0, 2.0],
        [0.5, 0.5],
    ]


def test_htdemucs_load():
    model = DemucsBagOfModel.from_pretrained(
        "htdemucs_ft",
    )

    assert len(model.models) > 0
    assert model.sources
    assert model.samplerate > 0
    assert model.audio_channels > 0


def test_demucs_model_sources():
    model = DemucsBagOfModel.from_pretrained("htdemucs")

    assert model.sources == [
        "drums",
        "bass",
        "other",
        "vocals",
    ]


def test_mdx_extra_q_forward():
    from iantirta.models.demucs import apply_model

    model = DemucsBagOfModel.from_pretrained("mdx_extra_q")

    x = torch.randn(1, 2, 44100 * 5)

    model.eval()

    with torch.inference_mode():
        result = apply_model(model, x)

    assert result.ndim == 4
    assert result.shape[0] == 1
